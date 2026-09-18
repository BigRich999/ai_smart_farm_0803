"""Qwen3-VL 통합 클라이언트 (로컬 Transformers / OpenAI 호환 API)."""

from __future__ import annotations

import base64
import tempfile
import threading
from pathlib import Path
from typing import Any

from config import (
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_API_MODEL,
    QWEN_BACKEND,
    QWEN_DEVICE_MAP,
    QWEN_MAX_NEW_TOKENS,
    QWEN_QUANTIZATION,
    QWEN_VL_MODEL,
)

_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None
# Streamlit 재실행 / 연속 질의 시 generate 중복 진입 방지
_GENERATE_LOCK = threading.Lock()
# 고해상도 원본(4K급)이 VRAM을 과도하게 잡아 2회차부터 멈추는 것 방지
# 병반 식별을 위해 너무 작게 줄이지 않음 (VRAM과 정확도 절충)
_MAX_IMAGE_SIDE = 1536


def resolve_backend() -> str:
    """사용할 백엔드를 결정한다."""
    if QWEN_BACKEND in {"local", "api"}:
        return QWEN_BACKEND
    # auto
    if QWEN_API_BASE and QWEN_API_KEY:
        return "api"
    if QWEN_API_KEY and not QWEN_API_BASE:
        # DashScope 등 키가만 있는 경우 기본 엔드포인트 가정하지 않고 local 우선
        # Ollama처럼 base만 있는 경우도 있음
        pass
    if QWEN_API_BASE:
        return "api"
    return "local"


def _guess_mime(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def _resize_image_file(src: Path, dest: Path, max_side: int = _MAX_IMAGE_SIDE) -> None:
    """긴 변을 max_side 이하로 줄여 VRAM 사용량을 낮춘다."""
    from PIL import Image

    with Image.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        img.save(dest, format="JPEG", quality=90)


def _to_temp_image(image_source: str | Path | bytes, filename: str = "upload.jpg") -> Path:
    """로컬 추론용으로 리사이즈된 임시 이미지 경로를 확보한다."""
    suffix = ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    tmp.close()

    if isinstance(image_source, (bytes, bytearray)):
        raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".jpg")
        raw_tmp.write(bytes(image_source))
        raw_tmp.flush()
        raw_tmp.close()
        raw_path = Path(raw_tmp.name)
        try:
            _resize_image_file(raw_path, tmp_path)
        finally:
            try:
                raw_path.unlink(missing_ok=True)
            except OSError:
                pass
        return tmp_path

    src = Path(image_source)
    _resize_image_file(src, tmp_path)
    return tmp_path


def _model_primary_device(model: Any) -> Any:
    """device_map=auto 환경에서도 안전한 primary device를 반환한다."""
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _clear_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _encode_data_url(image_source: str | Path | bytes, filename: str = "upload.jpg") -> str:
    if isinstance(image_source, (bytes, bytearray)):
        raw = bytes(image_source)
        mime = _guess_mime(filename)
    else:
        path = Path(image_source)
        raw = path.read_bytes()
        mime = _guess_mime(path.name)
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _select_quantization() -> str:
    """auto면 VRAM에 따라 4bit/none을 고른다."""
    mode = QWEN_QUANTIZATION
    if mode != "auto":
        return mode
    try:
        import torch

        if not torch.cuda.is_available():
            return "none"
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        if vram_gb < 10:
            return "4bit"
        return "none"
    except Exception:
        return "none"


def _load_local_model() -> tuple[Any, Any]:
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR

    import torch
    from transformers import AutoProcessor

    quant = _select_quantization()
    model_kwargs: dict[str, Any] = {
        "device_map": QWEN_DEVICE_MAP,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }

    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Qwen3-VL 클래스명이 환경/버전에 따라 다를 수 있어 순차 시도
    model = None
    last_err: Exception | None = None
    for loader_name in (
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
    ):
        try:
            import transformers as tf

            cls = getattr(tf, loader_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(QWEN_VL_MODEL, **model_kwargs)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            model = None

    if model is None:
        raise RuntimeError(
            f"로컬 Qwen3-VL 로드 실패. 모델={QWEN_VL_MODEL}, 원인={last_err}. "
            "QWEN_API_BASE/QWEN_API_KEY로 API 백엔드를 사용하거나 "
            "transformers/토치/VRAM을 확인하세요."
        )

    processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL)
    model.eval()
    _LOCAL_MODEL = model
    _LOCAL_PROCESSOR = processor
    return model, processor


def _generate_local(
    *,
    system_prompt: str,
    user_text: str,
    image_source: str | Path | bytes | None = None,
    filename: str = "upload.jpg",
    max_new_tokens: int | None = None,
) -> str:
    import torch

    # 연속 질의에서 generate가 겹치거나 VRAM이 고갈되면 무한 대기처럼 보임
    if not _GENERATE_LOCK.acquire(timeout=600):
        raise TimeoutError("이전 Qwen3-VL 생성이 아직 끝나지 않았습니다. 잠시 후 다시 시도하세요.")

    tmp_path: Path | None = None
    inputs = None
    output_ids = None
    try:
        model, processor = _load_local_model()
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if image_source is not None:
            tmp_path = _to_temp_image(image_source, filename)
            content.insert(0, {"type": "image", "image": str(tmp_path)})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # qwen_vl_utils가 있으면 공식 전처리 사용
        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if tmp_path is not None:
                from PIL import Image

                image = Image.open(tmp_path).convert("RGB")
                inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            else:
                inputs = processor(text=[text], return_tensors="pt", padding=True)

        device = _model_primary_device(model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in dict(inputs).items()
            }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs["input_ids"]
            trimmed = [out[len(inp) :] for inp, out in zip(input_ids, output_ids)]
            text_out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return (text_out or "").strip()
    finally:
        # 텐서/캐시를 비워 두 번째 질의부터 VRAM 고갈로 멈추는 현상 완화
        try:
            del inputs
        except Exception:
            pass
        try:
            del output_ids
        except Exception:
            pass
        _clear_cuda()
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        _GENERATE_LOCK.release()

def _generate_api(
    *,
    system_prompt: str,
    user_text: str,
    image_source: str | Path | bytes | None = None,
    filename: str = "upload.jpg",
    max_new_tokens: int | None = None,
) -> str:
    from openai import OpenAI

    if not QWEN_API_BASE:
        raise RuntimeError(
            "API 백엔드인데 QWEN_API_BASE가 없습니다. "
            "예: https://dashscope.aliyuncs.com/compatible-mode/v1 또는 http://localhost:11434/v1"
        )

    client = OpenAI(base_url=QWEN_API_BASE, api_key=QWEN_API_KEY or "EMPTY")
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_source is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_data_url(image_source, filename)},
            }
        )

    response = client.chat.completions.create(
        model=QWEN_API_MODEL,
        temperature=0.2,
        max_tokens=max_new_tokens or QWEN_MAX_NEW_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def qwen_generate(
    *,
    system_prompt: str,
    user_text: str,
    image_source: str | Path | bytes | None = None,
    filename: str = "upload.jpg",
    max_new_tokens: int | None = None,
) -> str:
    """텍스트(+선택 이미지)를 Qwen3-VL로 생성한다."""
    backend = resolve_backend()
    if backend == "api":
        return _generate_api(
            system_prompt=system_prompt,
            user_text=user_text,
            image_source=image_source,
            filename=filename,
            max_new_tokens=max_new_tokens,
        )
    return _generate_local(
        system_prompt=system_prompt,
        user_text=user_text,
        image_source=image_source,
        filename=filename,
        max_new_tokens=max_new_tokens,
    )


def describe_runtime() -> dict[str, Any]:
    """UI/디버그용 런타임 정보."""
    info: dict[str, Any] = {
        "backend": resolve_backend(),
        "model": QWEN_VL_MODEL if resolve_backend() == "local" else QWEN_API_MODEL,
        "api_base": QWEN_API_BASE or "-",
        "quantization": _select_quantization() if resolve_backend() == "local" else "-",
    }
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    except Exception:
        info["cuda"] = False
    return info
