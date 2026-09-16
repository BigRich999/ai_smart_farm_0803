"""
병해충 이미지 분석 Streamlit 앱

사용법:
  1) 노트북에서 학습해 best_model.pth 를 이 폴더(또는 지정 경로)에 둡니다.
  2) pip install -r requirements_app.txt
  3) python -m streamlit run pest_predict_app.py

주의:
  torch / torchaudio 버전이 어긋나면 transformers import 시
  libtorchaudio 로드 오류가 납니다. 이 앱은 오디오가 필요 없으므로
  문제가 있으면 다음으로 제거하세요.
    pip uninstall -y torchaudio
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


def _import_transformers():
    """깨진 torchaudio 때문에 transformers가 같이 실패하는 경우를 안내."""
    try:
        from transformers import ViTConfig, ViTModel
    except OSError as e:
        raise ImportError(
            "transformers 로드 중 네이티브 라이브러리 오류가 발생했습니다.\n"
            "대개 torch와 버전이 다른 torchaudio 때문입니다. 아래를 실행하세요:\n"
            "  pip uninstall -y torchaudio\n"
            f"원인: {e}"
        ) from e

    try:
        from transformers import AutoImageProcessor as ImageProcessorCls
    except ImportError:
        try:
            from transformers import ViTImageProcessor as ImageProcessorCls
        except ImportError as e:
            raise ImportError(
                "transformers에서 이미지 프로세서를 불러올 수 없습니다.\n"
                '  pip install -U "transformers>=4.40.0" torchvision pillow\n'
                f"원인: {e}"
            ) from e
    return ViTConfig, ViTModel, ImageProcessorCls


ViTConfig, ViTModel, ImageProcessorCls = _import_transformers()

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = APP_DIR / "best_model.pth"
DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"


class PestAnalysisModel(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: bool = False,
        num_crops: int = 8,
        num_parts: int = 3,
        num_diseases: int = 18,
    ):
        super().__init__()
        self.model_name = model_name
        if pretrained:
            self.backbone = ViTModel.from_pretrained(model_name)
        else:
            self.backbone = ViTModel(ViTConfig.from_pretrained(model_name))
        hidden = self.backbone.config.hidden_size
        self.fc_crop = nn.Linear(hidden, num_crops)
        self.fc_part = nn.Linear(hidden, num_parts)
        self.fc_disease = nn.Linear(hidden, num_diseases)
        self.fc_bbox = nn.Linear(hidden, 4)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            feat = outputs.pooler_output
        else:
            feat = outputs.last_hidden_state[:, 0]
        out_c = self.fc_crop(feat)
        out_p = self.fc_part(feat)
        out_d = self.fc_disease(feat)
        out_b = torch.sigmoid(self.fc_bbox(feat))
        return out_c, out_p, out_d, out_b


def _normalize_id_map(raw: dict) -> dict[int, str]:
    """체크포인트의 id2* 키 타입(str/int)을 int로 통일."""
    return {int(k): str(v) for k, v in raw.items()}


@st.cache_resource(show_spinner="모델 로딩 중...")
def load_bundle(ckpt_path: str, device_str: str):
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")

    device = torch.device(device_str)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", DEFAULT_MODEL_NAME)
    num_crops = int(ckpt["num_crops"])
    num_parts = int(ckpt["num_parts"])
    num_diseases = int(ckpt["num_diseases"])
    id2crop = _normalize_id_map(ckpt["id2crop"])
    id2part = _normalize_id_map(ckpt["id2part"])
    id2disease = _normalize_id_map(ckpt["id2disease"])

    processor = ImageProcessorCls.from_pretrained(model_name)
    model = PestAnalysisModel(
        model_name=model_name,
        pretrained=False,
        num_crops=num_crops,
        num_parts=num_parts,
        num_diseases=num_diseases,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "model_name": model_name,
        "id2crop": id2crop,
        "id2part": id2part,
        "id2disease": id2disease,
        "image_size": ckpt.get("image_size", 224),
    }


def topk(logits: torch.Tensor, id2name: dict[int, str], k: int = 3):
    prob = torch.softmax(logits, dim=-1)[0]
    k = min(k, prob.numel())
    vals, idxs = torch.topk(prob, k=k)
    return [(id2name[int(i)], float(v) * 100.0) for v, i in zip(vals, idxs)]


@torch.inference_mode()
def predict(image: Image.Image, bundle: dict, top_k: int = 3):
    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]

    rgb = image.convert("RGB")
    w, h = rgb.size
    inputs = processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    out_c, out_p, out_d, out_b = model(pixel_values)
    x, y, bw, bh = out_b[0].float().cpu().tolist()
    bbox = {
        "x": x * w,
        "y": y * h,
        "w": bw * w,
        "h": bh * h,
    }
    return {
        "crop": topk(out_c, bundle["id2crop"], top_k),
        "part": topk(out_p, bundle["id2part"], top_k),
        "disease": topk(out_d, bundle["id2disease"], top_k),
        "bbox": bbox,
        "image": rgb,
    }


def draw_bbox(image: Image.Image, bbox: dict, color: str = "red", width: int = 4) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    draw.rectangle([x, y, x + bw, y + bh], outline=color, width=width)
    return vis


def fmt_topk(items: list[tuple[str, float]]) -> str:
    lines = [f"{i}. **{name}** — {score:.1f}%" for i, (name, score) in enumerate(items, start=1)]
    return "\n".join(lines)


def main():
    st.set_page_config(
        page_title="병해충 이미지 분석",
        page_icon="🌿",
        layout="wide",
    )
    st.title("병해충 이미지 분석")
    st.caption("Hugging Face ViT + best_model.pth · 작물 / 부위 / 병해충 / bbox")

    with st.sidebar:
        st.header("설정")
        ckpt_path = st.text_input("체크포인트 경로", value=str(DEFAULT_CKPT))
        use_cuda = st.checkbox("GPU 사용 (CUDA)", value=torch.cuda.is_available())
        top_k = st.slider("Top-K", min_value=1, max_value=5, value=3)
        st.markdown("---")
        st.markdown(
            """
**사용 순서**
1. 노트북 학습으로 `best_model.pth` 생성
2. 위 경로가 올바른지 확인
3. 이미지 업로드 후 예측
"""
        )
        if st.button("모델 캐시 새로고침"):
            load_bundle.clear()
            st.success("캐시를 비웠습니다. 다시 예측하면 모델을 다시 로드합니다.")

    device_str = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if not Path(ckpt_path).is_file():
        st.warning(
            f"`best_model.pth`가 없습니다.\n\n"
            f"현재 경로: `{ckpt_path}`\n\n"
            "노트북에서 학습을 마친 뒤 이 폴더에 두거나, 사이드바에서 경로를 바꿔 주세요."
        )
        st.stop()

    try:
        bundle = load_bundle(ckpt_path, device_str)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

    st.sidebar.success(f"로드 완료 · {bundle['model_name']}")
    st.sidebar.write(f"device: `{device_str}`")
    st.sidebar.write(
        f"클래스: crop={len(bundle['id2crop'])}, "
        f"part={len(bundle['id2part'])}, "
        f"disease={len(bundle['id2disease'])}"
    )

    uploaded = st.file_uploader(
        "작물 이미지를 업로드하세요 (jpg / png)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("이미지를 올리면 작물·부위·병해충 Top-K와 bbox를 표시합니다.")
        st.stop()

    image = Image.open(uploaded)
    with st.spinner("예측 중..."):
        result = predict(image, bundle, top_k=top_k)

    vis = draw_bbox(result["image"], result["bbox"])
    crop1, part1, disease1 = result["crop"][0], result["part"][0], result["disease"][0]
    bbox = result["bbox"]

    col_img, col_out = st.columns([1.2, 1])
    with col_img:
        st.subheader("예측 bbox")
        st.image(vis, use_container_width=True)
        st.caption(
            f"bbox(pixel): x={bbox['x']:.1f}, y={bbox['y']:.1f}, "
            f"w={bbox['w']:.1f}, h={bbox['h']:.1f}"
        )

    with col_out:
        st.subheader("결과 요약")
        m1, m2, m3 = st.columns(3)
        m1.metric("작물", crop1[0], f"{crop1[1]:.1f}%")
        m2.metric("부위", part1[0], f"{part1[1]:.1f}%")
        m3.metric("병해충", disease1[0], f"{disease1[1]:.1f}%")

        st.markdown("#### 작물 Top-K")
        st.markdown(fmt_topk(result["crop"]))
        st.markdown("#### 부위 Top-K")
        st.markdown(fmt_topk(result["part"]))
        st.markdown("#### 병해충 Top-K")
        st.markdown(fmt_topk(result["disease"]))


if __name__ == "__main__":
    main()
