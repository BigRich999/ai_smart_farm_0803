"""
토마토 숙도 객체 탐지 Streamlit 앱 (YOLO26 best.pt)

사용법:
  1) tomato_yolo26_colab.ipynb 학습으로 models/tomato_yolo26_best.pt 생성
  2) pip install -r requirements_app.txt
  3) python -m streamlit run tomato_predict_app.py

주의:
  - PyTorch는 이미 설치된 CUDA 버전을 그대로 사용합니다. (이 앱이 torch를 재설치하지 않음)
  - torchvision이 CPU 빌드이면 CUDA NMS가 없어 TorchNMS로 대체합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = APP_DIR / "models" / "tomato_yolo26_best.pt"

CLASS_ID_TO_KO = {0: "미숙", 1: "완숙"}
CLASS_ID_TO_EN = {0: "unripe", 1: "ripe"}
CLASS_COLORS_BGR = {0: (46, 160, 46), 1: (45, 45, 210)}  # OpenCV BGR

HARVEST_HIGH = 0.70
HARVEST_MID = 0.40


def patch_cpu_torchvision_nms() -> bool:
    """CPU용 torchvision에는 CUDA NMS 커널이 없어 GPU 추론이 실패할 수 있음."""
    if not torch.cuda.is_available():
        return False
    try:
        import torchvision.ops as tv_ops
    except ImportError:
        return False

    try:
        boxes = torch.rand(2, 4, device="cuda")
        scores = torch.rand(2, device="cuda")
        tv_ops.nms(boxes, scores, 0.5)
        return False  # CUDA NMS 정상
    except Exception:
        pass

    from ultralytics.utils.nms import TorchNMS

    orig_nms = tv_ops.nms

    def nms_gpu_safe(boxes, scores, iou_threshold):
        if torch.is_tensor(boxes) and boxes.is_cuda:
            return TorchNMS.nms(boxes, scores, iou_threshold)
        return orig_nms(boxes, scores, iou_threshold)

    tv_ops.nms = nms_gpu_safe
    try:
        import torchvision.ops.boxes as tv_boxes

        tv_boxes.nms = nms_gpu_safe
    except Exception:
        pass
    return True


def harvest_decision(ripe_ratio: float | None) -> str:
    if ripe_ratio is None or (isinstance(ripe_ratio, float) and np.isnan(ripe_ratio)):
        return "객체 없음"
    if ripe_ratio >= HARVEST_HIGH:
        return "수확 권장"
    if ripe_ratio >= HARVEST_MID:
        return "부분 수확"
    return "생육 관찰"


@st.cache_resource(show_spinner="YOLO 모델 로딩 중...")
def load_model(ckpt_path: str, device_str: str):
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"가중치를 찾을 수 없습니다: {path}")

    nms_patched = patch_cpu_torchvision_nms()

    from ultralytics import YOLO

    model = YOLO(str(path))
    # 워밍업으로 device 확인
    device = 0 if device_str == "cuda" and torch.cuda.is_available() else "cpu"
    return {
        "model": model,
        "device": device,
        "device_str": "cuda:0" if device == 0 else "cpu",
        "names": dict(model.names) if model.names else CLASS_ID_TO_EN,
        "nms_patched": nms_patched,
        "ckpt": str(path.resolve()),
    }


def predict_image(image: Image.Image, bundle: dict, conf: float, iou: float, imgsz: int):
    model = bundle["model"]
    device = bundle["device"]
    rgb = image.convert("RGB")

    # PIL(RGB)로 넣어야 색이 맞습니다. np.ndarray는 OpenCV BGR로 해석됩니다.
    results = model.predict(
        source=rgb,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    r = results[0]
    # pil=True → RGB PIL Image. (기본 plot()은 BGR ndarray)
    annotated = r.plot(pil=True)
    if not isinstance(annotated, Image.Image):
        # 버전별 호환: ndarray면 BGR → RGB
        annotated = Image.fromarray(np.asarray(annotated)[:, :, ::-1])

    rows = []
    n_unripe = n_ripe = 0
    confs: list[float] = []

    if r.boxes is not None and len(r.boxes):
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for i, cid in enumerate(cls_ids):
            if cid == 0:
                n_unripe += 1
            elif cid == 1:
                n_ripe += 1
            confs.append(float(conf_arr[i]))
            rows.append(
                {
                    "class_id": int(cid),
                    "class_en": CLASS_ID_TO_EN.get(int(cid), str(int(cid))),
                    "class_ko": CLASS_ID_TO_KO.get(int(cid), str(int(cid))),
                    "confidence": float(conf_arr[i]),
                    "x1": float(xyxy[i, 0]),
                    "y1": float(xyxy[i, 1]),
                    "x2": float(xyxy[i, 2]),
                    "y2": float(xyxy[i, 3]),
                }
            )

    total = n_unripe + n_ripe
    ripe_ratio = (n_ripe / total) if total else float("nan")
    decision = harvest_decision(ripe_ratio)

    return {
        "annotated": annotated,
        "detections": pd.DataFrame(rows),
        "n_unripe": n_unripe,
        "n_ripe": n_ripe,
        "total": total,
        "ripe_ratio": ripe_ratio,
        "mean_conf": float(np.mean(confs)) if confs else float("nan"),
        "decision": decision,
        "original": rgb,
    }


def decision_color(decision: str) -> str:
    return {
        "수확 권장": "🟢",
        "부분 수확": "🟡",
        "생육 관찰": "🔵",
        "객체 없음": "⚪",
    }.get(decision, "⚪")


def main():
    st.set_page_config(
        page_title="토마토 숙도 작물 분석",
        page_icon="🍅",
        layout="wide",
    )
    st.title("토마토 숙도 작물 분석")
    st.caption("Ultralytics YOLO26 · models/tomato_yolo26_best.pt · 미숙/완숙 탐지 + 수확 판정")

    with st.sidebar:
        st.header("설정")
        ckpt_path = st.text_input("가중치 경로 (best.pt)", value=str(DEFAULT_CKPT))
        use_cuda = st.checkbox("GPU 사용 (CUDA)", value=torch.cuda.is_available())
        conf = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        iou = st.slider("IoU (NMS)", min_value=0.10, max_value=0.95, value=0.70, step=0.05)
        imgsz = st.select_slider("imgsz", options=[320, 416, 512, 640, 800], value=640)
        st.markdown("---")
        st.markdown(
            f"""
**수확 판정 기준**
- 완숙비율 ≥ {HARVEST_HIGH:.0%} → 수확 권장
- {HARVEST_MID:.0%} ~ {HARVEST_HIGH:.0%} → 부분 수확
- < {HARVEST_MID:.0%} → 생육 관찰

**사용 순서**
1. 노트북에서 `models/tomato_yolo26_best.pt` 생성
2. 위 경로 확인 후 이미지 업로드
"""
        )
        if st.button("모델 캐시 새로고침"):
            load_model.clear()
            st.success("캐시를 비웠습니다. 다시 예측하면 모델을 다시 로드합니다.")

    device_str = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if not Path(ckpt_path).is_file():
        st.warning(
            f"`best.pt`가 없습니다.\n\n"
            f"현재 경로: `{ckpt_path}`\n\n"
            "노트북 학습 후 `models/tomato_yolo26_best.pt`를 두거나, 사이드바에서 경로를 바꿔 주세요."
        )
        st.stop()

    try:
        bundle = load_model(ckpt_path, device_str)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.exception(e)
        st.stop()

    st.sidebar.success("모델 로드 완료")
    st.sidebar.write(f"device: `{bundle['device_str']}`")
    st.sidebar.write(f"클래스: `{bundle['names']}`")
    st.sidebar.write(f"PyTorch: `{torch.__version__}`")
    if bundle["nms_patched"]:
        st.sidebar.info("torchvision CPU 빌드 → TorchNMS 대체 적용")

    uploaded = st.file_uploader(
        "토마토 이미지를 업로드하세요 (jpg / jpeg / png)",
        type=["jpg", "jpeg", "png"],
    )
    sample_dir = APP_DIR / "yolo_tomato" / "images" / "test"
    sample_files = sorted(sample_dir.glob("*.*")) if sample_dir.is_dir() else []

    use_sample = False
    sample_path = None
    if sample_files:
        with st.expander("또는 test 샘플 이미지 선택", expanded=uploaded is None):
            names = [p.name for p in sample_files]
            choice = st.selectbox("test 이미지", options=["(선택 안 함)"] + names)
            if choice != "(선택 안 함)":
                sample_path = sample_dir / choice
                use_sample = True

    if uploaded is None and not use_sample:
        st.info("이미지를 올리면 미숙/완숙 박스를 그리고 수확 판정을 표시합니다.")
        st.stop()

    if uploaded is not None:
        image = Image.open(uploaded)
        source_name = uploaded.name
    else:
        image = Image.open(sample_path)
        source_name = sample_path.name

    with st.spinner("예측 중..."):
        result = predict_image(image, bundle, conf=conf, iou=iou, imgsz=imgsz)

    ratio_txt = f"{result['ripe_ratio']:.1%}" if result["total"] else "N/A"
    mean_conf_txt = f"{result['mean_conf']:.3f}" if result["total"] else "N/A"

    st.subheader(f"분석 결과 · `{source_name}`")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("미숙", result["n_unripe"])
    c2.metric("완숙", result["n_ripe"])
    c3.metric("전체", result["total"])
    c4.metric("완숙비율", ratio_txt)
    c5.metric("판정", f"{decision_color(result['decision'])} {result['decision']}")

    col_img, col_tbl = st.columns([1.2, 1])
    with col_img:
        st.markdown("#### 탐지 결과")
        st.image(result["annotated"], use_container_width=True)
        st.caption(f"평균 confidence: {mean_conf_txt} · conf={conf} · iou={iou} · imgsz={imgsz}")

    with col_tbl:
        st.markdown("#### 객체 목록")
        if result["detections"].empty:
            st.warning("탐지된 객체가 없습니다. Confidence를 낮춰 보세요.")
        else:
            df_show = result["detections"].copy()
            df_show["confidence"] = df_show["confidence"].map(lambda x: round(x, 3))
            for col in ("x1", "y1", "x2", "y2"):
                df_show[col] = df_show[col].map(lambda x: round(x, 1))
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.markdown("#### 수확 가이드")
        st.markdown(
            f"""
| 판정 | 조건 |
|------|------|
| 수확 권장 | 완숙비율 ≥ {HARVEST_HIGH:.0%} |
| 부분 수확 | {HARVEST_MID:.0%} ≤ 완숙비율 < {HARVEST_HIGH:.0%} |
| 생육 관찰 | 완숙비율 < {HARVEST_MID:.0%} |
| 객체 없음 | 탐지 0개 |

현재: **{result['decision']}** (완숙비율 {ratio_txt})
"""
        )


if __name__ == "__main__":
    # Windows에서 Streamlit 재실행 시 경로 이슈 완화
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    main()
