"""
Qwen3-VL 기반 농업 질의응답 Streamlit 앱

실행:
  1) python prepare_data.py
  2) python build_vectordb.py
  3) streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import CHROMA_DIR, EMBEDDING_MODEL, SAMPLE_IMAGE_DIR
from qa_pipeline import run_multimodal_qa
from qwen_client import describe_runtime
from rag_store import load_vectorstore

st.set_page_config(
    page_title="Qwen3-VL 농업 질의응답",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #5f6b76; margin-bottom: 1.2rem; }
    .source-box {
        background: #f4f7f5;
        border-left: 4px solid #2f7d4a;
        padding: 0.8rem 1rem;
        border-radius: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Qwen3-VL 기반 농업 질의응답 시스템</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">고추·무·배추 이미지 분석(Qwen3-VL) + 농업기술 PDF RAG(로컬 Embedding)</div>',
    unsafe_allow_html=True,
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "busy" not in st.session_state:
    st.session_state.busy = False

runtime = describe_runtime()

with st.sidebar:
    st.header("설정")
    st.caption("지식베이스: `pdf_data` 5종 PDF + 배추 보완요약")
    st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
    st.caption(f"Qwen backend: `{runtime['backend']}` / model: `{runtime['model']}`")
    if runtime.get("cuda"):
        st.caption(f"GPU: {runtime.get('gpu')} ({runtime.get('vram_gb')} GB)")
    else:
        st.caption("CUDA: 없음 (API 백엔드 또는 CPU 로컬)")
    if runtime.get("backend") == "local":
        st.caption("Embedding은 VRAM 보호를 위해 CPU 사용")

    ready = (CHROMA_DIR / ".ready").exists()
    if ready:
        st.success("Vector DB 준비됨")
    else:
        st.warning("Vector DB 없음 → 첫 실행 시 자동 구축(시간 소요)")

    if st.button("지식베이스 로드/확인"):
        with st.spinner("Chroma 로딩 중..."):
            vs = load_vectorstore()
            st.info(f"컬렉션 문서 수: {vs._collection.count()}")  # noqa: SLF001

    st.divider()
    st.subheader("샘플 이미지")
    sample_files: list[Path] = []
    if SAMPLE_IMAGE_DIR.exists():
        sample_files = sorted(SAMPLE_IMAGE_DIR.rglob("*.*"))
        sample_files = [
            p
            for p in sample_files
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    selected_sample = st.selectbox(
        "미리 준비된 샘플 선택",
        options=["(선택 안 함)"] + [str(p.relative_to(SAMPLE_IMAGE_DIR)) for p in sample_files],
    )

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("1) 이미지 업로드")
    uploaded = st.file_uploader(
        "고추·무·배추 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png"],
    )

    image_bytes: bytes | None = None
    image_name = "upload.jpg"

    if selected_sample != "(선택 안 함)":
        sample_path = SAMPLE_IMAGE_DIR / selected_sample
        image_bytes = sample_path.read_bytes()
        image_name = sample_path.name
        st.image(str(sample_path), caption=f"샘플: {selected_sample}", use_container_width=True)
    elif uploaded is not None:
        image_bytes = uploaded.getvalue()
        image_name = uploaded.name
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

    st.subheader("2) 질문 입력")
    default_q = "이 배추에 어떤 문제가 있는 것 같나요? 증상과 관리 방법을 알려주세요."
    question = st.text_area("농업 관련 질문", value=default_q, height=110)
    run = st.button(
        "질의응답 실행",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.busy,
    )

with col_right:
    st.subheader("3) 결과")
    if run:
        if image_bytes is None:
            st.error("이미지를 업로드하거나 샘플을 선택하세요.")
        elif not question.strip():
            st.error("질문을 입력하세요.")
        else:
            st.session_state.busy = True
            with st.spinner("Qwen3-VL 분석 → RAG 검색 → 답변 생성 중..."):
                try:
                    result = run_multimodal_qa(
                        image_source=image_bytes,
                        user_question=question.strip(),
                        filename=image_name,
                    )
                    st.session_state.last_result = result
                except Exception as e:
                    st.session_state.busy = False
                    st.exception(e)
                    st.stop()
                finally:
                    st.session_state.busy = False

    result = st.session_state.last_result
    if result:
        analysis = result["analysis"]
        st.markdown("#### 이미지 분석 결과 (Qwen3-VL)")
        m1, m2, m3 = st.columns(3)
        m1.metric("작물", analysis.get("crop", "-"))
        m2.metric("상태", analysis.get("health_status", "-"))
        m3.metric(
            "신뢰도",
            analysis.get("confidence") or analysis.get("severity") or "-",
        )
        st.write("**관찰 증상:**", ", ".join(analysis.get("observed_symptoms") or []) or "-")
        st.write("**의심 질병:**", ", ".join(analysis.get("suspected_diseases") or []) or "-")
        st.info(analysis.get("summary", ""))

        st.markdown("#### RAG 검색 질의")
        st.code(result["search_query"], language=None)

        with st.expander("검색된 농업기술 조각 보기"):
            for i, doc in enumerate(result["retrieved_docs"], start=1):
                st.markdown(
                    f"**[{i}] {doc['source']} (p.{doc['page']})**\n\n{doc['preview']}..."
                )

        st.markdown("#### 최종 답변")
        st.markdown(result["answer"])

        st.markdown("#### 참고 문서")
        sources = result.get("sources") or []
        if sources:
            st.markdown(
                '<div class="source-box">'
                + "<br>".join(f"• {s}" for s in sources)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("참고 문서를 찾지 못했습니다.")

    else:
        st.caption("이미지를 넣고 질문을 입력한 뒤 [질의응답 실행]을 누르세요.")
        st.markdown(
            """
처리 흐름
1. 농작물 이미지 + 사용자 질문
2. Qwen3-VL 이미지 분석 (작물·정상/질병·증상·의심병)
3. 분석 결과 기반 검색 질의 생성
4. 농업기술 PDF RAG 검색 (로컬 Embedding)
5. Qwen3-VL + RAG 결합 최종 답변
"""
        )
