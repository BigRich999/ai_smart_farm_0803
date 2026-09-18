# Qwen3-VL 기반 농업 질의응답 시스템

고추·무·배추 이미지를 **Qwen3-VL**로 분석하고, 농업기술 PDF를 **로컬 Embedding + Chroma RAG**로
활용해 증상·원인·관리방법을 답변하는 Streamlit 서비스입니다.

OpenAI gpt-4o-mini를 사용하지 않습니다.

## 구성

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit 웹 UI |
| `qwen_client.py` | Qwen3-VL 로컬/API 통합 클라이언트 |
| `vlm_analyzer.py` | Qwen3-VL 이미지 분석(JSON) |
| `rag_store.py` | PDF 로딩·청킹·로컬 임베딩/Chroma 검색 |
| `qa_pipeline.py` | VLM + RAG + Qwen3-VL 최종 답변 |
| `build_vectordb.py` | 벡터DB 사전 구축 |
| `prepare_data.py` | zip 해제 및 샘플 이미지 정리 |
| `pdf_data/` | 농업기술 PDF 5종 |
| `sample_images/` | AI Hub 샘플(고추/무/배추 × 정상/질병) |
| `knowledge_supplement/` | 스캔본 배추 PDF 보완 요약 |

## 처리 흐름

```
이미지 + 질문
  → Qwen3-VL 분석 (작물/정상·질병/증상/의심병/severity)
  → 검색 질의 생성
  → 농업기술 RAG 검색 (Chroma + BAAI/bge-m3)
  → Qwen3-VL 최종 답변 (증상·원인·환경·관리·참고문서)
```

## 데이터 준비

폴더에 있는 zip 2개를 사용합니다.

```bash
python prepare_data.py
# 또는 샘플을 더 많이: python prepare_data.py --samples 5
```

- `pdf_data.zip` → `pdf_data/`
- `노지 작물 질병 진단 이미지_sample.zip` → `sample_images/{고추,무,배추}_{정상,질병}/`

## 준비 (패키지)

```bash
pip install -r requirements.txt
```

GPU 4bit 양자화를 쓰려면 (CUDA 환경):

```bash
pip install bitsandbytes
```

환경변수는 `C:\env\.env` 또는 프로젝트 `.env`에 설정합니다.

### A) 로컬 Transformers (기본)

```env
QWEN_BACKEND=local
QWEN_VL_MODEL=Qwen/Qwen3-VL-4B-Instruct
QWEN_QUANTIZATION=auto
EMBEDDING_MODEL=BAAI/bge-m3
```

- VRAM이 부족하면 `QWEN_QUANTIZATION=4bit` 또는 `Qwen/Qwen3-VL-2B-Instruct` 사용
- GPU가 없으면 로컬 추론이 매우 느리거나 실패할 수 있음 → B안 권장

### B) OpenAI 호환 API (DashScope / vLLM / Ollama)

GPU가 없거나 로컬 로드가 어려울 때:

```env
QWEN_BACKEND=api
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-...
QWEN_API_MODEL=qwen3-vl-plus
EMBEDDING_MODEL=BAAI/bge-m3
```

Ollama 예시:

```env
QWEN_BACKEND=api
QWEN_API_BASE=http://localhost:11434/v1
QWEN_API_KEY=ollama
QWEN_API_MODEL=qwen3-vl
```

`QWEN_BACKEND=auto`(기본)이면 `QWEN_API_BASE`가 있을 때 API, 없으면 local을 사용합니다.

## 벡터DB 구축

```bash
python build_vectordb.py
```

- PDF 텍스트는 `cache/pdf_texts.json`에 캐시됩니다.
- Chroma DB는 Windows 한글 경로 이슈를 피해 `C:\MyCursorLab\_vlm_agri_chroma_qwen3vl`에 저장됩니다.
- 재구축: `python build_vectordb.py --force`

## 앱 실행

```bash
streamlit run app.py
```

## 테스트 시나리오

1. 사이드바에서 `배추_질병` 샘플 선택
2. 기본 질문 유지: `이 배추에 어떤 문제가 있는 것 같나요? 증상과 관리 방법을 알려주세요.`
3. **질의응답 실행** 클릭
4. 결과에서 확인
   - 이미지 분석 (작물=배추, 질병의심, 증상, severity)
   - RAG 검색 질의
   - 최종 답변 (증상/원인/환경/관리)
   - 참고 문서명 표시

고추 정상/질병, 무 질병 샘플로도 동일하게 확인합니다.

## 참고

- `농업기술길잡이128_배추.PDF`는 스캔 이미지 중심이라 텍스트 추출이 되지 않아,
  `knowledge_supplement/배추_병해충_요지.txt`를 보완 지식으로 함께 인덱싱합니다.
- Embedding은 OpenAI가 아니라 `BAAI/bge-m3`(로컬)를 사용합니다.
- **로컬 Qwen3-VL 사용 시 Embedding은 기본 CPU**입니다. VLM과 GPU VRAM을 나눠 쓰면
  첫 질의는 되고 두 번째부터 무한 대기처럼 보일 수 있습니다.
  (`EMBEDDING_DEVICE=cuda`로 강제 가능하지만 VRAM 여유 필요)
- 이미지 긴 변은 추론 전 1280px로 축소합니다(원본 4K로 VRAM 고갈 방지).
- 메모리/속도가 부담되면 `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` 등으로 교체 가능합니다.
- 본 실습은 OpenAI Vision 모델(`gpt-4o-mini` 등)로 VLM을 대체하지 않습니다.

### 두 번째 질의부터 멈추는 경우

1. Streamlit을 **완전히 종료 후 재실행** (`Ctrl+C` → `streamlit run app.py`)
2. `QWEN_QUANTIZATION=4bit` 또는 `Qwen/Qwen3-VL-2B-Instruct`로 VRAM 절약
3. 터미널에서 CUDA OOM 메시지 여부 확인
