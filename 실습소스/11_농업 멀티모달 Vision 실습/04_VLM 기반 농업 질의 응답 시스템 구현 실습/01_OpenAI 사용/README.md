# VLM 기반 농업 질의응답 시스템

고추·무·배추 이미지를 VLM으로 분석하고, 농업기술 PDF를 RAG 지식베이스로 활용해
증상·원인·관리방법을 답변하는 Streamlit 서비스입니다.

## 구성

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit 웹 UI |
| `vlm_analyzer.py` | GPT-4o-mini Vision 이미지 분석 |
| `rag_store.py` | PDF 로딩·청킹·Chroma 임베딩/검색 |
| `qa_pipeline.py` | VLM + RAG + LLM 통합 파이프라인 |
| `build_vectordb.py` | 벡터DB 사전 구축 |
| `pdf_data/` | 농업기술 PDF 5종 |
| `sample_images/` | AI Hub 샘플 이미지(고추/무/배추) |
| `knowledge_supplement/` | 스캔본 배추 PDF 보완 요약 |

## 처리 흐름

```
이미지 + 질문
  → VLM 분석 (작물/정상·질병/증상/의심병)
  → 검색 질의 생성
  → 농업기술 RAG 검색 (Chroma)
  → LLM 최종 답변 (증상·원인·환경·관리·참고문서)
```

## 준비

1. OpenAI API 키: `C:\env\.env` 에 `OPENAI_API_KEY=...` (기존 실습과 동일)
2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. 벡터DB 구축 (최초 1회)

```bash
python build_vectordb.py
```

- PDF 텍스트는 `cache/pdf_texts.json` 에 캐시됩니다.
- Chroma DB는 Windows 한글 경로 이슈를 피해 `C:\MyCursorLab\_vlm_agri_chroma` 에 저장됩니다 (`config.py` 참고).
- 재구축: `python build_vectordb.py --force`
4. 앱 실행

```bash
streamlit run app.py
```

## 테스트 시나리오

1. 사이드바에서 `배추_질병` 샘플 선택
2. 기본 질문 유지: `이 배추에 어떤 문제가 있는 것 같나요? 증상과 관리 방법을 알려주세요.`
3. **질의응답 실행** 클릭
4. 결과에서 확인
   - 이미지 분석 (작물=배추, 질병의심, 증상)
   - RAG 검색 질의
   - 최종 답변 (증상/원인/환경/관리)
   - 참고 문서명 표시

## 참고

- `농업기술길잡이128_배추.PDF` 는 스캔 이미지 중심이라 텍스트 추출이 되지 않아,
  `knowledge_supplement/배추_병해충_요지.txt` 를 보완 지식으로 함께 인덱싱합니다.
- 나머지 4개 PDF는 텍스트 추출 후 청킹·임베딩합니다.
- VLM/LLM 기본 모델: `gpt-4o-mini` (`config.py` 에서 변경 가능)
