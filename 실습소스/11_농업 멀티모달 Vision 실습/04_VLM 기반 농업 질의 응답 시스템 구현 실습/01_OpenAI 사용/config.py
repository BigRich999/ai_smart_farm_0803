"""프로젝트 공통 설정."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# 기존 실습과 동일하게 C:\env\.env 우선 로드
load_dotenv(r"C:\env\.env")
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf_data"
SAMPLE_IMAGE_DIR = BASE_DIR / "sample_images"
SUPPLEMENT_DIR = BASE_DIR / "knowledge_supplement"
CACHE_DIR = BASE_DIR / "cache"
# Windows + Chroma HNSW는 한글 경로에서 header 파일 오류가 날 수 있어 ASCII 경로 사용
CHROMA_DIR = Path(r"C:\MyCursorLab\_vlm_agri_chroma")

CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENT_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "agri_tech_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
VLM_MODEL = "gpt-4o-mini"
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVE_K = 6

PDF_FILES = [
    "농업기술길잡이47_채소병해충.PDF",
    "농업기술길잡이-(2권) 채소병해충.pdf",
    "5 고추 최종파일_단면.pdf",
    "20 무_고화질_단면.pdf",
    "농업기술길잡이128_배추.PDF",
]
