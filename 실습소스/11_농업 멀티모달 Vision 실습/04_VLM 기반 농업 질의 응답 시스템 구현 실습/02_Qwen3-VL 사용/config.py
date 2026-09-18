"""Qwen3-VL 농업 질의응답 프로젝트 공통 설정."""

from __future__ import annotations

import os
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
CHROMA_DIR = Path(r"C:\MyCursorLab\_vlm_agri_chroma_qwen3vl")

CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENT_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "agri_tech_knowledge_qwen3"

# ----- Qwen3-VL -----
# backend: "auto" | "local" | "api"
# auto = API 환경변수가 있으면 api, 아니면 local
QWEN_BACKEND = os.getenv("QWEN_BACKEND", "auto").strip().lower()
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
# 로컬 양자화: none | 4bit | 8bit
QWEN_QUANTIZATION = os.getenv("QWEN_QUANTIZATION", "auto").strip().lower()
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "1024"))
QWEN_DEVICE_MAP = os.getenv("QWEN_DEVICE_MAP", "auto")

# OpenAI 호환 API (DashScope / vLLM / Ollama 등)
# 예) DASHSCOPE: https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_BASE = os.getenv("QWEN_API_BASE", os.getenv("OPENAI_BASE_URL", "")).strip()
QWEN_API_KEY = os.getenv(
    "QWEN_API_KEY",
    os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
).strip()
# API에서 사용할 모델명 (제공사별 다를 수 있음)
QWEN_API_MODEL = os.getenv("QWEN_API_MODEL", QWEN_VL_MODEL)

# ----- Embedding / RAG -----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
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
