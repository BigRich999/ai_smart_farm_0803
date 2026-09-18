"""농업기술 PDF RAG 지식베이스 구축/검색 모듈."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CACHE_DIR,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_DIR,
    PDF_FILES,
    RETRIEVE_K,
    SUPPLEMENT_DIR,
)
def _file_fingerprint(path: Path) -> str:
    h = hashlib.md5()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(str(path.stat().st_mtime_ns).encode("utf-8"))
    h.update(str(path.stat().st_size).encode("utf-8"))
    return h.hexdigest()


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """PDF에서 페이지 단위 텍스트를 추출한다."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  ... {i + 1}/{total} 페이지 처리")
        text = (page.extract_text() or "").strip()
        if len(text) < 40:
            continue
        pages.append(
            {
                "page": i + 1,
                "text": text,
                "source": pdf_path.name,
            }
        )
    return pages


def load_or_extract_pdf_cache(force: bool = False) -> list[dict[str, Any]]:
    """PDF 텍스트 추출 결과를 캐시하고 재사용한다."""
    cache_path = CACHE_DIR / "pdf_texts.json"
    fingerprints = {}
    for name in PDF_FILES:
        path = PDF_DIR / name
        if path.exists():
            fingerprints[name] = _file_fingerprint(path)

    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprints") == fingerprints:
            return cached["pages"]

    all_pages: list[dict[str, Any]] = []
    for name in PDF_FILES:
        path = PDF_DIR / name
        if not path.exists():
            print(f"[WARN] PDF 없음: {path}")
            continue
        print(f"[INFO] 추출 중: {name}")
        pages = extract_pdf_pages(path)
        print(f"  -> 유효 페이지 {len(pages)}개")
        all_pages.extend(pages)

    # 배추 PDF는 스캔본이므로 보완 텍스트 추가
    supplement_path = SUPPLEMENT_DIR / "배추_병해충_요지.txt"
    if supplement_path.exists():
        all_pages.append(
            {
                "page": 1,
                "text": supplement_path.read_text(encoding="utf-8"),
                "source": "농업기술길잡이128_배추.PDF (보완요약)",
            }
        )

    payload = {"fingerprints": fingerprints, "pages": all_pages}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return all_pages


def pages_to_documents(pages: list[dict[str, Any]]) -> list[Document]:
    docs: list[Document] = []
    for item in pages:
        docs.append(
            Document(
                page_content=item["text"],
                metadata={
                    "source": item["source"],
                    "page": item["page"],
                },
            )
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "。", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """PDF(+보완자료)를 임베딩하여 Chroma에 저장한다."""
    marker = CHROMA_DIR / ".ready"
    embeddings = get_embeddings()

    if marker.exists() and not force_rebuild:
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

    # 기존 컬렉션 재생성
    if force_rebuild and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # PDF 텍스트는 캐시가 있으면 재사용 (force는 벡터DB만 재구축)
    pages = load_or_extract_pdf_cache(force=False)
    print(f"[INFO] 인덱싱 페이지 수: {len(pages)}")
    docs = pages_to_documents(pages)
    chunks = split_documents(docs)
    print(f"[INFO] 청크 수: {len(chunks)}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    marker.write_text("ready", encoding="utf-8")
    return vectorstore


def load_vectorstore() -> Chroma:
    marker = CHROMA_DIR / ".ready"
    if not marker.exists():
        return build_vectorstore(force_rebuild=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def search_knowledge(query: str, k: int = RETRIEVE_K) -> list[Document]:
    """검색 질의로 농업기술 문서를 검색한다."""
    vs = load_vectorstore()
    return vs.similarity_search(query, k=k)


def format_retrieved_context(docs: list[Document]) -> tuple[str, list[str]]:
    blocks: list[str] = []
    sources: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        sources.append(f"{source} (p.{page})")
        blocks.append(f"[자료 {i}] 출처: {source} / 페이지: {page}\n{doc.page_content}")
    # 문서명만 중복 제거
    unique_names: list[str] = []
    seen: set[str] = set()
    for s in sources:
        name = s.split(" (p.")[0]
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    return "\n\n".join(blocks), unique_names
