"""농업기술 PDF → 로컬 Embedding → Chroma Vector DB 구축 스크립트.

사용법:
  python build_vectordb.py
  python build_vectordb.py --force
"""

from __future__ import annotations

import argparse
import sys

from rag_store import build_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="농업기술 RAG Vector DB 구축 (Qwen3-VL 실습)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="벡터DB를 무시하고 재구축",
    )
    args = parser.parse_args()

    print("=== 로컬 임베딩 및 Chroma 저장 ===")
    vs = build_vectorstore(force_rebuild=args.force)
    count = vs._collection.count()  # noqa: SLF001
    print(f"구축 완료. 컬렉션 문서 수: {count}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
