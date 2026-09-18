"""zip 데이터 준비 스크립트: PDF 해제 + 샘플 이미지 정리.

사용법:
  python prepare_data.py
  python prepare_data.py --samples 5
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from config import BASE_DIR, PDF_DIR, SAMPLE_IMAGE_DIR

IMAGE_ZIP_CANDIDATES = [
    "노지 작물 질병 진단 이미지_sample.zip",
]
PDF_ZIP_NAME = "pdf_data.zip"

CROP_MAP = [
    ("원천데이터/01.고추/0.정상/", "고추_정상"),
    ("원천데이터/01.고추/1.질병/", "고추_질병"),
    ("원천데이터/02.무/0.정상/", "무_정상"),
    ("원천데이터/02.무/1.질병/", "무_질병"),
    ("원천데이터/03.배추/0.정상/", "배추_정상"),
    ("원천데이터/03.배추/1.질병/", "배추_질병"),
]


def find_image_zip() -> Path:
    for name in IMAGE_ZIP_CANDIDATES:
        path = BASE_DIR / name
        if path.exists():
            return path
    # 이름 인코딩이 달라도 zip 중 pdf_data가 아닌 것 선택
    for path in BASE_DIR.glob("*.zip"):
        if path.name != PDF_ZIP_NAME:
            return path
    raise FileNotFoundError("이미지 sample zip을 찾을 수 없습니다.")


def extract_pdfs() -> None:
    zip_path = BASE_DIR / PDF_ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"없음: {zip_path}")
    if PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf")) or any(PDF_DIR.glob("*.PDF")):
        print(f"[SKIP] PDF 이미 존재: {PDF_DIR}")
        return
    print(f"[INFO] PDF 압축 해제: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(BASE_DIR)
    print(f"[OK] {PDF_DIR}")


def extract_samples(n_per_group: int = 3) -> None:
    zip_path = find_image_zip()
    SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 샘플 이미지 추출: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for prefix, out_name in CROP_MAP:
            out_dir = SAMPLE_IMAGE_DIR / out_name
            out_dir.mkdir(parents=True, exist_ok=True)
            matches = [
                n
                for n in names
                if n.startswith(prefix) and n.lower().endswith((".jpg", ".jpeg"))
            ]
            count = 0
            for src in matches[:n_per_group]:
                count += 1
                ext = Path(src).suffix.lower() or ".jpg"
                dest = out_dir / f"{out_name}_{count:02d}{ext}"
                if dest.exists():
                    continue
                with zf.open(src) as src_f, open(dest, "wb") as dst_f:
                    dst_f.write(src_f.read())
            print(f"  - {out_name}: {count}장")
    print(f"[OK] {SAMPLE_IMAGE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL 실습 데이터 준비")
    parser.add_argument("--samples", type=int, default=3, help="작물·상태별 샘플 수")
    args = parser.parse_args()
    extract_pdfs()
    extract_samples(n_per_group=args.samples)
    print("데이터 준비 완료")


if __name__ == "__main__":
    main()
