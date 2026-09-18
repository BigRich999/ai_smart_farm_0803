"""VLM 기반 농작물 이미지 분석 모듈."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import VLM_MODEL

ANALYSIS_SCHEMA_HINT = """
반드시 아래 JSON 형식만 출력하세요. 다른 설명·마크다운은 넣지 마세요.
{
  "crop": "고추|무|배추|기타|불명",
  "health_status": "정상|질병의심|판단불가",
  "observed_symptoms": ["증상1", "증상2"],
  "suspected_diseases": ["의심질병1", "의심질병2"],
  "severity": "low|medium|high",
  "confidence": "low|medium|high",
  "summary": "한글로 2~4문장 요약"
}
"""

SYSTEM_PROMPT = """당신은 노지 작물(고추, 무, 배추) 병해 진단에 능숙한 농업 비전 전문가입니다.
이미지만으로 확진하지 말고, 관찰 가능한 시각적 근거를 바탕으로 분석하세요.

[판정 원칙 — 매우 중요]
1) 정상을 기본값으로 두지 마세요. 병반·변색·위축·부패·반점·주름·함몰이
   하나라도 보이면 health_status는 반드시 "질병의심"입니다.
2) 애매하면 "정상"이 아니라 "질병의심" 또는 "판단불가"를 선택하세요.
3) 사용자가 문제·증상·관리를 물으면, 건강한 과실만 골라 보지 말고
   손상·이상 부위를 우선 찾아 기술하세요.
4) observed_symptoms가 비어 있는데 health_status가 "정상"이면 안 됩니다.
   이상이 없으면 증상은 []이고 정상, 이상이 있으면 증상을 반드시 채우세요.

[작물별 주의 증상]
- 고추: 과실 흑변/암갈변, 원형·부정형 병반, 꼭지부 썩음, 주름·위축, 물컹한 무름,
  탄저병·무름병·역병 의심 가능
- 무: 잎 황화·반점·시들음, 뿌리 갈변·무름
- 배추: 잎 황화·흑반·무름·시들음, 무름병·검은무늬병·노균병 의심 가능

고추·무·배추 외 작물이면 crop을 기타로 표시하세요.
severity와 confidence는 같은 값으로 두고, 근거가 분명할수록 high로 두세요.
"""

_PROBLEM_HINT = re.compile(r"(문제|질병|병해|증상|이상|관리|진단|뭐가|무엇이)")


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


def encode_image(image_path: str | Path) -> tuple[str, str]:
    path = Path(image_path)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return data, _guess_mime(path)


def encode_image_bytes(image_bytes: bytes, filename: str = "upload.jpg") -> tuple[str, str]:
    data = base64.b64encode(image_bytes).decode("utf-8")
    mime, _ = mimetypes.guess_type(filename)
    return data, mime or "image/jpeg"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_analysis(result: dict[str, Any], raw: str) -> dict[str, Any]:
    crop = str(result.get("crop") or "불명")
    status = str(result.get("health_status") or "판단불가").strip()
    symptoms = result.get("observed_symptoms") or []
    diseases = result.get("suspected_diseases") or []
    level = str(result.get("confidence") or result.get("severity") or "low").lower()
    if level not in {"low", "medium", "high"}:
        level = "low"
    if not isinstance(symptoms, list):
        symptoms = [str(symptoms)]
    if not isinstance(diseases, list):
        diseases = [str(diseases)]
    symptoms = [str(s).strip() for s in symptoms if str(s).strip()]
    diseases = [str(d).strip() for d in diseases if str(d).strip()]

    if symptoms and status == "정상":
        status = "질병의심"
    if diseases and status == "정상":
        status = "질병의심"

    return {
        "crop": crop,
        "health_status": status,
        "observed_symptoms": symptoms,
        "suspected_diseases": diseases,
        "severity": level,
        "confidence": level,
        "summary": str(result.get("summary") or raw[:500]),
        "raw_model_output": raw,
    }


def _looks_under_diagnosed(analysis: dict[str, Any], user_question: str) -> bool:
    if not _PROBLEM_HINT.search(user_question or ""):
        return False
    status = analysis.get("health_status")
    symptoms = analysis.get("observed_symptoms") or []
    diseases = analysis.get("suspected_diseases") or []
    return status == "정상" and not symptoms and not diseases


def _call_vlm(
    client: OpenAI,
    *,
    b64: str,
    mime: str,
    user_text: str,
) -> str:
    response = client.chat.completions.create(
        model=VLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content or "{}"


def analyze_crop_image(
    image_source: str | Path | bytes,
    user_question: str = "",
    filename: str = "upload.jpg",
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """이미지를 VLM으로 분석하여 구조화 결과를 반환한다."""
    client = client or OpenAI()

    if isinstance(image_source, (bytes, bytearray)):
        b64, mime = encode_image_bytes(bytes(image_source), filename)
    else:
        b64, mime = encode_image(image_source)

    question_part = f"\n사용자 질문: {user_question}" if user_question.strip() else ""
    user_text = (
        "다음 농작물 이미지를 정밀 분석하세요.\n"
        "과실·잎·줄기에서 색 변화, 병반, 주름, 부패, 시들음이 있는지 먼저 나열한 뒤 판정하세요.\n"
        f"{question_part}\n"
        "작물은 고추·무·배추 중심으로 판별하세요.\n"
        f"{ANALYSIS_SCHEMA_HINT}"
    )

    raw = _call_vlm(client, b64=b64, mime=mime, user_text=user_text)
    try:
        analysis = _normalize_analysis(_extract_json(raw), raw)
    except Exception:
        analysis = {
            "crop": "불명",
            "health_status": "판단불가",
            "observed_symptoms": [],
            "suspected_diseases": [],
            "severity": "low",
            "confidence": "low",
            "summary": raw[:500],
            "raw_model_output": raw,
        }

    if _looks_under_diagnosed(analysis, user_question):
        retry_text = (
            "이전 판정이 '정상'이었지만, 사용자는 문제·증상을 묻고 있습니다.\n"
            "이미지를 다시 자세히 보세요. 특히 과실의 흑변·갈변·반점·주름·꼭지부 썩음,"
            "잎의 황화·반점·시들음을 찾아 observed_symptoms에 적으세요.\n"
            "이상이 보이면 health_status는 반드시 질병의심입니다.\n"
            f"{question_part}\n"
            f"{ANALYSIS_SCHEMA_HINT}"
        )
        raw2 = _call_vlm(client, b64=b64, mime=mime, user_text=retry_text)
        try:
            analysis = _normalize_analysis(_extract_json(raw2), raw2)
            analysis["raw_model_output"] = raw2
            analysis["retried"] = True
        except Exception:
            analysis["retried"] = False

    return analysis


def build_search_query(analysis: dict[str, Any], user_question: str) -> str:
    """VLM 분석 결과와 사용자 질문으로 RAG 검색 질의를 구성한다."""
    parts: list[str] = []
    crop = analysis.get("crop") or ""
    status = analysis.get("health_status") or ""
    symptoms = analysis.get("observed_symptoms") or []
    diseases = analysis.get("suspected_diseases") or []

    if crop and crop not in {"기타", "불명"}:
        parts.append(f"{crop} 병해충")
        parts.append(f"{crop} 병해")

    if status and status != "정상":
        parts.append(str(status))

    parts.extend([str(s) for s in symptoms[:5]])
    parts.extend([str(d) for d in diseases[:4]])

    if crop == "고추" and not diseases and not symptoms:
        parts.extend(["탄저병", "무름병", "과실 병반", "꼭지부 썩음"])
    elif crop == "무" and not diseases and not symptoms:
        parts.extend(["무름병", "검은무늬병", "잎 황화"])
    elif crop == "배추" and not diseases and not symptoms:
        parts.extend(["무름병", "검은무늬병", "노균병", "시들음"])

    if user_question.strip():
        parts.append(user_question.strip())

    seen: set[str] = set()
    ordered: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    return " ".join(ordered)
