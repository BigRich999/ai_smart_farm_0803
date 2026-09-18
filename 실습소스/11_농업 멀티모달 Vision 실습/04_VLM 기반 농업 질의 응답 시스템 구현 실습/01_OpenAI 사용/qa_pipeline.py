"""VLM 분석 + RAG 검색 + LLM 최종 답변 파이프라인."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import LLM_MODEL
from rag_store import format_retrieved_context, search_knowledge
from vlm_analyzer import analyze_crop_image, build_search_query

ANSWER_SYSTEM = """당신은 농업기술 상담 전문가입니다.
제공된 (1) 이미지 분석 결과와 (2) 농업기술 문서 검색 결과만을 근거로 한국어로 답변하세요.
문서에 없는 내용은 추측하지 말고, 확인이 필요하다고 안내하세요.
확진이 아니라 의심/가능성 수준으로 표현하세요.
이미지 분석이 '정상'이어도 사용자가 문제·증상을 묻고 검색 문서에 병해 정보가 있으면,
관찰이 불충분했을 가능성을 언급하고 관련 병해 관리 요령을 함께 안내하세요.
답변에는 반드시 다음 항목을 포함하세요:
1) 이미지 분석 결과
2) 주요 증상
3) 발생 원인
4) 발생 환경
5) 관리 방법
6) 참고 문서
"""


def generate_final_answer(
    user_question: str,
    analysis: dict[str, Any],
    context: str,
    sources: list[str],
    client: OpenAI | None = None,
) -> str:
    client = client or OpenAI()
    source_text = ", ".join(sources) if sources else "검색 결과 없음"

    user_prompt = f"""
[사용자 질문]
{user_question}

[VLM 이미지 분석 결과]
- 작물: {analysis.get('crop')}
- 상태: {analysis.get('health_status')}
- 관찰 증상: {', '.join(analysis.get('observed_symptoms') or [])}
- 의심 질병: {', '.join(analysis.get('suspected_diseases') or [])}
- 신뢰도: {analysis.get('confidence')}
- 요약: {analysis.get('summary')}

[농업기술 검색 결과]
{context}

[참고 문서 후보]
{source_text}

위 정보를 종합해 농가가 바로 활용할 수 있게 답변하세요.
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def run_multimodal_qa(
    image_source: str | bytes,
    user_question: str,
    filename: str = "upload.jpg",
) -> dict[str, Any]:
    """
    전체 파이프라인:
    이미지+질문 → VLM 분석 → 검색질의 생성 → RAG 검색 → LLM 최종 답변
    """
    client = OpenAI()
    analysis = analyze_crop_image(
        image_source=image_source,
        user_question=user_question,
        filename=filename,
        client=client,
    )
    query = build_search_query(analysis, user_question)
    docs = search_knowledge(query)
    context, sources = format_retrieved_context(docs)
    answer = generate_final_answer(
        user_question=user_question,
        analysis=analysis,
        context=context,
        sources=sources,
        client=client,
    )
    return {
        "analysis": analysis,
        "search_query": query,
        "retrieved_docs": [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "preview": d.page_content[:240],
            }
            for d in docs
        ],
        "sources": sources,
        "answer": answer,
    }
