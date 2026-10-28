from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from rag.contracts import Scope
from rag.generation.answer_formatter import format_answer

questions_router = APIRouter()
logger = logging.getLogger(__name__)


@questions_router.post("/api/questions")
async def ask_question(request: Request):
    payload = None
    content_type = (
        request.headers.get("content-type", "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )
    if content_type == "application/json" or (
        content_type.startswith("application/")
        and content_type.endswith("+json")
    ):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if not isinstance(payload, dict):
        return JSONResponse(
            {"error": "Expected a JSON object."},
            status_code=400,
        )

    question = payload.get("question")
    course_id = payload.get("course_id")

    if not isinstance(question, str) or not question.strip():
        return JSONResponse(
            {"error": "Question is required."},
            status_code=400,
        )

    if len(question) > 3000:
        return JSONResponse(
            {"error": "Question must be at most 3000 characters."},
            status_code=400,
        )

    if not isinstance(course_id, str) or not course_id.strip():
        return JSONResponse(
            {"error": "course_id is required."},
            status_code=400,
        )

    course_id = course_id.strip()

    if len(course_id) > 80 or not all(
        ch.isalnum() or ch in "_-" for ch in course_id
    ):
        return JSONResponse(
            {"error": "Invalid course_id."},
            status_code=400,
        )

    rag_engine = request.app.state.rag_engine
    tenant_id = request.app.state.rag_tenant_id

    scope = Scope(
        tenant_id=tenant_id,
        course_ids=frozenset({course_id}),
    )

    result = await run_in_threadpool(
        rag_engine.answer,
        question.strip(),
        scope,
    )

    store = getattr(rag_engine, "store", None)
    store_count = store.count() if store is not None else "unavailable"

    logger.warning(
        "[QUESTION DEBUG] store_count=%s tenant=%s courses=%s "
        "abstained=%s reason=%s sources=%s",
        store_count,
        scope.tenant_id,
        sorted(scope.course_ids),
        result.abstained,
        result.reason,
        len(result.sources),
    )

    sources = []

    for retrieved in result.sources:
        chunk = retrieved.chunk

        sources.append({
            "document_id": chunk.document_id,
            "filename": chunk.source_uri,
            "source_uri": chunk.source_uri,
            "page": chunk.page,
            "section": chunk.section,
            "text": chunk.text,
            "score": retrieved.score,
        })

    messages = {
        "no_evidence": (
            "No relevant evidence was retrieved from this course."
        ),
        "no_llm_configured": (
            "Evidence was retrieved, but answer generation is disabled. "
            "Configure LLM_API_KEY to generate an answer."
        ),
    }

    message = messages.get(
        result.reason,
        (
            "The course material did not provide enough evidence "
            "for a reliable answer."
            if result.abstained
            else ""
        ),
    )

    formatted_answer = format_answer(result.answer)

    return {
        "question": result.question,
        "answer": result.answer,
        "answer_format": formatted_answer["answer_format"],
        "content": formatted_answer["content"],
        "abstained": result.abstained,
        "reason": result.reason,
        "message": message,
        "sources": sources,
    }