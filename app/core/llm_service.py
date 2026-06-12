"""
app/core/llm_service.py
LLM service — OpenAI only.
"""

import logging
from typing import Any, Dict, List

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document Q&A assistant.

Rules:
1. Answer ONLY using the provided document chunks below.
2. Always cite the source: mention the document name and chunk index.
3. If the answer is not in the chunks, say: "I could not find that information in the provided documents."
4. Be concise and accurate. Avoid speculation.
"""


def _get_client() -> OpenAI:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _build_user_message(query: str, chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        doc_name = meta.get("original_name", meta.get("doc_id", "unknown"))
        chunk_idx = meta.get("chunk_index", i)
        similarity = chunk.get("similarity", 0.0)
        context_parts.append(
            f"[Chunk {i + 1} | Doc: {doc_name} | Chunk #{chunk_idx} | Similarity: {similarity:.3f}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"Document chunks:\n\n{context}\n\n---\n\nQuestion: {query}"


def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer using OpenAI chat completions.

    Returns a dict with keys:
      - answer        (str)
      - model         (str)
      - sources       (list of dicts: filename / document_id / relevance_score)
      - tokens_used   (dict: prompt / completion / total  — or None on error)
    """
    client = _get_client()
    user_msg = _build_user_message(query, context_chunks)

    logger.info("Calling OpenAI model '%s'…", settings.OPENAI_MODEL)

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    answer = response.choices[0].message.content.strip()

    # Token usage
    tokens_used = None
    if response.usage:
        tokens_used = {
            "prompt":     response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total":      response.usage.total_tokens,
        }

    # Deduplicate sources by document_id
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        if doc_id not in seen:
            seen.add(doc_id)
            sources.append(
                {
                    "filename":        meta.get("original_name", doc_id),
                    "document_id":     doc_id,
                    "relevance_score": round(chunk.get("similarity", 0.0), 4),
                }
            )

    logger.info(
        "Answer generated | model=%s | tokens=%s | sources=%s",
        settings.OPENAI_MODEL,
        response.usage.total_tokens if response.usage else "n/a",
        [s["filename"] for s in sources],
    )

    return {
        "answer":      answer,
        "model":       settings.OPENAI_MODEL,
        "sources":     sources,
        "tokens_used": tokens_used,
    }