"""
app/api/query.py
Query API: retrieve relevant chunks then generate an answer via Gemini/OpenAI.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.llm_service import generate_response
from app.core.vector_store import query_similar_chunks
from app.db.database import get_db
from app.models.document import Document, DocumentStatus
from app.schemas.schemas import QueryRequest, QueryResponse, SourceDocument, TokenUsage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=QueryResponse)
def query_documents(request: QueryRequest, db: Session = Depends(get_db)):
    """
    RAG query endpoint:
      1. Embed the user query
      2. Retrieve top-k similar chunks from ChromaDB
      3. Send chunks + query to Gemini/OpenAI
      4. Return the answer with sources
    """

  
    ready_count = db.query(Document).filter(
        Document.status == DocumentStatus.READY
    ).count()

    if ready_count == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No documents are ready for querying. "
                "Upload a document and wait until its status is 'ready'."
            ),
        )

    
    if request.document_ids:
        for doc_id in request.document_ids:
            doc = db.query(Document).filter(
                Document.id == doc_id,
                Document.status == DocumentStatus.READY,
            ).first()
            if not doc:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Document '{doc_id}' not found or not ready. "
                        "Check status with GET /api/v1/documents/{id}"
                    ),
                )

    logger.info(f"━━━ QUERY: '{request.query[:100]}' ━━━")

    # ── Step 1: Retrieve relevant chunks ────────────────────────────
    try:
        chunks = query_similar_chunks(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except Exception as exc:
        logger.error(f"Vector store retrieval failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to retrieve chunks: {exc}. "
                "If the vector store is empty, re-upload your documents."
            ),
        )

    logger.info(f"Retrieved {len(chunks)} chunks for query")

    if len(chunks) == 0:
        logger.warning(
            "No chunks retrieved — returning early with a helpful message."
        )
        return QueryResponse(
            query=request.query,
            answer=(
                "No relevant content was found in the uploaded documents for your query. "
                "Try rephrasing, or check that the document was processed successfully "
                "(status should be 'ready' with chunk_count > 0)."
            ),
            sources=[],
            model="none",
            tokens_used=None,
            chunks_retrieved=0,
        )

    # ── Step 2: Generate answer via LLM ─────────────────────────────
    try:
        result = generate_response(query=request.query, context_chunks=chunks)
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"LLM error: {exc}",
        )

    sources = [
        SourceDocument(
            filename        = s["filename"],
            document_id     = s["document_id"],
            relevance_score = s["relevance_score"],
        )
        for s in result.get("sources", [])
    ]

    tokens = None
    if result.get("tokens_used"):
        tokens = TokenUsage(**result["tokens_used"])

    logger.info(
        f"Answer generated | model={result['model']} "
        f"| chunks_used={len(chunks)} | sources={[s.filename for s in sources]}"
    )

    return QueryResponse(
        query           = request.query,
        answer          = result["answer"],
        sources         = sources,
        model           = result["model"],
        tokens_used     = tokens,
        chunks_retrieved= len(chunks),
    )
