"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Document Schemas ──────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    page_count: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class DeleteResponse(BaseModel):
    message: str
    document_id: str


# ── Query Schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    document_ids: Optional[List[str]] = Field(
        None, description="Filter to specific document IDs (optional)"
    )
    top_k: Optional[int] = Field(
        5, ge=1, le=20, description="Number of chunks to retrieve"
    )


class SourceDocument(BaseModel):
    filename: str
    document_id: str
    relevance_score: float


class TokenUsage(BaseModel):
    prompt: int
    completion: int
    total: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument]
    model: str
    tokens_used: Optional[TokenUsage] = None
    chunks_retrieved: int


# ── Health Schemas ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: Dict[str, Any]
    total_documents: int
