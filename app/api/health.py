"""
Health check endpoint.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.document import Document, DocumentStatus
from app.core.vector_store import get_collection_stats
from app.schemas.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """System health and statistics."""
    total_docs = db.query(Document).filter(
        Document.status == DocumentStatus.READY
    ).count()

    vector_stats = get_collection_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store=vector_stats,
        total_documents=total_docs,
    )
