"""
app/api/documents.py
Documents API: upload, list, retrieve metadata, delete.
"""

import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.document_processor import chunk_text, extract_text_from_file, validate_file
from app.core.vector_store import add_document_chunks, delete_document_chunks
from app.db.database import get_db
from app.models.document import Document, DocumentStatus
from app.schemas.schemas import DeleteResponse, DocumentListResponse, DocumentResponse

logger = logging.getLogger(__name__)
router = APIRouter()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


# ── Background processing ─────────────────────────────────────────

def process_document_background(document_id: str, file_path: str, filename: str) -> None:
    """
    Runs in a thread after upload:
      1. Extract text from the file
      2. Split into overlapping chunks
      3. Embed chunks and store in ChromaDB
      4. Mark document as READY in SQLite
    """
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            logger.error(f"Document {document_id} not found in DB — aborting.")
            return

        logger.info(f"━━━ START processing '{filename}' ({document_id}) ━━━")

        # Step 1: Extract
        logger.info(f"[1/3] Extracting text from '{filename}'...")
        text, page_count = extract_text_from_file(file_path, filename)
        logger.info(f"[1/3] ✓ Extracted {len(text):,} characters across {page_count} page(s)")

        if not text.strip():
            raise ValueError("No extractable text found — file may be image-only or corrupted.")

        # Step 2: Chunk
        logger.info(f"[2/3] Chunking text (chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
        chunks = chunk_text(text)
        logger.info(f"[2/3] ✓ Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("Chunking produced 0 chunks — document may be too short.")

        # Step 3: Embed + Store
        logger.info(f"[3/3] Embedding and storing {len(chunks)} chunks in ChromaDB...")
        stored_count = add_document_chunks(document_id, chunks, filename)
        logger.info(f"[3/3] ✓ Stored {stored_count} chunks in vector store")

        # Update DB record
        doc.page_count  = page_count
        doc.chunk_count = stored_count
        doc.status      = DocumentStatus.READY
        db.commit()

        logger.info(
            f"━━━ DONE '{filename}' — "
            f"pages={page_count}, chunks={stored_count}, status=READY ━━━"
        )

    except Exception as exc:
        logger.error(f"━━━ FAILED processing '{filename}': {exc} ━━━", exc_info=True)
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status        = DocumentStatus.FAILED
            doc.error_message = str(exc)
            db.commit()
    finally:
        db.close()


# ── Routes ────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a document (PDF/TXT/DOCX/MD). Processing happens in the background."""

    # Enforce document limit
    active = db.query(Document).filter(Document.status != DocumentStatus.FAILED).count()
    if active >= settings.MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {settings.MAX_DOCUMENTS} documents reached. Delete some first.",
        )

    content   = await file.read()
    file_size = len(content)

    try:
        validate_file(file.filename, file_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Save file to disk
    document_id    = str(uuid.uuid4())
    ext            = os.path.splitext(file.filename)[1].lower()
    saved_filename = f"{document_id}{ext}"
    file_path      = os.path.join(settings.UPLOAD_DIR, saved_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record with PROCESSING status
    doc = Document(
        id                = document_id,
        filename          = saved_filename,
        original_filename = file.filename,
        file_size         = file_size,
        status            = DocumentStatus.PROCESSING,
        file_path         = file_path,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Queue background processing
    background_tasks.add_task(
        process_document_background,
        document_id = document_id,
        file_path   = file_path,
        filename    = file.filename,
    )

    logger.info(f"Upload queued: '{file.filename}' → {document_id}")
    return DocumentResponse(**doc.to_dict())


@router.get("/", response_model=DocumentListResponse)
def list_documents(status: Optional[str] = None, db: Session = Depends(get_db)):
    """List all documents, optionally filtered by status (processing/ready/failed)."""
    query = db.query(Document)
    if status:
        try:
            query = query.filter(Document.status == DocumentStatus(status))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Use: processing, ready, failed",
            )
    docs = query.order_by(Document.created_at.desc()).all()
    return DocumentListResponse(
        documents=[DocumentResponse(**d.to_dict()) for d in docs],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get metadata and current status for a specific document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentResponse(**doc.to_dict())


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and all its vector embeddings."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    deleted_chunks = delete_document_chunks(document_id)

    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    db.delete(doc)
    db.commit()

    logger.info(f"Deleted '{doc.original_filename}' + {deleted_chunks} chunks")
    return DeleteResponse(
        message=f"Document '{doc.original_filename}' and {deleted_chunks} chunks deleted.",
        document_id=document_id,
    )
