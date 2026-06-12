"""Vector store — ChromaDB persistent local client."""

import logging
from typing import List, Dict, Any
from functools import lru_cache

import chromadb
from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    logger.info("ChromaDB persistent client at %s", settings.CHROMA_PERSIST_DIR)
    return client


def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


class VectorStoreService:

    def add_chunks(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        collection = _get_collection()
        ids = [f"{doc_id}__chunk__{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks for doc %s", len(chunks), doc_id)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        collection = _get_collection()
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def delete_document(self, doc_id: str) -> None:
        collection = _get_collection()
        results = collection.get(where={"doc_id": {"$eq": doc_id}}, include=[])
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks for doc %s", len(results["ids"]), doc_id)

    def collection_stats(self) -> Dict[str, Any]:
        collection = _get_collection()
        return {"total_chunks": collection.count(), "collection": settings.CHROMA_COLLECTION}