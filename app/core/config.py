"""Application configuration — reads from environment / .env file."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # --- App ---
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- OpenAI LLM ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"   # swap to gpt-4o, gpt-3.5-turbo, etc.
    OPENAI_MAX_TOKENS: int = 1024
    OPENAI_TEMPERATURE: float = 0.2

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- SQLite (local file DB) ---
    SQLITE_DB_URL: str = "sqlite:///./rag.db"

    # --- ChromaDB (local persistent) ---
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "rag_documents"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # --- Document ingestion ---
    MAX_DOCUMENTS: int = 20
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES_PER_DOC: int = 1000
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    UPLOAD_DIR: str = "./uploads"

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
