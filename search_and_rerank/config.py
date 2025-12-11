from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Voyage AI Configuration
    voyage_api_key: str = "pa-oCVLLrg6rFLTodSYrkQW8n5UgYMcmCWWwYK03AUWJm8"
    voyage_model: str = "voyage-3.5"  # voyage-3.5-lite, voyage-3, voyage-code-3
    voyage_batch_size: int = 128  # Max batch size for Voyage AI (hard limit, cannot exceed)
    voyage_reranker_model: str = os.getenv("VOYAGE_RERANKER_MODEL", "rerank-2.5-lite")  # rerank-2.5-lite, rerank-2.5
    
    # Vector dimensions (voyage-3 produces 1024-dim vectors)
    vector_dimension: int = 1024
    
    qdrant_url: str = os.getenv("QDRANT_URL", "http://35.196.162.113:80")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "candidates_store")
    qdrant_host: str | None = os.getenv("QDRANT_HOST")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "80"))
    
    # App Configuration
    app_name: str = "Search and Rerank Service"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

