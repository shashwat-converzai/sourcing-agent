from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Voyage AI Configuration
    voyage_api_key: str  # Required: Set via VOYAGE_API_KEY environment variable
    voyage_model: str = "voyage-3.5"  # voyage-3.5-lite, voyage-3, voyage-code-3
    voyage_batch_size: int = 128  # Max batch size for Voyage AI (hard limit, cannot exceed)
    
    # Processing chunk size (can be larger than Voyage batch size)
    processing_chunk_size: int = int(os.getenv("PROCESSING_CHUNK_SIZE", "1000"))  # Process this many candidates at once
    
    # Vector dimensions (voyage-3 produces 1024-dim vectors)
    vector_dimension: int = 1024
    
    # Qdrant Configuration
    qdrant_url: str  # Required: Set via QDRANT_URL environment variable
    qdrant_api_key: str | None = None  # Optional: Set via QDRANT_API_KEY environment variable
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "candidates_store")
    qdrant_host: str | None = None  # Optional: Set via QDRANT_HOST environment variable
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "80"))

    # Anthropic Claude Configuration
    anthropic_api_key: str  # Required: Set via ANTHROPIC_API_KEY environment variable
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")  # claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022, claude-3-opus-20240229
    anthropic_batch_size: int = int(os.getenv("ANTHROPIC_BATCH_SIZE", "128"))  # Batch size for summarization (can be up to 1000+)
    anthropic_max_workers: int = int(os.getenv("ANTHROPIC_MAX_WORKERS", "50"))  # Max concurrent API calls for summarization
    
    # App Configuration
    app_name: str = "Vector Ingestion Service"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
