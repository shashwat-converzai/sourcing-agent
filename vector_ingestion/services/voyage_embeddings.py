import voyageai
from typing import List
import logging
import time

from ..config import Settings

logger = logging.getLogger(__name__)


class VoyageEmbeddingService:
    """Service for generating embeddings using Voyage AI batch inference."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = voyageai.Client(api_key=settings.voyage_api_key)
        self.model = settings.voyage_model
        self.batch_size = settings.voyage_batch_size
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Voyage AI.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches if needed
        max_retries = 3  # Total of 4 attempts: initial + 3 retries
        wait_times = [5, 10, 30]  # Wait times in seconds for each retry
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            for attempt in range(max_retries + 1):
                try:
                    result = self.client.embed(
                        texts=batch,
                        model=self.model,
                        input_type="document"  # Use "document" for storage, "query" for search
                    )
                    all_embeddings.extend(result.embeddings)
                    
                    logger.info(f"Generated embeddings for batch {batch_num}, "
                               f"texts: {len(batch)}, total tokens: {result.total_tokens}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = (
                        "rate limit" in error_str or 
                        "429" in error_str or
                        "too many requests" in error_str or
                        "quota" in error_str
                    )
                    
                    if attempt < max_retries and (is_rate_limit or True):  # Retry on any error
                        wait_time = wait_times[attempt]  # Get wait time for this attempt
                        logger.warning(f"Error generating embeddings for batch {batch_num} (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Error generating embeddings for batch {batch_num} after {attempt + 1} attempts: {e}")
                        raise
        
        return all_embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses input_type="query" for optimized retrieval.
        
        Args:
            query: Search query string
            
        Returns:
            Embedding vector as list of floats
        """
        max_retries = 3  # Total of 4 attempts: initial + 3 retries
        wait_times = [5, 10, 30]  # Wait times in seconds for each retry
        
        for attempt in range(max_retries + 1):
            try:
                result = self.client.embed(
                    texts=[query],
                    model=self.model,
                    input_type="query"
                )
                return result.embeddings[0]
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str or 
                    "429" in error_str or
                    "too many requests" in error_str or
                    "quota" in error_str
                )
                
                if attempt < max_retries and (is_rate_limit or True):  # Retry on any error
                    wait_time = wait_times[attempt]  # Get wait time for this attempt
                    logger.warning(f"Error generating query embedding (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error generating query embedding after {attempt + 1} attempts: {e}")
                    raise

