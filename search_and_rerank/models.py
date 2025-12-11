"""
Pydantic models for search API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class JobDescriptionSearchRequest(BaseModel):
    """Job description search request model."""
    job_description: str
    limit: int = 200
    min_experience_years: Optional[float] = None
    ef: Optional[int] = 300  # HNSW ef parameter (200-500 recommended)


class RerankRequest(BaseModel):
    """Rerank request model."""
    job_description: str
    candidates: List[Dict[str, Any]]  # List of candidate objects with their data

