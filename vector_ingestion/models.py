"""
Pydantic models for candidate ingestion API.
Simplified for new flat data structure.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, RootModel, ConfigDict


# ============================================================================
# Core Request/Response Models
# ============================================================================

class IngestionResult(BaseModel):
    """Result of ingesting a single candidate."""
    candidate_id: str
    success: bool
    vector_id: Optional[str] = None
    error: Optional[str] = None


class BatchIngestionResponse(BaseModel):
    """Response for batch ingestion."""
    total: int
    successful: int
    failed: int
    results: List[IngestionResult]


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    limit: int = 10
    min_experience_years: Optional[float] = None


class JobDescriptionSearchRequest(BaseModel):
    """Job description search request model."""
    job_description: str
    limit: int = 200
    min_experience_years: Optional[float] = None
    ef: Optional[int] = 300  # HNSW ef parameter (200-500 recommended)


# ============================================================================
# Candidate Profile Models (Simple Structure)
# ============================================================================

class TitleExp(BaseModel):
    """Title and experience entry."""
    model_config = ConfigDict(extra='allow')
    
    title: Optional[str] = None
    exp: Optional[str] = None
    months: Optional[int] = None


class EducationEntry(BaseModel):
    """Education entry."""
    model_config = ConfigDict(extra='allow')
    
    SCHOOL: Optional[str] = None
    MAJOR: Optional[str] = None
    DEGREE: Optional[str] = None
    YEAR: Optional[str] = None


class CandidateRawProfile(BaseModel):
    """Raw candidate profile - main model for ingestion."""
    model_config = ConfigDict(extra='allow')
    
    candidate_id: str
    client_id: Optional[str] = None
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    primary_role_code: Optional[str] = None
    current_role: Optional[str] = None
    skills: Optional[List[str]] = []
    country: Optional[str] = None
    city: Optional[str] = None
    address1: Optional[str] = None
    address2: Optional[str] = None
    workphone: Optional[str] = None
    homephone: Optional[str] = None
    cellphone: Optional[str] = None
    phone1: Optional[str] = None
    phone2: Optional[str] = None
    phone3: Optional[str] = None
    phone4: Optional[str] = None
    alternateemail: Optional[str] = None
    years_experience_total: Optional[float] = None
    title_exp: Optional[List[TitleExp]] = []
    education: Optional[List[EducationEntry]] = []
    resume_data: Optional[str] = None  # String containing resume text


# ============================================================================
# Batch Request Model (RootModel for array handling)
# ============================================================================

class BatchIngestionRequest(RootModel[List[CandidateRawProfile]]):
    """Batch ingestion request - wraps a list of candidates."""
    root: List[CandidateRawProfile]
    
    def __iter__(self):
        return iter(self.root)
    
    def __len__(self):
        return len(self.root)
    
    def __getitem__(self, index):
        return self.root[index]
