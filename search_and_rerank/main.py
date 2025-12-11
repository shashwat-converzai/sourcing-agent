from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging

from .config import Settings, get_settings
from .models import JobDescriptionSearchRequest, RerankRequest
from .services.voyage_embeddings import VoyageEmbeddingService
from .services.qdrant_storage import QdrantStorageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global service instances
voyage_service: VoyageEmbeddingService | None = None
qdrant_service: QdrantStorageService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global voyage_service, qdrant_service
    
    settings = get_settings()
    
    logger.info("Initializing Voyage AI embedding service...")
    voyage_service = VoyageEmbeddingService(settings)
    
    logger.info("Initializing Qdrant storage service...")
    qdrant_service = QdrantStorageService(settings)
    
    logger.info("Search and Rerank Service started successfully")
    yield
    
    logger.info("Shutting down Search and Rerank Service")


app = FastAPI(
    title="Search and Rerank Service",
    description="""
    ## Search and Rerank Service API
    
    A FastAPI service for searching candidates using job descriptions.
    
    ### Features:
    - **Semantic Search**: Search candidates using job description with Voyage AI embeddings
    - **Reranking**: Rerank candidates based on job description relevance using Voyage AI reranker
    
    ### Endpoints:
    - `/search/job-description/{collection_name}` - Search candidates by job description
    - `/rerank` - Rerank candidates based on job description
    
    ### Documentation:
    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI JSON: `/openapi.json`
    """,
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/search/job-description/{collection_name}",
    tags=["Search"],
    summary="Search candidates by job description",
    description="""
    Search for top matching candidates based on a job description text.
    
    **Features:**
    - Semantic similarity search using job description embeddings
    - Returns top 200 most similar candidates by default
    - Uses optimized HNSW search parameters (ef=300) for quality
    - Optional filters by experience, location, skills, etc.
    
    **HNSW Configuration:**
    - M=32 (bi-directional links, balances speed/quality)
    - ef_construction=200 (construction quality)
    - ef=300 (search quality, 200-500 recommended)
    - Distance: Cosine similarity
    
    **Example Job Description:**
    "We are looking for a Senior Oracle Fusion Financials Consultant with 5+ years of experience
    in GL, AP, AR, FA, and CM modules. Experience with full lifecycle implementations,
    data migration using FBDI, UAT/SIT testing, and production support required."
    """,
    response_description="Search results with candidate matches and similarity scores",
)
async def search_by_job_description(collection_name: str, request: JobDescriptionSearchRequest):
    """
    Search for candidates matching a job description.
    
    Args:
        collection_name: Name of the collection to search
        request: Job description search request with job description text and optional filters
        
    Returns:
        Dictionary with query, total results, and list of matching candidates with similarity scores
    """
    if voyage_service is None or qdrant_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Generate query embedding from job description
    try:
        query_vector = voyage_service.generate_query_embedding(request.job_description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job description embedding failed: {str(e)}")
    
    # Build filters
    filters = {}
    if request.min_experience_years:
        filters["years_experience_total"] = {"gte": request.min_experience_years}
    # Note: Additional filters can be added here (location, skills, primary_role_code, etc.)
    
    # Validate ef parameter (should be between 200-500 for optimal quality)
    ef = request.ef if request.ef else 300
    if ef < 200:
        ef = 200
        logger.warning(f"ef parameter too low ({request.ef}), using minimum 200")
    elif ef > 500:
        ef = 500
        logger.warning(f"ef parameter too high ({request.ef}), using maximum 500")
    
    # Search with HNSW ef parameter for quality
    try:
        results = qdrant_service.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=request.limit,
            filters=filters if filters else None,
            ef=ef,  # HNSW ef parameter for search quality
        )
        
        return {
            "job_description": request.job_description,
            "total_results": len(results),
            "limit": request.limit,
            "ef": ef,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post(
    "/rerank",
    tags=["Rerank"],
    summary="Rerank candidates by job description",
    description="""
    Rerank a list of candidates based on their relevance to a job description using Voyage AI reranker.
    
    **Features:**
    - Uses Voyage AI reranker model (rerank-2.5-lite by default)
    - Returns candidates sorted by relevance score (highest first)
    - Each candidate should include text data that can be compared to the job description
    
    **Input:**
    - `job_description`: The job description text
    - `candidates`: List of candidate objects. Each candidate should have a text field or be a string.
                    If candidate is a dict, the service will try to extract text from common fields.
    
    **Output:**
    - Reranked list of candidates with relevance scores
    - Original candidate data preserved with added relevance_score
    
    **Example Request:**
    ```json
    {
        "job_description": "Senior Oracle Fusion Financials Consultant with 5+ years experience",
        "candidates": [
            {"candidate_id": "123", "text": "Oracle Fusion consultant with 6 years experience..."},
            {"candidate_id": "456", "text": "Software engineer with Java experience..."}
        ]
    }
    ```
    """,
    response_description="Reranked candidates with relevance scores",
)
async def rerank_candidates(request: RerankRequest):
    """
    Rerank candidates based on their relevance to a job description.
    
    Args:
        request: Rerank request with job description and list of candidates
        
    Returns:
        Dictionary with reranked candidates and their relevance scores
    """
    if voyage_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    if not request.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")
    
    # Extract text from candidates
    # Handle both string candidates and dict candidates
    candidate_texts = []
    candidate_metadata = []
    
    for idx, candidate in enumerate(request.candidates):
        if isinstance(candidate, str):
            # Direct string candidate
            candidate_texts.append(candidate)
            candidate_metadata.append({"index": idx, "original": candidate})
        elif isinstance(candidate, dict):
            # Dict candidate - try to extract text from common fields
            text = None
            # Try common text fields
            for field in ["text", "resume_text", "resume_data", "description", "summary", "content"]:
                if field in candidate and candidate[field]:
                    text = str(candidate[field])
                    break
            
            # If no text field found, try to use the whole dict as string (fallback)
            if not text:
                # Try to build text from common candidate fields
                text_parts = []
                for key in ["current_role", "skills", "years_experience_total"]:
                    if key in candidate and candidate[key]:
                        text_parts.append(f"{key}: {candidate[key]}")
                text = " ".join(text_parts) if text_parts else str(candidate)
            
            candidate_texts.append(text)
            candidate_metadata.append({"index": idx, "original": candidate})
        else:
            # Fallback: convert to string
            candidate_texts.append(str(candidate))
            candidate_metadata.append({"index": idx, "original": candidate})
    
    # Perform reranking
    try:
        reranked_results = voyage_service.rerank(
            query=request.job_description,
            documents=candidate_texts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
    
    # Combine reranked results with original candidate data
    reranked_candidates = []
    for result in reranked_results:
        original_idx = result["index"]
        original_candidate = candidate_metadata[original_idx]["original"]
        
        # Add relevance score to candidate data
        if isinstance(original_candidate, dict):
            candidate_with_score = original_candidate.copy()
            candidate_with_score["relevance_score"] = result["relevance_score"]
        else:
            candidate_with_score = {
                "candidate": original_candidate,
                "relevance_score": result["relevance_score"]
            }
        
        reranked_candidates.append(candidate_with_score)
    
    return {
        "job_description": request.job_description,
        "total_candidates": len(request.candidates),
        "reranked_candidates": reranked_candidates,
    }

