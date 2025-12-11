from fastapi import FastAPI, HTTPException, status, Body, Request, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, ValidationError
import logging
import asyncio
import time
import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Settings, get_settings
from .models import (
    CandidateRawProfile,
    BatchIngestionRequest,
    BatchIngestionResponse,
    IngestionResult,
    SearchRequest,
    JobDescriptionSearchRequest,
)
from .services.voyage_embeddings import VoyageEmbeddingService
from .services.qdrant_storage import QdrantStorageService
from .services.resume_summarizer import ResumeSummarizerService
from .utils_ingestion import (
    get_token_count,
    build_candidate_text,
    build_candidate_payload,
)
from . import utils_ingestion

# Configure logging to both console and file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Use a single log file that appends (daily log file)
log_filename = log_dir / f"ingestion_{datetime.now().strftime('%Y%m%d')}.log"

# Configure root logger (for general logs - goes to both file and console)
# FileHandler with mode='a' appends to existing file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for summaries (file only, no console)
summary_logger = logging.getLogger("summaries")
summary_logger.setLevel(logging.INFO)
# Remove any existing handlers to avoid duplicates
summary_logger.handlers = []
# Add only file handler (no console) - append mode
summary_file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
summary_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
summary_logger.addHandler(summary_file_handler)
summary_logger.propagate = False  # Don't propagate to root logger to avoid console output

logger.info(f"Logging initialized. Log file: {log_filename} (appending mode)")

# Global service instances
voyage_service: VoyageEmbeddingService | None = None
qdrant_service: QdrantStorageService | None = None
summarizer_service: ResumeSummarizerService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global voyage_service, qdrant_service, summarizer_service
    
    settings = get_settings()
    
    logger.info("Initializing Voyage AI embedding service...")
    voyage_service = VoyageEmbeddingService(settings)
    
    logger.info("Initializing Qdrant storage service...")
    qdrant_service = QdrantStorageService(settings)
    
    logger.info("Initializing Resume Summarizer service...")
    summarizer_service = ResumeSummarizerService(settings)
    
    logger.info("Vector Ingestion Service started successfully")
    yield
    
    logger.info("Shutting down Vector Ingestion Service")


app = FastAPI(
    title="Vector Ingestion Service",
    description="""
    ## Vector Ingestion Service API
    
    A high-performance FastAPI service for ingesting candidate profiles into Qdrant vector database 
    using Voyage AI embeddings with automatic batch processing.
    
    ### Features:
    - **Automatic Batch Chunking**: Handles any batch size, automatically chunks into optimal 128-candidate batches
    - **Parallel Preprocessing**: Processes up to 32 candidates simultaneously
    - **Voyage AI Integration**: Uses voyage-3.5-lite for efficient embeddings
    - **Qdrant Vector Storage**: Stores embeddings with rich metadata for filtering
    
    ### Endpoints:
    - `/ingest` - Batch ingest candidate profiles (auto-chunks large batches)
    - `/search/{collection_name}` - Semantic search for similar candidates
    - `/search/job-description/{collection_name}` - Search top 200 candidates by job description
    - `/health` - Service health check with status of all dependencies
    - `/collections/{collection_name}` - Get collection information
    - `/collections/{collection_name}/problematic` - Find problematic/failed candidates
    - `/candidates/{collection_name}/{candidate_id}` - Delete a candidate
    
    ### Documentation:
    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI JSON: `/openapi.json`
    """,
    version="1.0.0",
    lifespan=lifespan,
    tags_metadata=[
        {
            "name": "Ingestion",
            "description": "Endpoints for ingesting candidate profiles into the vector database.",
        },
        {
            "name": "Search",
            "description": "Endpoints for searching and retrieving candidate profiles.",
        },
        {
            "name": "Health",
            "description": "Health check and service status endpoints.",
        },
        {
            "name": "Collections",
            "description": "Endpoints for managing Qdrant collections.",
        },
    ],
)


def custom_openapi():
    """Custom OpenAPI schema to add request body schema for /ingest endpoint."""
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add request body schema for /ingest endpoint
    ingest_path = openapi_schema["paths"].get("/ingest", {})
    if "post" in ingest_path:
        ingest_path["post"]["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/BatchIngestionRequest"},
                            {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/CandidateRawProfile"},
                                "description": "Direct array format (use with collection_name query param)"
                            }
                        ]
                    },
                    "examples": {
                        "wrapped_format": {
                            "summary": "Wrapped format (recommended)",
                            "value": {
                                "candidates": [
                                    {
                                        "candidateId": "8284346",
                                        "clientId": "8428fbd0-a3e7-40f5-a274-27425226f96a",
                                        "resumeData": {
                                            "ContactInformation": {
                                                "CandidateName": {"FormattedName": "John Doe"},
                                                "EmailAddresses": ["john@example.com"]
                                            }
                                        }
                                    }
                                ],
                                "collection_name": "candidates_store"
                            }
                        },
                        "array_format": {
                            "summary": "Direct array format",
                            "value": [
                                {
                                    "candidateId": "8284346",
                                    "clientId": "8428fbd0-a3e7-40f5-a274-27425226f96a",
                                    "resumeData": {
                                        "ContactInformation": {
                                            "CandidateName": {"FormattedName": "John Doe"},
                                            "EmailAddresses": ["john@example.com"]
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def _preprocess_candidate_build_text(candidate: CandidateRawProfile) -> Tuple[str, Dict[str, Any], str]:
    """
    Preprocess a single candidate: build text and payload (without summarization).
    Summarization will be done in batch later.
    This function is designed to run in parallel.
    Returns: (text, payload, candidate_id)
    """
    try:
        # Get candidate_id from new or old format
        candidate_id = getattr(candidate, 'candidate_id', None) or getattr(candidate, 'candidateId', 'unknown')
        
        # Build text from resume_data (new format) or resumeData (legacy)
        text = build_candidate_text(candidate)
        
        # Remove PII from text using candidate's contact info
        text = utils_ingestion._remove_pii_from_text(text, None, None, candidate)
        
        # Build payload (initially set text_summarized=False, will be updated if summarization is used)
        payload = build_candidate_payload(candidate, text_summarized=False)
        
        return text, payload, candidate_id
    except Exception as e:
        candidate_id = getattr(candidate, 'candidate_id', None) or getattr(candidate, 'candidateId', 'unknown')
        logger.error(f"Error preprocessing candidate {candidate_id}: {e}")
        raise


async def _preprocess_candidates_parallel(candidates: List[CandidateRawProfile]) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[CandidateRawProfile]]:
    """
    Preprocess all candidates in parallel (without summarization).
    
    Process:
    1. Build all texts in parallel (without summarization)
    2. Return all texts, payloads, ids, and candidate references
    
    Maintains original order of candidates.
    Returns: (texts, payloads, ids, candidates_list)
    """
    
    loop = asyncio.get_event_loop()
    
    # Build all texts and payloads in parallel (without summarization)
    max_workers = min(32, len(candidates))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all text building tasks
        futures = [
            loop.run_in_executor(executor, _preprocess_candidate_build_text, candidate)
            for candidate in candidates
        ]
        
        # Wait for all to complete and collect results in order
        texts = []
        payloads = []
        ids = []
        candidates_list = []  # Keep reference to candidates for PII removal and summarization
        
        for i, future in enumerate(futures):
            candidate = candidates[i]
            try:
                text, payload, candidate_id = await future
                texts.append(text)
                payloads.append(payload)
                ids.append(candidate_id)
                candidates_list.append(candidate)
            except Exception as e:
                candidate_id = getattr(candidate, 'candidate_id', 'unknown')
                logger.error(f"Failed to preprocess candidate {candidate_id}: {e}")
                # Add fallback placeholders to maintain order
                texts.append(f"Candidate ID: {candidate_id}")
                payloads.append({
                    "candidate_id": candidate_id,
                    "client_id": getattr(candidate, 'client_id', None)
                })
                ids.append(candidate_id)
                candidates_list.append(candidate)
    
    return texts, payloads, ids, candidates_list


async def _process_chunk(
    chunk: List[CandidateRawProfile],
    chunk_num: int,
    total_chunks: int,
    collection_name: str,
    summarization: bool = True,
) -> Tuple[List[IngestionResult], int, int]:
    """
    Process a single chunk of candidates (configurable size, default 1000).
    Embeddings are automatically split into Voyage-compatible 128 sub-batches.
    Returns: (results, successful_count, failed_count)
    """
    settings = get_settings()
    PROCESSING_CHUNK_SIZE = settings.processing_chunk_size
    chunk_start_idx = (chunk_num - 1) * PROCESSING_CHUNK_SIZE
    chunk_end_idx = chunk_start_idx + len(chunk)
    
    logger.info(f"Processing chunk {chunk_num}/{total_chunks}: candidates {chunk_start_idx}-{chunk_end_idx-1} ({len(chunk)} candidates)")
    
    chunk_start_time = time.time()
    results: List[IngestionResult] = []
    successful = 0
    failed = 0
    
    try:
        # Step 1: Preprocess chunk in parallel (build texts without summarization)
        preprocess_start = time.time()
        texts, payloads, ids, candidates_list = await _preprocess_candidates_parallel(chunk)
        preprocess_time = time.time() - preprocess_start
        
        logger.info(f"Chunk {chunk_num} preprocessing complete in {preprocess_time:.2f}s")
        
        # Step 2: Process in batches of 128: summarize → embed → upsert
        VOYAGE_BATCH_SIZE = 128
        TOKEN_THRESHOLD = 800
        
        # Initialize timing variables
        embed_time = 0.0
        qdrant_time = 0.0
        
        # Identify texts that need summarization (only if summarization is enabled)
        texts_to_summarize_info = []  # List of (idx, text, candidate) tuples
        if summarization:
            for idx, text in enumerate(texts):
                token_count = get_token_count(text)
                if token_count > TOKEN_THRESHOLD:
                    texts_to_summarize_info.append((idx, text, candidates_list[idx]))
            
            summary_logger.info(f"Chunk {chunk_num}: {len(texts_to_summarize_info)} texts need summarization out of {len(texts)} total")
        else:
            logger.info(f"Chunk {chunk_num}: Summarization disabled, embedding raw texts directly")
        
        # Process in batches of 128
        for batch_start in range(0, len(texts), VOYAGE_BATCH_SIZE):
            batch_end = min(batch_start + VOYAGE_BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            batch_candidates = candidates_list[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start // VOYAGE_BATCH_SIZE + 1}: candidates {batch_start}-{batch_end-1} ({len(batch_texts)} candidates)")
            
            # Step 2a: Summarize texts that need it (only if summarization is enabled)
            batch_texts_to_summarize = []
            batch_indices_to_summarize = []
            batch_candidates_to_summarize = []
            
            if summarization:
                for local_idx, (global_idx, text, candidate) in enumerate(texts_to_summarize_info):
                    if batch_start <= global_idx < batch_end:
                        local_idx_in_batch = global_idx - batch_start
                        batch_texts_to_summarize.append(text)
                        batch_indices_to_summarize.append(local_idx_in_batch)
                        batch_candidates_to_summarize.append(candidate)
            
            if batch_texts_to_summarize and summarizer_service and summarization:
                summary_logger.info(f"  Summarizing {len(batch_texts_to_summarize)} texts in this batch")
                try:
                    # Summarize batch (max 128 async calls)
                    summarized_texts = await summarizer_service.summarize_batch(batch_texts_to_summarize)
                    
                    # Process summaries and update batch_texts
                    for i, (local_idx, summarized_text) in enumerate(zip(batch_indices_to_summarize, summarized_texts)):
                        candidate = batch_candidates_to_summarize[i]
                        original_text = batch_texts_to_summarize[i]
                        
                        # Validate summary
                        if summarized_text is None or not summarized_text.strip() or summarized_text.strip() == original_text.strip():
                            candidate_id = getattr(candidate, 'candidate_id', 'unknown')
                            summary_logger.warning(f"❌ SUMMARIZATION FAILED - Candidate ID: {candidate_id} - Will be excluded")
                            # Mark for exclusion by setting text to None
                            batch_texts[local_idx] = None
                            results.append(IngestionResult(
                                candidate_id=batch_ids[local_idx],
                                success=False,
                                error="Summarization failed",
                            ))
                            failed += 1
                            continue
                        
                        # Remove PII from summarized text (use candidate object for new format, legacy format fallback)
                        summarized_text = utils_ingestion._remove_pii_from_text(summarized_text, None, None, candidate)
                        
                        batch_texts[local_idx] = summarized_text
                        # Update payload to mark as summarized
                        batch_payloads[local_idx]["text_summarized"] = True
                        
                except Exception as e:
                    summary_logger.error(f"Error during summarization: {e}")
                    # Mark all that needed summarization as failed
                    for local_idx in batch_indices_to_summarize:
                        batch_texts[local_idx] = None
                        results.append(IngestionResult(
                            candidate_id=batch_ids[local_idx],
                            success=False,
                            error=f"Summarization error: {str(e)}",
                        ))
                        failed += 1
            
            # Step 2b: Filter out failed candidates (None texts) and set haiku_summarized metadata
            valid_texts = []
            valid_payloads = []
            valid_ids = []
            valid_indices = []  # Track original indices for mapping back
            
            for idx, text in enumerate(batch_texts):
                if text is not None:
                    # Set text_summarized metadata (False if summarization was disabled or not used)
                    if "text_summarized" not in batch_payloads[idx]:
                        batch_payloads[idx]["text_summarized"] = False
                    valid_texts.append(text)
                    valid_payloads.append(batch_payloads[idx])
                    valid_ids.append(batch_ids[idx])
                    valid_indices.append(idx)
            
            if not valid_texts:
                logger.warning(f"  Batch {batch_start // VOYAGE_BATCH_SIZE + 1}: All candidates failed, skipping embed/upsert")
                continue
            
            # Step 2c: Embed the valid texts (128 max)
            embed_start = time.time()
            embeddings = voyage_service.generate_embeddings(valid_texts)
            batch_embed_time = time.time() - embed_start
            embed_time += batch_embed_time
            logger.info(f"  Batch {batch_start // VOYAGE_BATCH_SIZE + 1} embeddings generated in {batch_embed_time:.2f}s ({len(embeddings)} vectors)")
            
            # Step 2d: Upsert to Qdrant (128 max)
            qdrant_start = time.time()
            try:
                stored_ids = qdrant_service.upsert_vectors(
                    collection_name=collection_name,
                    vectors=embeddings,
                    payloads=valid_payloads,
                    ids=None,  # Let Qdrant generate UUIDs automatically
                )
                batch_qdrant_time = time.time() - qdrant_start
                qdrant_time += batch_qdrant_time
                logger.info(f"  Batch {batch_start // VOYAGE_BATCH_SIZE + 1} upserted in {batch_qdrant_time:.2f}s")
                
                # Record successes
                for candidate_id, vector_id in zip(valid_ids, stored_ids):
                    if vector_id is not None:
                        results.append(IngestionResult(
                            candidate_id=candidate_id,
                            success=True,
                            vector_id=vector_id,
                        ))
                        successful += 1
                    else:
                        results.append(IngestionResult(
                            candidate_id=candidate_id,
                            success=False,
                            error="Upsert failed",
                        ))
                        failed += 1
            except Exception as e:
                logger.error(f"  Error upserting batch {batch_start // VOYAGE_BATCH_SIZE + 1}: {e}")
                # Mark all as failed
                for candidate_id in valid_ids:
                    results.append(IngestionResult(
                        candidate_id=candidate_id,
                        success=False,
                        error=f"Upsert error: {str(e)}",
                    ))
                    failed += 1
        
        chunk_total_time = time.time() - chunk_start_time
        logger.info(f"Chunk {chunk_num} complete in {chunk_total_time:.2f}s (preprocess: {preprocess_time:.2f}s, embed: {embed_time:.2f}s, qdrant: {qdrant_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"Failed to process chunk {chunk_num}: {e}")
        # Mark all in chunk as failed
        for candidate in chunk:
            candidate_id = getattr(candidate, 'candidate_id', 'unknown')
            results.append(IngestionResult(
                candidate_id=candidate_id,
                success=False,
                error=str(e),
            ))
            failed += 1
    
    return results, successful, failed


async def parse_ingestion_request(
    request: Request,
    collection_name: Optional[str] = "candidates_store"
) -> Tuple[List[CandidateRawProfile], str]:
    """
    Custom dependency to parse ingestion request.
    Handles:
    1. File path in request body: {"file_path": "path/to/file.json"}
    2. Array format: [...]
    3. Wrapped format: {"candidates": [...], "collection_name": "..."}
    """
    try:
        body = await request.json()
        
        # Check if it's a file path request
        if isinstance(body, dict) and "file_path" in body:
            file_path = body["file_path"]
            collection_name = body.get("collection_name", collection_name)
            
            # Read JSON file
            import os
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # File should contain a list of candidates
            if isinstance(file_data, list):
                validated = BatchIngestionRequest.model_validate(file_data, strict=False)
                return validated.root, collection_name
            else:
                raise HTTPException(
                    status_code=422,
                    detail="File must contain a JSON array of candidates"
                )
        
        # Preferred format: raw array of candidates
        if isinstance(body, list):
            validated = BatchIngestionRequest.model_validate(body, strict=False)
            return validated.root, collection_name
        elif isinstance(body, dict):
            # Legacy wrapped format support: {"candidates": [...], "collection_name": "..."}
            if "candidates" in body:
                validated = BatchIngestionRequest.model_validate(body["candidates"], strict=False)
                return validated.root, body.get("collection_name", collection_name)
            else:
                # Treat a single object as one candidate
                validated = BatchIngestionRequest.model_validate([body], strict=False)
                return validated.root, collection_name
        else:
            raise HTTPException(
                status_code=422,
                detail="Invalid format. Expected either an array of candidates or an object with 'candidates' field."
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
    except ValidationError as e:
        # Log the full error for debugging
        logger.error(f"Validation error details: {e}")
        # Try to provide more helpful error messages
        error_details = []
        if hasattr(e, 'errors'):
            for err in e.errors():
                error_details.append(f"{err.get('loc', 'unknown')}: {err.get('msg', 'unknown error')}")
        raise HTTPException(
            status_code=422,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Error parsing request: {str(e)}"
        )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    qdrant_url: str
    qdrant_status: str
    voyage_status: str
    timestamp: str


async def _ingest_candidates_background(
    candidates: List[CandidateRawProfile],
    collection_name: str,
    ingest_mode: str = "omit",
    summarization: bool = True
) -> BatchIngestionResponse:
    """
    Process candidate profiles ingestion.
    Returns results for the response.
    
    Args:
        candidates: List of candidate profiles to ingest
        collection_name: Name of the Qdrant collection
        ingest_mode: "update" (overwrite all) or "omit" (skip existing candidates)
        summarization: True to use Anthropic summarization, False to skip and embed raw text
    """
    if voyage_service is None or qdrant_service is None:
        logger.error("Services not initialized - cannot process ingestion")
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    total_candidates = len(candidates)
    
    if total_candidates == 0:
        logger.warning("No candidates provided for ingestion")
        return BatchIngestionResponse(
            total=0,
            successful=0,
            failed=0,
            results=[]
        )
    
    logger.info(f"=" * 80)
    logger.info(f"BACKGROUND INGESTION STARTED: {total_candidates} candidates into collection '{collection_name}'")
    logger.info(f"Ingest mode: {ingest_mode}, Summarization: {summarization}")
    logger.info(f"=" * 80)
    
    # Ensure collection exists
    try:
        qdrant_service.ensure_collection(collection_name)
        logger.info(f"Collection '{collection_name}' verified/created")
    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}")
        logger.error(f"INGESTION FAILED: Could not create/verify collection")
        raise HTTPException(status_code=500, detail=f"Failed to create/verify collection: {str(e)}")
    
    # In omit mode, filter out existing candidates before processing
    skipped_results = []
    original_total = total_candidates  # Track original count for final response
    if ingest_mode == "omit":
        existing_candidate_ids = qdrant_service.get_existing_candidate_ids(collection_name)
        original_count = len(candidates)
        original_total = original_count  # Update with actual original count
        
        # Create results for skipped candidates
        skipped_results = [
            IngestionResult(
                candidate_id=str(c.candidate_id),
                success=False,
                error="Candidate already exists in collection (omit mode)"
            )
            for c in candidates if str(c.candidate_id) in existing_candidate_ids
        ]
        
        # Filter out existing candidates
        candidates = [c for c in candidates if str(c.candidate_id) not in existing_candidate_ids]
        skipped_count = original_count - len(candidates)
        
        if skipped_count > 0:
            logger.info(f"INGEST_MODE=OMIT: Skipped {skipped_count} existing candidates, processing {len(candidates)} new candidates")
        else:
            logger.info(f"INGEST_MODE=OMIT: All {original_count} candidates are new, processing all")
        
        if len(candidates) == 0:
            logger.info("INGEST_MODE=OMIT: No new candidates to process, all candidates already exist in collection")
            return BatchIngestionResponse(
                total=original_count,
                successful=0,
                failed=original_count,
                results=skipped_results
            )
    
    # Get processing chunk size from settings
    settings = get_settings()
    PROCESSING_CHUNK_SIZE = settings.processing_chunk_size
    
    all_results: List[IngestionResult] = []
    total_successful = 0
    total_failed = 0
    
    # Calculate number of chunks needed
    total_chunks = (total_candidates + PROCESSING_CHUNK_SIZE - 1) // PROCESSING_CHUNK_SIZE
    
    if total_chunks > 1:
        logger.info(f"Auto-chunking {total_candidates} candidates into {total_chunks} chunks of up to {PROCESSING_CHUNK_SIZE} candidates each")
    else:
        logger.info(f"Processing {total_candidates} candidates in a single chunk")
    
    overall_start_time = time.time()
    
    # Collect all failed candidates (for summarization failures)
    all_failed_candidates = []
    
    # Process each chunk sequentially (to avoid overwhelming services)
    for chunk_num in range(1, total_chunks + 1):
        chunk_start = (chunk_num - 1) * PROCESSING_CHUNK_SIZE
        chunk_end = min(chunk_start + PROCESSING_CHUNK_SIZE, total_candidates)
        chunk = candidates[chunk_start:chunk_end]
        
        chunk_results, chunk_successful, chunk_failed = await _process_chunk(
            chunk, chunk_num, total_chunks, collection_name, summarization
        )
        
        all_results.extend(chunk_results)
        total_successful += chunk_successful
        total_failed += chunk_failed
        
        # Collect failed candidates from this chunk
        for result in chunk_results:
            if not result.success:
                all_failed_candidates.append({
                    "candidate_id": result.candidate_id,
                    "error": result.error
                })
    
    total_time = time.time() - overall_start_time
    
    # Log final results
    logger.info(f"=" * 80)
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total candidates: {total_candidates}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success rate: {(total_successful/total_candidates*100):.2f}%")
    logger.info(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per candidate: {total_time/total_candidates:.2f}s")
    
    # Log all failed candidates with details
    if total_failed > 0:
        logger.warning(f"=" * 80)
        logger.warning(f"FAILED CANDIDATES SUMMARY ({total_failed} total):")
        logger.warning(f"=" * 80)
        
        # Group by error type
        error_groups = {}
        for failed in all_failed_candidates:
            error_type = failed.get("error", "Unknown error")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(failed["candidate_id"])
        
        for error_type, candidate_ids in error_groups.items():
            logger.warning(f"  {error_type}: {len(candidate_ids)} candidates")
            logger.warning(f"    Candidate IDs: {', '.join(candidate_ids[:20])}")
            if len(candidate_ids) > 20:
                logger.warning(f"    ... and {len(candidate_ids) - 20} more")
        
        logger.warning(f"=" * 80)
        logger.warning(f"COMPLETE LIST OF FAILED CANDIDATE IDs:")
        failed_ids = [f["candidate_id"] for f in all_failed_candidates]
        logger.warning(f"  {failed_ids}")
        logger.warning(f"=" * 80)
    else:
        logger.info("✓ All candidates processed successfully - no failures")
    
    # Log sample successful candidates
    successful_count = 0
    logger.info(f"Sample successful candidates (first 10):")
    for result in all_results:
        if result.success and successful_count < 10:
            logger.info(f"  - Candidate ID: {result.candidate_id}, Vector ID: {result.vector_id}")
            successful_count += 1
    
    logger.info(f"=" * 80)
    
    # Include skipped results from omit mode in final response
    final_results = skipped_results + all_results
    
    # Use original_total if in omit mode (includes skipped), otherwise use total_candidates
    final_total = original_total if ingest_mode == "omit" else total_candidates
    
    return BatchIngestionResponse(
        total=final_total,
        successful=total_successful,
        failed=total_failed + len(skipped_results),
        results=final_results
    )


@app.post(
    "/ingest",
    response_model=BatchIngestionResponse,
    tags=["Ingestion"],
    summary="Ingest candidate profiles",
    description=""" 
    Ingest candidate profiles with automatic chunking for batches > 128.
    **This endpoint processes ingestion synchronously and returns results.**
    
    **Input Formats Accepted:**
    1. **File path format**:
    ```json
    {
      "file_path": "C:/path/to/final_data.json",
      "collection_name": "candidates_store"
    }
    ```
    
    2. **Wrapped format**:
    ```json
    {
      "candidates": [...],
      "collection_name": "candidates_store"
    }
    ```
    
    3. **Direct array format**:
    ```json
    [...]
    ```
    When using direct array, use query parameter: `?collection_name=candidates_store`
    
    **Features:**
    - Processes ingestion synchronously and returns results
    - Automatically chunks large batches into groups of 128 (Voyage AI limit)
    - Preprocesses each chunk in parallel (concurrent text and payload building)
    - Generates embeddings using Voyage AI batch inference (128 documents per batch)
    - Stores vectors with metadata in Qdrant using batch upsert
    
    **Performance:**
    - Parallel preprocessing of up to 32 candidates simultaneously
    - Single batch embedding call per chunk (Voyage AI handles up to 128)
    - Batch upsert to Qdrant per chunk
    
    **Example:**
    - 1000 candidates → automatically split into 8 chunks
    - Each chunk processed sequentially for optimal resource usage
    """,
    response_description="Ingestion results with successful and failed candidates",
)
async def ingest_candidates(
    request: Request,
    collection_name: Optional[str] = "candidates_store",
    ingest_mode: Optional[str] = "omit",
    summarization: Optional[bool] = True
) -> BatchIngestionResponse:
    """
    Ingest candidate profiles with automatic chunking for batches > 128.
    Processes ingestion synchronously and returns results.
    
    **Parameters:**
    - `ingest_mode`: "omit" (default) or "update"
      - `omit`: Skip candidates that already exist in the collection (check before processing)
      - `update`: Overwrite existing candidates, process all candidates
    - `summarization`: True (default) or False
      - `True`: Use Anthropic Claude to summarize long texts (>800 tokens) before embedding
      - `False`: Skip summarization and directly embed raw text
    
    Accepts multiple input formats:
    1. File path: {"file_path": "path/to/file.json", "collection_name": "..."}
    2. Wrapped format: {"candidates": [...], "collection_name": "...", "mode": "..."}
    3. Direct array: [...] (collection_name and mode can be passed as query params)
    
    Returns BatchIngestionResponse with detailed results.
    """
    if voyage_service is None or qdrant_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Check if ingest_mode and summarization are in request body
    try:
        body = await request.json()
        if isinstance(body, dict):
            if "ingest_mode" in body:
                ingest_mode = body["ingest_mode"]
            if "summarization" in body:
                summarization = body["summarization"]
    except:
        pass  # Use query params if not in body
    
    # Validate ingest_mode
    if ingest_mode not in ["update", "omit"]:
        raise HTTPException(status_code=422, detail=f"Invalid ingest_mode: {ingest_mode}. Must be 'update' or 'omit'")
    
    # Validate summarization
    if not isinstance(summarization, bool):
        raise HTTPException(status_code=422, detail=f"Invalid summarization: {summarization}. Must be True or False")
    
    try:
        candidates, collection_name = await parse_ingestion_request(request, collection_name)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse request: {str(e)}")
    
    total_candidates = len(candidates)
    
    if total_candidates == 0:
        raise HTTPException(status_code=400, detail="No candidates provided")
    
    # Process ingestion synchronously and return results
    return await _ingest_candidates_background(candidates, collection_name, ingest_mode, summarization)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="""
    Check the health status of the service and all dependencies.
    
    **Checks:**
    - Service status
    - Qdrant connection status
    - Voyage AI service status
    
    Returns detailed status information for monitoring and debugging.
    """,
    response_description="Health status of the service and dependencies",
)
async def health_check():
    """
    Health check endpoint with detailed service status.
    
    Checks:
    - Service availability
    - Qdrant connection
    - Voyage AI service initialization
    """
    settings = get_settings()
    health_status = {
        "status": "healthy",
        "qdrant_url": settings.qdrant_url,
        "qdrant_status": "unknown",
        "voyage_status": "unknown",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    # Check Qdrant connection
    if qdrant_service is not None:
        try:
            # Try to get collections to verify connection
            collections = qdrant_service.client.get_collections()
            health_status["qdrant_status"] = "connected"
        except Exception as e:
            health_status["qdrant_status"] = f"error: {str(e)[:100]}"
            health_status["status"] = "degraded"
    else:
        health_status["qdrant_status"] = "not_initialized"
        health_status["status"] = "degraded"
    
    # Check Voyage AI service
    if voyage_service is not None:
        try:
            # Check if client is initialized
            if voyage_service.client:
                health_status["voyage_status"] = "initialized"
            else:
                health_status["voyage_status"] = "not_initialized"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["voyage_status"] = f"error: {str(e)[:100]}"
            health_status["status"] = "degraded"
    else:
        health_status["voyage_status"] = "not_initialized"
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    if health_status["status"] == "healthy":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=health_status
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )


@app.get(
    "/collections/{collection_name}",
    tags=["Collections"],
    summary="Get collection information",
    description="""
    Retrieve comprehensive information about a Qdrant collection.
    
    **Returns:**
    - Collection name and status
    - Statistics (vectors count, points count, indexed vectors)
    - Vector configuration (size, distance metric, HNSW settings)
    - Optimizer status
    - Payload schema (indexed fields for filtering)
    - **List of all candidate_ids** in the collection (sorted)
    - Candidate IDs count
    
    **HNSW Configuration:**
    - M: Number of bi-directional links (default: 32)
    - ef_construct: Construction quality parameter (default: 200)
    - full_scan_threshold: Threshold for full scan vs index
    
    **Payload Schema:**
    Shows all indexed fields that can be used for filtering in search queries.
    
    **Candidate IDs:**
    Returns a complete list of all unique candidate_ids stored in the collection.
    This may take a moment for large collections as it scrolls through all points.
    """,
    response_description="Detailed collection information including configuration and statistics",
)
async def get_collection_info(collection_name: str):
    """
    Get comprehensive information about a Qdrant collection.
    
    Returns detailed information including:
    - Basic statistics (vectors, points, status)
    - Vector configuration (size, distance metric)
    - HNSW index settings (M, ef_construct)
    - Optimizer status
    - Payload schema (indexed fields for filtering)
    - Complete list of all candidate_ids in the collection (sorted)
    - Candidate IDs count
    """
    if qdrant_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        info = qdrant_service.get_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")


@app.get(
    "/collections/{collection_name}/problematic",
    tags=["Collections"],
    summary="Find problematic candidates",
    description="""
    Scan the collection to find candidates with issues.
    
    **Checks for:**
    - Empty or null vectors (vectors that are empty or all zeros)
    - Missing candidate_id in payload
    - Missing client_id in payload
    - Malformed payload data
    
    **Returns:**
    - List of candidate_ids with empty vectors
    - List of candidate_ids missing client_id
    - Count of points missing candidate_id
    - List of malformed payloads with details
    - Total count of unique problematic candidates
    - Total points scanned
    
    **Use Cases:**
    - Identify candidates that failed during ingestion
    - Find candidates with corrupted data
    - Clean up invalid entries
    - Audit data quality
    
    **Note:** This operation scans through all points in the collection,
    which may take time for large collections.
    """,
    response_description="List of problematic candidates grouped by issue type",
)
async def get_problematic_candidates(collection_name: str):
    """
    Find candidates with issues in the collection.
    
    Scans the collection to identify:
    - Candidates with empty/null vectors
    - Candidates missing required payload fields
    - Candidates with malformed data
    
    Returns detailed information about problematic candidates grouped by issue type.
    """
    if qdrant_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        problematic = qdrant_service.find_problematic_candidates(collection_name)
        return {
            "collection_name": collection_name,
            "scan_summary": {
                "total_scanned": problematic.get("total_scanned", 0),
                "total_problematic": problematic.get("total_problematic", 0),
            },
            "issues": {
                "empty_vectors": {
                    "count": len(problematic.get("empty_vectors", [])),
                    "candidate_ids": problematic.get("empty_vectors", []),
                    "description": "Candidates with empty, null, or all-zero vectors"
                },
                "missing_client_id": {
                    "count": len(problematic.get("missing_client_id", [])),
                    "candidate_ids": problematic.get("missing_client_id", []),
                    "description": "Candidates missing client_id in payload"
                },
                "missing_candidate_id": {
                    "count": problematic.get("missing_candidate_id", 0),
                    "description": "Points missing candidate_id in payload (cannot identify candidate)"
                },
                "malformed_payload": {
                    "count": len(problematic.get("malformed_payload", [])),
                    "details": problematic.get("malformed_payload", []),
                    "description": "Candidates with multiple or severe payload issues"
                }
            },
            "error": problematic.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan collection: {str(e)}")


@app.post(
    "/search/{collection_name}",
    tags=["Search"],
    summary="Search candidates",
    description="""
    Search for similar candidates using semantic search with Voyage AI embeddings.
    
    **Features:**
    - Semantic similarity search using query embeddings
    - Optional filters by experience, location, skills, etc.
    - Returns top-k most similar candidates with similarity scores
    
    **Example Query:**
    "Senior software engineer with Python and machine learning experience"
    """,
    response_description="Search results with candidate matches and similarity scores",
)
async def search_candidates(collection_name: str, request: SearchRequest):
    """
    Search for similar candidates using semantic search.
    
    Args:
        collection_name: Name of the collection to search
        request: Search request with query and optional filters
    
    Returns:
        List of matching candidates with similarity scores
    """
    if voyage_service is None or qdrant_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Generate query embedding
    try:
        query_vector = voyage_service.generate_query_embedding(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(e)}")
    
    # Build filters
    filters = {}
    if request.min_experience_years:
        filters["years_experience_total"] = {"gte": request.min_experience_years}
    # Note: location, skills, primary_profession filters can be added here
    
    # Search
    try:
        results = qdrant_service.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=request.limit,
            filters=filters if filters else None,
        )
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post(
    "/search/job-description/{collection_name}",
    tags=["Search"],
    summary="Search candidates by job description",
    description="""
    Search for top 200 matching candidates based on a job description text.
    
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
    response_description="Search results with top 200 candidate matches and similarity scores",
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


@app.delete(
    "/candidates/{collection_name}/{candidate_id}",
    tags=["Ingestion"],
    summary="Delete candidate",
    description="Delete a candidate's vector from the specified collection.",
    response_description="Deletion status confirmation",
)
async def delete_candidate(collection_name: str, candidate_id: str):
    """
    Delete a candidate's vector from the collection.
    
    Args:
        collection_name: Name of the collection
        candidate_id: ID of the candidate to delete
    
    Returns:
        Deletion confirmation
    """
    if qdrant_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        qdrant_service.delete_by_candidate_id(collection_name, candidate_id)
        return {"status": "deleted", "candidate_id": candidate_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
