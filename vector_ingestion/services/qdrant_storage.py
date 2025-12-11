from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, HnswConfigDiff
from typing import List, Dict, Any, Optional
import logging
import uuid
import time

from ..config import Settings

logger = logging.getLogger(__name__)


class QdrantStorageService:
    """Service for storing and retrieving vectors from Qdrant."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url)
        self.vector_dimension = settings.vector_dimension
        logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
    
    def ensure_collection(self, collection_name: str) -> None:
        """
        Ensure a collection exists with the correct configuration.
        Creates it if it doesn't exist.
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating collection: {collection_name}")
            # Configure HNSW with recommended settings for speed/quality balance
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=32,  # Number of bi-directional links for each node (balances speed/quality)
                        ef_construct=200,  # Size of the candidate list during construction (128-200 recommended)
                    ),
                ),
            )
            
            # Create payload indexes for filtering (required metadata fields)
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="candidate_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="client_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="primary_role_code",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="skills",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="country",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="city",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="years_experience_total",
                field_schema=models.PayloadSchemaType.FLOAT,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="current_role",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            
            logger.info(f"Collection {collection_name} created with indexes")
        else:
            logger.debug(f"Collection {collection_name} already exists")
    
    def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List] = None,
    ) -> List[str]:
        """
        Upsert vectors with their payloads into Qdrant.
        
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            payloads: List of metadata payloads (one per vector)
            ids: Optional list of IDs (UUIDs or unsigned integers). If not provided, UUIDs will be generated.
            
        Returns:
            List of point IDs that were upserted (as strings)
        """
        if not vectors:
            return []
        
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors must match number of payloads")
        
        # Generate UUIDs if not provided
        # Qdrant requires point IDs to be either UUID objects or unsigned integers
        if ids is None:
            ids = [uuid.uuid4() for _ in vectors]
        
        # Convert to UUID objects if they're strings (for validation)
        point_ids = []
        for point_id in ids:
            if isinstance(point_id, str):
                try:
                    # Try to convert string to UUID
                    point_ids.append(uuid.UUID(point_id))
                except ValueError:
                    # If it's not a valid UUID, generate a new one
                    logger.warning(f"Invalid point ID format: {point_id}, generating new UUID")
                    point_ids.append(uuid.uuid4())
            elif isinstance(point_id, uuid.UUID):
                point_ids.append(point_id)
            else:
                # For integers, keep as is (Qdrant accepts unsigned integers)
                point_ids.append(point_id)
        
        points = [
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(point_ids, vectors, payloads)
        ]
        
        max_retries = 3  # Total of 4 attempts: initial + 3 retries
        wait_times = [5, 10, 30]  # Wait times in seconds for each retry
        
        for attempt in range(max_retries + 1):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,  # Wait for indexing to complete
                )
                
                logger.info(f"Upserted {len(points)} vectors to collection {collection_name}")
                # Return IDs as strings for consistency
                return [str(pid) for pid in point_ids]
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str or 
                    "429" in error_str or
                    "too many requests" in error_str or
                    "quota" in error_str or
                    "connection" in error_str or
                    "timeout" in error_str
                )
                
                if attempt < max_retries and (is_rate_limit or True):  # Retry on any error
                    wait_time = wait_times[attempt]  # Get wait time for this attempt
                    logger.warning(f"Error upserting vectors to collection {collection_name} (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error upserting vectors to collection {collection_name} after {attempt + 1} attempts: {e}")
                    raise
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        ef: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional filters to apply
            ef: Size of the candidate list during search (200-500 recommended for quality)
            
        Returns:
            List of search results with scores and payloads
        """
        query_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                elif isinstance(value, dict) and "gte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(gte=value["gte"]),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            query_filter = models.Filter(must=must_conditions)
        
        # Build search params with HNSW ef parameter if provided
        search_params = None
        if ef is not None:
            # ef parameter for HNSW search quality (200-500 recommended)
            search_params = models.SearchParams(hnsw_ef=ef)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            search_params=search_params,
        )
        
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]
    
    def delete_by_candidate_id(
        self,
        collection_name: str,
        candidate_id: str,
    ) -> None:
        """Delete vectors by candidate_id."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="candidate_id",
                            match=models.MatchValue(value=candidate_id),
                        ),
                    ],
                ),
            ),
        )
        logger.info(f"Deleted vectors for candidate_id: {candidate_id}")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a collection.
        
        Returns comprehensive collection information including:
        - Basic stats (vectors, points, status)
        - Vector configuration (size, distance metric)
        - HNSW index settings
        - Optimizer status
        - Payload schema (indexed fields)
        """
        info = self.client.get_collection(collection_name)
        
        # Helper function to safely get attribute
        def safe_get(obj, attr, default=None):
            try:
                return getattr(obj, attr, default)
            except (AttributeError, TypeError):
                return default
        
        # Extract vector configuration
        vector_config = {}
        try:
            config = safe_get(info, 'config')
            if config:
                params = safe_get(config, 'params')
                if params:
                    vectors = safe_get(params, 'vectors')
                    if vectors:
                        vector_config = {
                            "vector_size": safe_get(vectors, 'size'),
                            "distance_metric": safe_get(safe_get(vectors, 'distance'), 'value') if safe_get(vectors, 'distance') else None,
                        }
                        
                        # Extract HNSW configuration if available
                        hnsw = safe_get(vectors, 'hnsw_config')
                        if not hnsw:
                            hnsw = safe_get(params, 'hnsw_config')
                        
                        if hnsw:
                            vector_config["hnsw"] = {
                                "m": safe_get(hnsw, 'm'),
                                "ef_construct": safe_get(hnsw, 'ef_construct'),
                                "full_scan_threshold": safe_get(hnsw, 'full_scan_threshold'),
                            }
        except Exception as e:
            logger.warning(f"Error extracting vector config: {e}")
        
        # Extract optimizer status
        optimizer_status = {}
        try:
            optimizer = safe_get(info, 'optimizer_status')
            if optimizer:
                optimizer_status = {
                    "ok": safe_get(optimizer, 'ok'),
                }
        except Exception as e:
            logger.warning(f"Error extracting optimizer status: {e}")
        
        # Extract payload schema (indexed fields)
        payload_schema = {}
        try:
            schema = safe_get(info, 'payload_schema')
            if schema:
                payload_schema = {}
                for field_name, field_schema in schema.items():
                    try:
                        data_type = safe_get(field_schema, 'data_type')
                        payload_schema[field_name] = {
                            "data_type": safe_get(data_type, 'value') if data_type else None,
                            "points": safe_get(field_schema, 'points'),
                        }
                    except Exception as e:
                        logger.warning(f"Error extracting schema for field {field_name}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting payload schema: {e}")
        
        # Extract all candidate_ids from the collection
        candidate_ids = []
        try:
            logger.info(f"Fetching all candidate_ids from collection '{collection_name}'...")
            # Use scroll to get all points with their payloads
            # Scroll through all points in batches
            scroll_limit = 1000  # Fetch 1000 points at a time
            offset = None
            candidate_ids_set = set()
            
            while True:
                # Scroll to get next batch
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # We don't need vectors, just payloads
                )
                
                points, next_offset = scroll_result
                
                # Extract candidate_ids from payloads
                for point in points:
                    if point.payload and 'candidate_id' in point.payload:
                        candidate_id = point.payload['candidate_id']
                        if candidate_id:
                            candidate_ids_set.add(str(candidate_id))
                
                # Check if there are more points
                if next_offset is None:
                    break
                offset = next_offset
            
            candidate_ids = sorted(list(candidate_ids_set))
            logger.info(f"Found {len(candidate_ids)} unique candidate_ids in collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error fetching candidate_ids from collection: {e}")
            candidate_ids = []
        
        # Build result with safe attribute access
        result = {
            "name": collection_name,
            "status": safe_get(safe_get(info, 'status'), 'value') if safe_get(info, 'status') else None,
            "statistics": {
                "vectors_count": safe_get(info, 'vectors_count', 0),
                "points_count": safe_get(info, 'points_count', 0),
                "indexed_vectors_count": safe_get(info, 'indexed_vectors_count', 0),
            },
            "vector_config": vector_config if vector_config else None,
            "optimizer_status": optimizer_status if optimizer_status else None,
            "payload_schema": payload_schema if payload_schema else None,
            "candidate_ids": candidate_ids,
            "candidate_ids_count": len(candidate_ids),
        }
        
        return result
    
    def get_existing_candidate_ids(self, collection_name: str) -> set:
        """
        Get set of all existing candidate_ids from the collection.
        Used for omit mode to skip already-ingested candidates.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Set of candidate_id strings
        """
        candidate_ids_set = set()
        try:
            logger.info(f"Fetching existing candidate_ids from collection '{collection_name}'...")
            scroll_limit = 1000
            offset = None
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_offset = scroll_result
                
                for point in points:
                    if point.payload and 'candidate_id' in point.payload:
                        candidate_id = point.payload['candidate_id']
                        if candidate_id:
                            candidate_ids_set.add(str(candidate_id))
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"Found {len(candidate_ids_set)} existing candidate_ids in collection '{collection_name}'")
        except Exception as e:
            logger.warning(f"Error fetching existing candidate_ids (collection may not exist yet): {e}")
            # Return empty set if collection doesn't exist or error occurs
            candidate_ids_set = set()
        
        return candidate_ids_set
    
    def find_problematic_candidates(
        self,
        collection_name: str,
    ) -> Dict[str, Any]:
        """
        Find candidates with issues in the collection.
        
        Checks for:
        - Empty or null vectors
        - Missing required payload fields (candidate_id, client_id)
        - Malformed payload data
        
        Returns:
            Dictionary with lists of problematic candidate_ids grouped by issue type
        """
        problematic = {
            "empty_vectors": [],
            "missing_candidate_id": 0,  # Count of points missing candidate_id
            "missing_client_id": [],
            "malformed_payload": [],
            "total_problematic": 0,
        }
        
        try:
            logger.info(f"Scanning collection '{collection_name}' for problematic candidates...")
            scroll_limit = 1000
            offset = None
            total_scanned = 0
            
            while True:
                # Scroll to get next batch
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,  # We need vectors to check if they're empty
                )
                
                points, next_offset = scroll_result
                
                # Check each point for issues
                for point in points:
                    total_scanned += 1
                    issues = []
                    candidate_id = None
                    point_info = {"point_id": str(point.id)}
                    
                    # Check payload
                    if not point.payload:
                        issues.append("missing_payload")
                        problematic["malformed_payload"].append({
                            "point_id": str(point.id),
                            "issues": ["missing_payload"]
                        })
                        continue
                    
                    # Check for candidate_id
                    if 'candidate_id' not in point.payload or not point.payload['candidate_id']:
                        issues.append("missing_candidate_id")
                        problematic["missing_candidate_id"] += 1
                    else:
                        candidate_id = str(point.payload['candidate_id'])
                        point_info["candidate_id"] = candidate_id
                    
                    # Check for client_id
                    if 'client_id' not in point.payload or not point.payload['client_id']:
                        issues.append("missing_client_id")
                        if candidate_id:
                            problematic["missing_client_id"].append(candidate_id)
                    
                    # Check for empty/null vector
                    vector_issue = None
                    if not point.vector or len(point.vector) == 0:
                        vector_issue = "empty_vector"
                        issues.append(vector_issue)
                    elif point.vector and all(v == 0.0 for v in point.vector):
                        # Vector exists but is all zeros (likely invalid)
                        vector_issue = "zero_vector"
                        issues.append(vector_issue)
                    
                    if vector_issue and candidate_id:
                        problematic["empty_vectors"].append(candidate_id)
                    
                    # Check for malformed payload (multiple issues or missing critical fields)
                    if len(issues) > 1 or (issues and candidate_id):
                        problematic["malformed_payload"].append({
                            **point_info,
                            "issues": issues
                        })
                
                # Check if there are more points
                if next_offset is None:
                    break
                offset = next_offset
            
            # Remove duplicates and get unique candidate_ids
            problematic["empty_vectors"] = sorted(list(set(problematic["empty_vectors"])))
            # missing_candidate_id is already a count
            problematic["missing_client_id"] = sorted(list(set(problematic["missing_client_id"])))
            
            # Count total unique problematic candidates
            all_problematic_ids = set()
            all_problematic_ids.update(problematic["empty_vectors"])
            all_problematic_ids.update(problematic["missing_client_id"])
            for item in problematic["malformed_payload"]:
                if "candidate_id" in item and item["candidate_id"]:
                    all_problematic_ids.add(item["candidate_id"])
            
            problematic["total_problematic"] = len(all_problematic_ids)
            problematic["total_scanned"] = total_scanned
            
            logger.info(f"Scan complete: {total_scanned} points scanned, {problematic['total_problematic']} unique problematic candidates found")
            
        except Exception as e:
            logger.error(f"Error scanning collection for problematic candidates: {e}")
            problematic["error"] = str(e)
        
        return problematic
