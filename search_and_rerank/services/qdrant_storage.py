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

