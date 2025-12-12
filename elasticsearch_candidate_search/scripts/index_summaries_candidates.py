"""
Index candidate summaries from summaries JSON file into Elasticsearch.
The input file should be a JSON object where keys are candidate_ids and values are candidate text summaries.
Usage: python index_summaries_candidates.py <summaries_json_file_path> [index_name] [limit]
"""

import json
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import Dict


def load_summaries(file_path: str) -> Dict[str, str]:
    """Load candidate summaries from JSON file.
    Expected format: {"candidate_id": "candidate_text_clean", ...}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError("JSON file must contain a dictionary with candidate_id as keys and candidate text as values")
    
    return data


def prepare_candidate_doc(candidate_id: str, candidate_text: str) -> Dict:
    """Prepare candidate document for indexing from summary text."""
    doc = {
        'candidate_id': candidate_id,
        'candidate_text_clean': candidate_text,
        # Searchable field - same as candidate_text_clean for now
        'full_text': candidate_text
    }
    return doc


def index_candidates(file_path: str, es_host: str = "localhost", es_port: int = 9200, 
                     index_name: str = "candidates_test", limit: int = None):
    """Index all candidates from summaries JSON file into Elasticsearch."""
    print(f"Loading candidate summaries from {file_path}...")
    summaries = load_summaries(file_path)
    print(f"Loaded {len(summaries)} candidate summaries")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        # Convert to list of items, limit, then back to dict
        items = list(summaries.items())[:limit]
        summaries = dict(items)
        print(f"Limiting to first {len(summaries)} candidates for indexing")
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        [f"http://{es_host}:{es_port}"],
        request_timeout=60
    )
    
    # Check connection
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch. Make sure it's running.")
    
    print("Connected to Elasticsearch")
    
    # Create index with mapping
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Deleting...")
        es.indices.delete(index=index_name)
    
    mapping = {
        "mappings": {
            "properties": {
                "candidate_id": {"type": "keyword"},
                "candidate_text_clean": {"type": "text"},
                "full_text": {"type": "text"}
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)
    print(f"Created index '{index_name}'")
    
    # Prepare documents for bulk indexing
    actions = []
    for candidate_id, candidate_text in summaries.items():
        doc = prepare_candidate_doc(candidate_id, candidate_text)
        action = {
            "_index": index_name,
            "_id": candidate_id,
            "_source": doc
        }
        actions.append(action)
    
    # Bulk index
    print(f"Indexing {len(actions)} candidates...")
    success, failed = bulk(es, actions, chunk_size=1000, request_timeout=60)
    
    print(f"[OK] Successfully indexed {success} candidates")
    if failed:
        print(f"[ERROR] Failed to index {len(failed)} candidates")
        for item in failed[:5]:  # Show first 5 errors
            print(f"  Error: {item}")
    
    # Refresh index
    es.indices.refresh(index=index_name)
    print("Index refreshed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_summaries_candidates.py <summaries_json_file_path> [index_name] [limit]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    index_name = sys.argv[2] if len(sys.argv) > 2 else "candidates_test"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    
    try:
        index_candidates(file_path, index_name=index_name, limit=limit)
        print("\n[SUCCESS] Indexing completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

