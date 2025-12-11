"""
Index candidate data into Elasticsearch.
Usage: python index_candidates.py <candidates_json_file_path> [index_name] [limit]
"""

import json
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import List, Dict


def load_candidates(file_path: str) -> List[Dict]:
    """Load candidate data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and single dict
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError("JSON file must contain a list or dict of candidates")


def prepare_candidate_doc(candidate: Dict) -> Dict:
    """Prepare candidate document for indexing."""
    # Extract skills as a list of strings
    skills = candidate.get('skills', [])
    if isinstance(skills, str):
        skills = [skills]
    
    # Extract title experience for searchable text
    title_exp_text = []
    for exp in candidate.get('title_exp', []):
        title = exp.get('title', '')
        if title:
            title_exp_text.append(title)
    
    # Prepare document
    doc = {
        'candidate_id': candidate.get('candidate_id', ''),
        'email': candidate.get('email', ''),
        'first_name': candidate.get('first_name', ''),
        'last_name': candidate.get('last_name', ''),
        'current_role': candidate.get('current_role', ''),
        'skills': skills,
        'country': candidate.get('country', ''),
        'city': candidate.get('city', ''),
        'years_experience_total': candidate.get('years_experience_total', 0),
        'title_exp': candidate.get('title_exp', []),
        'education': candidate.get('education', []),
        'resume_data': candidate.get('resume_data', ''),
        # Searchable fields
        'all_titles': ' '.join(title_exp_text),
        'all_skills_text': ' '.join(skills) if skills else '',
        'full_text': ' '.join([
            candidate.get('resume_data', ''),
            ' '.join(title_exp_text),
            ' '.join(skills) if skills else ''
        ])
    }
    return doc


def index_candidates(file_path: str, es_host: str = "localhost", es_port: int = 9200, index_name: str = "candidates", limit: int = None):
    """Index all candidates from JSON file into Elasticsearch."""
    print(f"Loading candidates from {file_path}...")
    candidates = load_candidates(file_path)
    print(f"Loaded {len(candidates)} candidates")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        candidates = candidates[:limit]
        print(f"Limiting to first {len(candidates)} candidates for indexing")
    
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
                "email": {"type": "keyword"},
                "first_name": {"type": "text"},
                "last_name": {"type": "text"},
                "current_role": {"type": "text"},
                "skills": {"type": "text"},
                "country": {"type": "keyword"},
                "city": {"type": "text"},
                "years_experience_total": {"type": "float"},
                "title_exp": {"type": "object", "enabled": False},
                "education": {"type": "object", "enabled": False},
                "resume_data": {"type": "text"},
                "all_titles": {"type": "text"},
                "all_skills_text": {"type": "text"},
                "full_text": {"type": "text"}
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)
    print(f"Created index '{index_name}'")
    
    # Prepare documents for bulk indexing
    actions = []
    for candidate in candidates:
        doc = prepare_candidate_doc(candidate)
        action = {
            "_index": index_name,
            "_id": doc['candidate_id'],
            "_source": doc
        }
        actions.append(action)
    
    # Bulk index
    print(f"Indexing {len(actions)} candidates...")
    success, failed = bulk(es, actions, chunk_size=1000, request_timeout=60)
    
    print(f"[OK] Successfully indexed {success} candidates")
    if failed:
        print(f"[ERROR] Failed to index {len(failed)} candidates")
    
    # Refresh index
    es.indices.refresh(index=index_name)
    print("Index refreshed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_candidates.py <candidates_json_file_path> [index_name] [limit]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    index_name = sys.argv[2] if len(sys.argv) > 2 else "candidates"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    
    try:
        index_candidates(file_path, index_name=index_name, limit=limit)
        print("\n[SUCCESS] Indexing completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

