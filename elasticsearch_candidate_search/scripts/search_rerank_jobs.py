"""
Search for top candidates matching jobs from rerank_input JSON file.
The input file should be a JSON array where each job has job_id and job_description_clean.
Usage: python search_rerank_jobs.py <rerank_input_json_file_path> [output_file_path] [index_name] [top_n] [limit]
"""

import json
import sys
import re
from elasticsearch import Elasticsearch
from typing import List, Dict, Any


def load_rerank_jobs(file_path: str) -> List[Dict]:
    """Load job data from rerank_input JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and single dict
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError("JSON file must contain a list or dict of jobs")


def extract_key_terms(text: str, max_terms: int = 10) -> List[str]:
    """Extract key terms from job description."""
    if not text:
        return []
    
    # Remove HTML tags and special formatting
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[[^\]]+\]', ' ', text)  # Remove [SECTION] markers
    # Remove special characters but keep words
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split into words
    words = text.split()
    
    # Filter out common stop words and short words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
                  'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
                  'those', 'years', 'year', 'experience', 'required', 'skills', 'skill', 'role', 'responsibilities'}
    
    # Extract meaningful terms (3+ characters, not stop words)
    key_terms = [w.upper() for w in words if len(w) >= 3 and w.upper() not in stop_words]
    
    # Return top N unique terms
    seen = set()
    unique_terms = []
    for term in key_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
            if len(unique_terms) >= max_terms:
                break
    
    return unique_terms


def build_job_query(job_description: str, top_n: int = 100, min_score: float = 0.0) -> Dict[str, Any]:
    """Build Elasticsearch query for a job description."""
    if not job_description:
        # Return a match_all query if no description
        return {
            "size": top_n,
            "min_score": min_score,
            "query": {
                "match_all": {}
            }
        }
    
    should_clauses = []
    
    # 1. Full text search on job description
    # Extract key terms from description
    key_terms = extract_key_terms(job_description, max_terms=15)
    
    if key_terms:
        # Multi-match query with all key terms
        terms_query = ' '.join(key_terms)
        should_clauses.append({
            "multi_match": {
                "query": terms_query,
                "fields": ["candidate_text_clean^3.0", "full_text^2.0"],
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "30%",
                "boost": 2.0
            }
        })
        
        # Individual match queries for top key terms
        for term in key_terms[:8]:
            should_clauses.append({
                "match": {
                    "candidate_text_clean": {
                        "query": term,
                        "boost": 1.5
                    }
                }
            })
        
        # Phrase matching for important terms
        for term in key_terms[:5]:
            if len(term.split()) == 1:  # Single word terms
                should_clauses.append({
                    "match_phrase": {
                        "candidate_text_clean": {
                            "query": term,
                            "boost": 1.2
                        }
                    }
                })
    
    # 2. Full text search on entire description (with lower boost)
    should_clauses.append({
        "match": {
            "candidate_text_clean": {
                "query": job_description,
                "operator": "or",
                "minimum_should_match": "20%",
                "boost": 0.8
            }
        }
    })
    
    # Build final query
    query = {
        "size": top_n,
        "min_score": min_score,
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
    }
    
    return query


def search_candidates_for_job(es: Elasticsearch, job_id: str, job_description: str, 
                              index_name: str = "candidates_test", top_n: int = 100) -> List[Dict]:
    """Search for top candidates matching a job."""
    query = build_job_query(job_description, top_n)
    
    try:
        response = es.search(index=index_name, body=query)
        candidates = []
        
        for hit in response['hits']['hits']:
            candidate = hit['_source']
            candidate['_score'] = hit['_score']
            candidates.append(candidate)
        
        return candidates
    except Exception as e:
        print(f"Error searching for job {job_id}: {e}")
        return []


def search_all_jobs(rerank_file_path: str, output_file_path: str = None, 
                    es_host: str = "localhost", es_port: int = 9200, 
                    index_name: str = "candidates_test", top_n: int = 100, limit: int = None):
    """Search for top candidates for each job in rerank_input file."""
    print(f"Loading jobs from {rerank_file_path}...")
    jobs = load_rerank_jobs(rerank_file_path)
    print(f"Loaded {len(jobs)} jobs")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        jobs = jobs[:limit]
        print(f"Limiting to first {len(jobs)} jobs for searching")
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        [f"http://{es_host}:{es_port}"],
        request_timeout=60
    )
    
    # Check connection
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch. Make sure it's running.")
    
    # Check if index exists
    if not es.indices.exists(index=index_name):
        raise ValueError(f"Index '{index_name}' does not exist. Please index candidates first.")
    
    print("Connected to Elasticsearch")
    print(f"Searching for top {top_n} candidates for each job...\n")
    
    results = []
    
    for i, job in enumerate(jobs, 1):
        job_id = job.get('job_id', f'job_{i}')
        job_description = job.get('job_description_clean', '')
        
        print(f"[{i}/{len(jobs)}] Searching for job: {job_id}")
        
        candidates = search_candidates_for_job(es, job_id, job_description, index_name, top_n)
        
        result = {
            "job_id": job_id,
            "job_description_clean": job_description,
            "candidates_found": len(candidates),
            "top_candidates": candidates
        }
        
        results.append(result)
        print(f"  Found {len(candidates)} candidates\n")
    
    # Save results
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OK] Results saved to {output_file_path}")
        
        # Create summary file with job_id to candidate_ids mapping
        summary_file_path = output_file_path.replace('.json', '_summary.json')
        summary = []
        for result in results:
            candidate_ids = [candidate.get('candidate_id', '') for candidate in result.get('top_candidates', [])]
            summary.append({
                "job_id": result['job_id'],
                "candidates_count": len(candidate_ids),
                "candidate_ids": candidate_ids
            })
        
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] Summary saved to {summary_file_path}")
    else:
        # Print summary
        print("\n" + "="*60)
        print("SEARCH RESULTS SUMMARY")
        print("="*60)
        for result in results:
            print(f"\nJob ID: {result['job_id']}")
            print(f"Candidates found: {result['candidates_found']}")
            if result['top_candidates']:
                print("Top 5 candidates:")
                for idx, candidate in enumerate(result['top_candidates'][:5], 1):
                    print(f"  {idx}. Candidate ID: {candidate.get('candidate_id', 'N/A')} "
                          f"(Score: {candidate.get('_score', 0):.2f})")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_rerank_jobs.py <rerank_input_json_file_path> [output_file_path] [index_name] [top_n] [limit]")
        sys.exit(1)
    
    rerank_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    index_name = sys.argv[3] if len(sys.argv) > 3 else "candidates_test"
    top_n = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 100
    limit = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else None
    
    try:
        search_all_jobs(rerank_file_path, output_file_path, index_name=index_name, top_n=top_n, limit=limit)
        print("\n[SUCCESS] Search completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

