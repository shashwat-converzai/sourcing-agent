"""
Search for top candidates matching each job.
Usage: python search_candidates.py <jobs_json_file_path> [output_file_path] [index_name] [limit]
"""

import json
import sys
import re
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Set


def load_jobs(file_path: str) -> List[Dict]:
    """Load job data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle nested structure with 'data' field
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    
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
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters but keep words
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split into words
    words = text.split()
    
    # Filter out common stop words and short words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'years', 'year', 'experience', 'required', 'skills', 'skill'}
    
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


def build_job_query(job: Dict, top_n: int = 200, min_score: float = 0.5) -> Dict[str, Any]:
    """Build improved Elasticsearch query for a job."""
    # Extract searchable fields from job
    job_title = job.get('JOBTITLE', '') or job.get('POSTING_TITLE', '')
    job_description = job.get('JOBDESCRIPTION', '') or job.get('POSTINGDESCRIPTION', '')
    
    # Build search query
    should_clauses = []
    must_clauses = []
    
    # 1. Job title matching with multiple strategies
    if job_title and job_title != 'Null':
        job_title_clean = job_title.strip()
        
        # Exact phrase match (highest boost)
        should_clauses.append({
            "match_phrase": {
                "all_titles": {
                    "query": job_title_clean,
                    "boost": 5.0
                }
            }
        })
        
        # Multi-match query for title across multiple fields
        should_clauses.append({
            "multi_match": {
                "query": job_title_clean,
                "fields": ["all_titles^3.0", "resume_data^1.0"],
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "75%",
                "boost": 3.0
            }
        })
        
        # Fuzzy match for variations
        should_clauses.append({
            "match": {
                "all_titles": {
                    "query": job_title_clean,
                    "fuzziness": "AUTO",
                    "boost": 1.5
                }
            }
        })
    
    # 2. Job description - extract key terms instead of full text
    if job_description:
        # Extract key terms from description
        key_terms = extract_key_terms(job_description, max_terms=15)
        
        if key_terms:
            # Search for each key term
            for term in key_terms:
                should_clauses.append({
                    "match": {
                        "resume_data": {
                            "query": term,
                            "boost": 0.8
                        }
                    }
                })
            
            # Multi-match for key terms
            terms_query = ' '.join(key_terms)
            should_clauses.append({
                "multi_match": {
                    "query": terms_query,
                    "fields": ["resume_data^1.5", "all_titles^1.0", "all_skills_text^1.0"],
                    "type": "best_fields",
                    "operator": "or",
                    "minimum_should_match": "30%",
                    "boost": 1.0
                }
            })
    
    # Build final query with minimum score
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


def search_candidates_for_job(es: Elasticsearch, job: Dict, index_name: str = "candidates", top_n: int = 200) -> List[Dict]:
    """Search for top candidates matching a job."""
    query = build_job_query(job, top_n)
    
    try:
        response = es.search(index=index_name, body=query)
        candidates = []
        
        for hit in response['hits']['hits']:
            candidate = hit['_source']
            candidate['_score'] = hit['_score']
            candidates.append(candidate)
        
        return candidates
    except Exception as e:
        print(f"Error searching for job {job.get('ID', 'unknown')}: {e}")
        return []


def search_all_jobs(jobs_file_path: str, output_file_path: str = None, 
                    es_host: str = "localhost", es_port: int = 9200, 
                    index_name: str = "candidates", top_n: int = 200, limit: int = None):
    """Search for top candidates for each job."""
    print(f"Loading jobs from {jobs_file_path}...")
    jobs = load_jobs(jobs_file_path)
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
        job_id = job.get('ID', f'job_{i}')
        job_title = job.get('JOBTITLE', '') or job.get('POSTING_TITLE', 'Unknown')
        
        print(f"[{i}/{len(jobs)}] Searching for job: {job_id} - {job_title}")
        
        candidates = search_candidates_for_job(es, job, index_name, top_n)
        
        result = {
            "job_id": job_id,
            "job_title": job_title,
            "company_name": job.get('COMPANYNAME', ''),
            "job_status": job.get('JOBSTATUS', ''),
            "skills": job.get('SKILLS', ''),
            "location": {
                "city": job.get('CITY', ''),
                "state": job.get('STATE', ''),
                "country": job.get('COUNTRY', '')
            },
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
                "job_title": result['job_title'],
                "company_name": result['company_name'],
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
            print(f"Title: {result['job_title']}")
            print(f"Company: {result['company_name']}")
            print(f"Candidates found: {result['candidates_found']}")
            if result['top_candidates']:
                print("Top 5 candidates:")
                for idx, candidate in enumerate(result['top_candidates'][:5], 1):
                    print(f"  {idx}. {candidate.get('first_name', '')} {candidate.get('last_name', '')} "
                          f"(Score: {candidate.get('_score', 0):.2f}, "
                          f"ID: {candidate.get('candidate_id', 'N/A')})")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_candidates.py <jobs_json_file_path> [output_file_path] [index_name] [limit]")
        sys.exit(1)
    
    jobs_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    index_name = sys.argv[3] if len(sys.argv) > 3 else "candidates"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else None
    
    try:
        search_all_jobs(jobs_file_path, output_file_path, index_name=index_name, limit=limit)
        print("\n[SUCCESS] Search completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

