"""
Validation script to compare Elasticsearch search results with source of truth.
Usage: python validate_results.py [results_summary_file] [source_of_truth_file]
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Set
from elasticsearch import Elasticsearch


def load_source_of_truth(file_path: str) -> Dict[str, Set[str]]:
    """
    Load source of truth and group by job_id.
    Returns: {job_id: set of candidate_ids}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by job_id
    job_candidates = defaultdict(set)
    for item in data:
        job_id = str(item.get('ats_job_id', ''))
        candidate_id = str(item.get('ats_candidate_id', ''))
        if job_id and candidate_id:
            job_candidates[job_id].add(candidate_id)
    
    return dict(job_candidates)


def load_es_results(file_path: str) -> Dict[str, Set[str]]:
    """
    Load ES search results.
    Returns: {job_id: set of candidate_ids}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    es_results = {}
    for item in data:
        job_id = str(item.get('job_id', ''))
        candidate_ids = [str(cid) for cid in item.get('candidate_ids', [])]
        if job_id:
            es_results[job_id] = set(candidate_ids)
    
    return es_results


def check_candidates_in_index(candidate_ids: Set[str], index_name: str = "candidates", 
                              es_host: str = "localhost", es_port: int = 9200) -> Dict[str, bool]:
    """Check if candidate IDs exist in Elasticsearch index."""
    es = Elasticsearch([f"http://{es_host}:{es_port}"], request_timeout=60)
    
    if not es.ping():
        print("[WARNING] Cannot connect to Elasticsearch. Skipping index check.")
        return {}
    
    if not es.indices.exists(index=index_name):
        print(f"[WARNING] Index '{index_name}' does not exist. Skipping index check.")
        return {}
    
    # Check each candidate ID
    exists_in_index = {}
    for candidate_id in candidate_ids:
        try:
            result = es.get(index=index_name, id=candidate_id, ignore=[404])
            exists_in_index[candidate_id] = result.get('found', False)
        except:
            exists_in_index[candidate_id] = False
    
    return exists_in_index


def validate_results(source_of_truth: Dict[str, Set[str]], 
                    es_results: Dict[str, Set[str]],
                    check_index: bool = True) -> Dict:
    """
    Compare source of truth with ES results and generate analysis.
    """
    analysis = {
        'total_jobs_in_source': len(source_of_truth),
        'total_jobs_in_es_results': len(es_results),
        'jobs_analyzed': [],
        'summary': {
            'total_candidates_in_source': 0,
            'total_candidates_found_in_es': 0,
            'total_candidates_missing_in_es': 0,
            'overall_recall': 0.0
        }
    }
    
    total_source_candidates = 0
    total_found = 0
    total_missing = 0
    
    # Collect all missing candidate IDs to check in index
    all_missing_ids = set()
    for job_id in source_of_truth.keys():
        es_candidates = es_results.get(job_id, set())
        missing = source_of_truth[job_id] - es_candidates
        all_missing_ids.update(missing)
    
    # Check if missing candidates exist in ES index
    index_check = {}
    if check_index and all_missing_ids:
        print("Checking if missing candidates exist in ES index...")
        index_check = check_candidates_in_index(all_missing_ids)
    
    # Analyze each job
    all_job_ids = set(source_of_truth.keys()) | set(es_results.keys())
    
    for job_id in sorted(all_job_ids):
        source_candidates = source_of_truth.get(job_id, set())
        es_candidates = es_results.get(job_id, set())
        
        found = source_candidates & es_candidates
        missing = source_candidates - es_candidates
        extra = es_candidates - source_candidates
        
        recall = len(found) / len(source_candidates) * 100 if source_candidates else 0
        
        # Check which missing candidates exist in index
        missing_in_index = []
        missing_not_in_index = []
        for missing_id in missing:
            if index_check.get(missing_id, False):
                missing_in_index.append(missing_id)
            else:
                missing_not_in_index.append(missing_id)
        
        job_analysis = {
            'job_id': job_id,
            'source_candidates_count': len(source_candidates),
            'es_candidates_count': len(es_candidates),
            'found_count': len(found),
            'missing_count': len(missing),
            'missing_in_index_count': len(missing_in_index),
            'missing_not_in_index_count': len(missing_not_in_index),
            'extra_count': len(extra),
            'recall_percentage': round(recall, 2),
            'found_candidate_ids': sorted(list(found)),
            'missing_candidate_ids': sorted(list(missing)),
            'missing_in_index_ids': sorted(missing_in_index),
            'missing_not_in_index_ids': sorted(missing_not_in_index)
        }
        
        analysis['jobs_analyzed'].append(job_analysis)
        
        total_source_candidates += len(source_candidates)
        total_found += len(found)
        total_missing += len(missing)
    
    # Calculate overall metrics
    total_missing_in_index = sum(job.get('missing_in_index_count', 0) for job in analysis['jobs_analyzed'])
    total_missing_not_in_index = sum(job.get('missing_not_in_index_count', 0) for job in analysis['jobs_analyzed'])
    
    analysis['summary']['total_candidates_in_source'] = total_source_candidates
    analysis['summary']['total_candidates_found_in_es'] = total_found
    analysis['summary']['total_candidates_missing_in_es'] = total_missing
    analysis['summary']['missing_but_in_index'] = total_missing_in_index
    analysis['summary']['missing_and_not_in_index'] = total_missing_not_in_index
    analysis['summary']['overall_recall'] = round(
        (total_found / total_source_candidates * 100) if total_source_candidates > 0 else 0, 
        2
    )
    analysis['summary']['potential_recall_if_all_indexed'] = round(
        ((total_found + total_missing_in_index) / total_source_candidates * 100) if total_source_candidates > 0 else 0,
        2
    )
    
    return analysis


def print_analysis(analysis: Dict):
    """Print formatted analysis results."""
    print("\n" + "="*80)
    print("VALIDATION ANALYSIS: ES Search Results vs Source of Truth")
    print("="*80)
    
    summary = analysis['summary']
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total jobs in source of truth: {analysis['total_jobs_in_source']}")
    print(f"  Total jobs in ES results: {analysis['total_jobs_in_es_results']}")
    print(f"  Total candidates in source of truth: {summary['total_candidates_in_source']}")
    print(f"  Total candidates found in ES: {summary['total_candidates_found_in_es']}")
    print(f"  Total candidates missing in ES: {summary['total_candidates_missing_in_es']}")
    print(f"    - Missing but exist in index: {summary.get('missing_but_in_index', 0)}")
    print(f"    - Missing and not in index: {summary.get('missing_and_not_in_index', 0)}")
    print(f"  Overall Recall: {summary['overall_recall']}%")
    if summary.get('missing_but_in_index', 0) > 0:
        print(f"  Potential Recall (if search improved): {summary.get('potential_recall_if_all_indexed', 0)}%")
    
    print(f"\n{'='*80}")
    print("PER-JOB ANALYSIS:")
    print(f"{'='*80}")
    
    for job in analysis['jobs_analyzed']:
        if job['source_candidates_count'] > 0:  # Only show jobs that have source candidates
            print(f"\nJob ID: {job['job_id']}")
            print(f"  Source candidates: {job['source_candidates_count']}")
            print(f"  ES candidates: {job['es_candidates_count']}")
            print(f"  Found: {job['found_count']} ({job['recall_percentage']}% recall)")
            print(f"  Missing: {job['missing_count']}")
            if job.get('missing_in_index_count', 0) > 0:
                print(f"    - In index but not found by search: {job['missing_in_index_count']}")
                print(f"      IDs: {', '.join(job.get('missing_in_index_ids', [])[:5])}")
            if job.get('missing_not_in_index_count', 0) > 0:
                print(f"    - Not in index: {job['missing_not_in_index_count']}")
            if job['missing_count'] > 0:
                print(f"  Missing IDs: {', '.join(job['missing_candidate_ids'][:10])}")
                if len(job['missing_candidate_ids']) > 10:
                    print(f"    ... and {len(job['missing_candidate_ids']) - 10} more")
            if job['extra_count'] > 0:
                print(f"  Extra (found by ES but not in source): {job['extra_count']}")


def main():
    results_file = sys.argv[1] if len(sys.argv) > 1 else "results_summary.json"
    source_file = sys.argv[2] if len(sys.argv) > 2 else "job_candidate_mapping.json"
    
    print(f"Loading source of truth from: {source_file}")
    source_of_truth = load_source_of_truth(source_file)
    print(f"  Loaded {len(source_of_truth)} jobs with candidates")
    
    print(f"\nLoading ES results from: {results_file}")
    es_results = load_es_results(results_file)
    print(f"  Loaded {len(es_results)} jobs with candidates")
    
    print("\nValidating results...")
    analysis = validate_results(source_of_truth, es_results, check_index=True)
    
    # Print analysis
    print_analysis(analysis)
    
    # Save detailed analysis to file
    output_file = "validation_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Detailed analysis saved to {output_file}")
    
    return analysis


if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Validation completed!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

