# Candidate Search Project

Elasticsearch-based candidate matching system for job postings.

## Overview

This project indexes candidate data into Elasticsearch and provides search functionality to find the top matching candidates for each job posting.

## Prerequisites

1. **Docker and Docker Compose** - For running Elasticsearch
2. **Python 3.8+**
3. **Elasticsearch running** (via docker-compose in parent directory)

## Setup

### 1. Start Elasticsearch

From the parent directory's `Elastic-docker` folder:

```bash
cd ../Elastic-docker
docker-compose up -d
```

This will start Elasticsearch on `http://localhost:9200`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Index Candidates

Index candidate data from a JSON file:

```bash
python index_candidates.py <candidates_json_file_path> [index_name] [limit]
```

Example:
```bash
# Index all candidates
python index_candidates.py candidates.json

# Index with custom index name
python index_candidates.py candidates.json my_index

# Index only first 10 candidates (for testing)
python index_candidates.py candidates.json candidates 10
```

Or with a full path:
```bash
python index_candidates.py "C:\Users\admin\Downloads\merged_result_5 2.json"
```

The script will:
- Load candidates from the JSON file
- Create an Elasticsearch index named `candidates` (or specified name)
- Index all candidate data with searchable fields
- Optionally limit the number of candidates indexed (useful for testing)

### Step 2: Search for Matching Candidates

Search for top 200 candidates matching each job:

```bash
python search_candidates.py <jobs_json_file_path> [output_file_path] [index_name] [limit]
```

Example:
```bash
# Search all jobs
python search_candidates.py jobs.json results.json

# Search with custom index name
python search_candidates.py jobs.json results.json my_index

# Search only first 10 jobs (for testing)
python search_candidates.py jobs.json results.json candidates 10
```

Or with a full path:
```bash
python search_candidates.py "C:\Users\admin\Downloads\job_data.json" results.json
```

The script will:
- Load jobs from the JSON file (handles nested `data` field automatically)
- For each job, search for top 200 matching candidates
- Save detailed results to a JSON file
- Automatically create a summary file (`*_summary.json`) with job_id to candidate_ids mapping
- Optionally limit the number of jobs to process (useful for testing)

### Step 3: Validate Results (Optional)

Compare search results with source of truth:

```bash
python validate_results.py [results_summary_file] [source_of_truth_file]
```

Example:
```bash
python validate_results.py results_summary.json job_candidate_mapping.json
```

This will:
- Compare ES search results with source of truth
- Calculate recall metrics
- Identify missing candidates and whether they exist in the index
- Generate detailed validation analysis

## Data Format

### Candidate Data Format

```json
{
  "candidate_id": "543557298077",
  "email": "jsagostinelli@gmail.com",
  "first_name": "James",
  "last_name": "Agostinelli",
  "skills": [],
  "title_exp": [
    {
      "title": "Technical Project / Program Manager",
      "exp": "05/2023 - 03/2025",
      "months": 22
    }
  ],
  "education": [...],
  "resume_data": "..."
}
```

### Job Data Format

```json
{
  "ID": "29092326",
  "JOBTITLE": "Application Developer Azure Cloud FullStack",
  "SKILLS": "(AZURE RED HAT OPENSHIFT OR ARO ) AND(SPRING BOOT...)",
  "JOBDESCRIPTION": "...",
  "COMPANYNAME": "IBM",
  "STATE": "NY",
  "COUNTRY": "US"
}
```

## Search Matching

The search algorithm uses advanced Elasticsearch queries to match candidates based on:

### Job Title Matching
- **Exact phrase match** (boost 5.0) - Exact job title phrase in candidate titles
- **Multi-match query** (boost 3.0) - Searches across `all_titles` and `resume_data` fields
- **Fuzzy match** (boost 1.5) - Handles typos and variations with automatic fuzziness

### Job Description Matching
- **Key terms extraction** - Extracts meaningful terms from job description (filters stop words)
- **Individual term matching** (boost 0.8) - Each key term searched in resume data
- **Multi-match query** (boost 1.0) - Combined search across resume data, titles, and skills text

### Query Features
- **Minimum score threshold** (0.5) - Filters out very low relevance results
- **Best fields matching** - Uses Elasticsearch's `best_fields` type for optimal scoring
- **Boolean query structure** - Combines multiple matching strategies with `should` clauses

**Note:** Skills and location filtering have been removed from the current implementation. The search focuses on job title and description matching only.

## Output Format

### Detailed Results File (`results.json`)

The detailed results file includes:
- Job information (ID, title, company, job status, skills, location)
- Number of candidates found
- Top 200 candidates with:
  - Full candidate details (name, email, skills, experience, education, resume data)
  - Relevance score (`_score`) from Elasticsearch

### Summary File (`results_summary.json`)

Automatically generated summary file with:
- Job ID and basic job information
- Candidates count
- Array of candidate IDs for each job

Example:
```json
{
  "job_id": "30498809",
  "job_title": ".NET developer",
  "company_name": "DXC Commercial",
  "candidates_count": 4,
  "candidate_ids": ["12929805834179", "2449899727476", ...]
}
```

## Project Files

- `index_candidates.py` - Index candidate data into Elasticsearch
- `search_candidates.py` - Search for matching candidates
- `validate_results.py` - Validate search results against source of truth
- `diagnose_search.py` - Diagnostic tool for investigating specific candidate matches
- `candidates.json` - Candidate data file
- `jobs.json` - Job data file
- `job_candidate_mapping.json` - Source of truth for job-candidate mappings

