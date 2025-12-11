# Batch Ingestion Optimization

## Overview

The `/ingest` endpoint is optimized for processing exactly **128 candidates** per request using Voyage AI batch inference with maximum concurrency and parallelism.

## Architecture

### 1. **Parallel Preprocessing** (Concurrent)
- **128 candidates** are preprocessed **simultaneously** using `ThreadPoolExecutor`
- Up to **32 worker threads** process candidates in parallel
- Each candidate's text and payload are built concurrently
- **Order is preserved** - results maintain the original input order

### 2. **Single Batch Embedding** (Voyage AI)
- All **128 preprocessed texts** are sent to Voyage AI in **one batch call**
- Voyage AI `voyage-3.5-lite` handles up to 128 documents per batch
- Single API call = maximum efficiency and cost savings

### 3. **Batch Upsert to Qdrant**
- All **128 vectors** are upserted to Qdrant in **one batch operation**
- Qdrant handles batch upserts efficiently
- Single network round-trip to Qdrant

## Performance Flow

```
Input: 128 JSON candidates
    ↓
[PARALLEL] Preprocess all 128 candidates simultaneously
    ├─ Thread 1: Candidate 1 (text + payload)
    ├─ Thread 2: Candidate 2 (text + payload)
    ├─ ...
    └─ Thread 32: Candidate 32 (text + payload)
    (continues until all 128 are processed)
    ↓
[WAIT] All preprocessing complete
    ↓
[BATCH] Single Voyage AI API call with 128 texts
    ↓
[WAIT] Embeddings received (128 vectors)
    ↓
[BATCH] Single Qdrant upsert with 128 vectors + payloads
    ↓
Output: Results for all 128 candidates
```

## Key Optimizations

1. **Concurrent Preprocessing**: 32 parallel threads process candidates simultaneously
2. **Single Batch Embedding**: One API call to Voyage AI for all 128 documents
3. **Batch Qdrant Upsert**: One network call to Qdrant for all vectors
4. **Order Preservation**: Results maintain input order despite parallel processing

## API Usage

### Request Format
```json
{
  "candidates": [
    {
      "candidateId": "8284346",
      "clientId": "8428fbd0-a3e7-40f5-a274-27425226f96a",
      "resumeData": { ... }
    },
    ... (up to 128 candidates)
  ],
  "collection_name": "candidates_store"
}
```

### Response Format
```json
{
  "total": 128,
  "successful": 128,
  "failed": 0,
  "results": [
    {
      "candidate_id": "8284346",
      "success": true,
      "vector_id": "8284346"
    },
    ...
  ]
}
```

## Performance Metrics

Expected performance for 128 candidates:
- **Preprocessing**: ~2-5 seconds (parallel, 32 workers)
- **Embedding**: ~3-8 seconds (single batch to Voyage AI)
- **Qdrant Upsert**: ~1-3 seconds (single batch)
- **Total**: ~6-16 seconds for 128 candidates

## Constraints

- **Maximum batch size**: 128 candidates (Voyage AI limit)
- **Minimum batch size**: 1 candidate
- **Validation**: API returns 400 error if > 128 candidates submitted

## Error Handling

- **Preprocessing errors**: Individual candidate failures are logged, fallback payloads created
- **Embedding errors**: Entire batch fails (all 128 candidates marked as failed)
- **Qdrant errors**: Entire batch fails (all 128 candidates marked as failed)

## Logging

The endpoint logs:
- Preprocessing start/completion with timing
- Embedding generation with timing
- Qdrant upsert with timing
- Total processing time breakdown

Example log output:
```
INFO: Preprocessing 128 candidates in parallel...
INFO: Preprocessing complete in 3.45s. Sample text length: 5420 chars
INFO: Generating embeddings for 128 candidates using Voyage AI batch inference...
INFO: Embeddings generated in 5.23s. Total vectors: 128
INFO: Upserting 128 vectors to Qdrant collection 'candidates_store'...
INFO: Qdrant upsert complete in 1.87s
INFO: Ingestion complete: 128 successful, 0 failed in 10.55s total
INFO:   - Preprocessing: 3.45s
INFO:   - Embedding: 5.23s
INFO:   - Qdrant upsert: 1.87s
```

