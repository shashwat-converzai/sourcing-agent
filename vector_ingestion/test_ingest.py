"""
Test script to verify the ingest endpoint works with both JSON formats.
"""
import json
import requests
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/ingest"

# Test with sample_candidate.json (array format)
print("Testing with sample_candidate.json (direct array format)...")
sample_file = Path(__file__).parent / "data" / "sample_candidate.json"

with open(sample_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    # Take first candidate for testing
    test_data = [data[0]] if isinstance(data, list) else data
    
    response = requests.post(
        f"{API_URL}?collection_name=candidates_store",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Processed {result['total']} candidates")
        print(f"Successful: {result['successful']}, Failed: {result['failed']}")
    else:
        print(f"Error: {response.text}")

print("\n" + "="*80 + "\n")

# Test with wrapped format
print("Testing with wrapped format...")
wrapped_data = {
    "candidates": test_data,
    "collection_name": "candidates_store"
}

response = requests.post(
    API_URL,
    json=wrapped_data,
    headers={"Content-Type": "application/json"}
)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Success! Processed {result['total']} candidates")
    print(f"Successful: {result['successful']}, Failed: {result['failed']}")
else:
    print(f"Error: {response.text}")

