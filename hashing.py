import hashlib
import json
from typing import Dict, Any

def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    Generate a deterministic SHA-256 hash of the input data.
    """
    # Create a copy and remove non-essential fields
    data_copy = data.copy()
    
    # Remove fields that shouldn't be hashed
    fields_to_remove = ['submission_timestamp', 'applicant_id', 'customer_id']
    for field in fields_to_remove:
        if field in data_copy:
            del data_copy[field]
    
    # Sort data to ensure consistent hashing
    sorted_data = json.dumps(data_copy, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(sorted_data.encode()).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    """
    Verify that data matches the original hash.
    """
    current_hash = generate_data_hash(data)
    return current_hash == original_hash
