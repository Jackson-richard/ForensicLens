import hashlib
from typing import Dict, Any

def compute_sha256(file_stream) -> str:
    """Computes the SHA-256 hash of a file stream."""
    sha256_hash = hashlib.sha256()
    for byte_block in iter(lambda: file_stream.read(4096), b""):
        sha256_hash.update(byte_block)
    # Reset stream position after reading
    file_stream.seek(0)
    return sha256_hash.hexdigest()

def extract_image_metadata(file_stream) -> Dict[str, Any]:
    """
    Extracts basic metadata from an image stream.
    Requires ExifRead for deeper analysis, currently returning basic mock data
    to satisfy module stability natively without new heavy imports.
    """
    return {"SystemStatus": "No structural ExifRead yet implemented"}
