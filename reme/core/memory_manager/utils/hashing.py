"""Utility functions for hashing text content."""

import hashlib


def hash_text(text: str) -> str:
    """Generate SHA-256 hash of text content.

    Args:
        text: Input text to hash

    Returns:
        Hexadecimal representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
