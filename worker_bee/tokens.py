"""Token counting with optional tiktoken support."""

from __future__ import annotations

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _encoder = None
    _HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """Count tokens in text. Uses tiktoken if available, else chars / 3."""
    if _HAS_TIKTOKEN:
        return len(_encoder.encode(text))
    return len(text) // 3
