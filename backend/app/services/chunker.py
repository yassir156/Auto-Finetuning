"""
FineTuneFlow — Text Chunking Service.

Splits long text into overlapping chunks using tiktoken for token counting.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Use cl100k_base (GPT-4 / modern LLM tokenizer) as default fallback.
# This gives a reasonable approximation for most models.
DEFAULT_ENCODING = "cl100k_base"


@dataclass
class TextChunk:
    """A single text chunk with metadata."""

    index: int
    content: str
    token_count: int
    char_count: int
    start_char: int  # Character offset in original text
    end_char: int


def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Get a tiktoken encoder, cached by name."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count the number of tokens in text."""
    enc = get_tokenizer(encoding_name)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks based on token count.

    Strategy:
    1. Encode full text into tokens.
    2. Slide a window of `chunk_size_tokens` with overlap of `chunk_overlap_tokens`.
    3. Decode each token window back to text.
    4. Try to break on sentence/paragraph boundaries when possible.

    Args:
        text: The full text to chunk.
        chunk_size_tokens: Max tokens per chunk (default from settings).
        chunk_overlap_tokens: Overlap between consecutive chunks (default from settings).
        encoding_name: tiktoken encoding name.

    Returns:
        List of TextChunk objects.
    """
    if not text or not text.strip():
        return []

    chunk_size = chunk_size_tokens if chunk_size_tokens is not None else settings.DEFAULT_CHUNK_SIZE_TOKENS
    overlap = chunk_overlap_tokens if chunk_overlap_tokens is not None else settings.DEFAULT_CHUNK_OVERLAP_TOKENS

    # Validate: prevent infinite loop or nonsensical values
    if chunk_size <= 0:
        raise ValueError(f"chunk_size_tokens must be > 0, got {chunk_size}")
    if overlap < 0:
        overlap = 0

    # Ensure overlap is smaller than chunk size
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    enc = get_tokenizer(encoding_name)
    tokens = enc.encode(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return []

    # If the whole text fits in one chunk, return it as-is
    stripped = text.strip()
    if total_tokens <= chunk_size:
        return [
            TextChunk(
                index=0,
                content=stripped,
                token_count=total_tokens,
                char_count=len(stripped),
                start_char=0,
                end_char=len(stripped),
            )
        ]

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    # Build a cumulative char-offset map from the original text
    # by decoding token prefixes. Pre-compute boundaries.
    # Approximate: track cumulative decoded length per step
    cumulative_chars = 0

    for i, start in enumerate(range(0, total_tokens, step)):
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_raw = enc.decode(chunk_tokens)

        # Try to find a clean break point (paragraph or sentence boundary)
        chunk_content = _find_clean_break(chunk_text_raw, is_last=(end >= total_tokens))

        token_count = len(enc.encode(chunk_content))
        char_count = len(chunk_content)

        # start_char is approximate due to tokenizer round-trip
        start_char = cumulative_chars

        chunks.append(
            TextChunk(
                index=i,
                content=chunk_content,
                token_count=token_count,
                char_count=char_count,
                start_char=start_char,
                end_char=start_char + char_count,
            )
        )

        # Advance by the step-sized window
        step_text = enc.decode(tokens[start: start + step])
        cumulative_chars += len(step_text)

        if end >= total_tokens:
            break

    logger.info(
        "chunker.done",
        total_tokens=total_tokens,
        num_chunks=len(chunks),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return chunks


def _find_clean_break(text: str, is_last: bool = False) -> str:
    """
    Attempt to trim text at a clean boundary (paragraph/sentence end).

    For the last chunk, we keep all content.
    For other chunks, we try to break at the last paragraph or sentence boundary.
    """
    text = text.strip()

    if is_last or len(text) < 100:
        return text

    # Try to find the last paragraph break (double newline)
    last_para = text.rfind("\n\n")
    if last_para > len(text) * 0.7:  # Only if we keep at least 70% of content
        return text[: last_para].strip()

    # Try to find the last sentence boundary
    for sep in (". ", ".\n", "! ", "!\n", "? ", "?\n"):
        last_sent = text.rfind(sep)
        if last_sent > len(text) * 0.7:
            return text[: last_sent + 1].strip()

    # No good break point found — return as-is
    return text
