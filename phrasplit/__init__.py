"""Phrasplit - Split text into sentences, clauses, or paragraphs."""

from .splitter import (
    Segment,
    split_clauses,
    split_long_lines,
    split_paragraphs,
    split_sentences,
    split_text,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Segment",
    "split_clauses",
    "split_long_lines",
    "split_paragraphs",
    "split_sentences",
    "split_text",
]
