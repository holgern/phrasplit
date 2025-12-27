"""Phrasplit - Split text into sentences, clauses, or paragraphs."""

from .splitter import (
    split_clauses,
    split_long_lines,
    split_paragraphs,
    split_sentences,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "split_clauses",
    "split_long_lines",
    "split_paragraphs",
    "split_sentences",
]
