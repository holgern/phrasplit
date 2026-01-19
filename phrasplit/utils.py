"""Utilities for validating segment boundaries against markup patterns.

This module provides helper functions for working with text that contains
markup or placeholders. It validates that segmentation doesn't break through
these special regions.

The utilities are markup-agnostic - you provide the regex pattern that matches
your markup syntax, and the functions ensure segments don't split it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phrasplit.types import SplitSegment


# Common markup patterns for reference
# These are not used by default - you must pass them explicitly
COMMON_PATTERNS = {
    "ssmd": r"\[([^\]]+)\]\{[^}]+\}",  # SSMD: [text]{lang="de"}
    "speechmarkdown": r"\(\(([^)]+)\)\[[^\]]+\]\)",  # Speech Markdown:
    # ((text)[key:"value";...])
    "mustache": r"\{\{[^}]+\}\}",  # Mustache/Handlebars: {{variable}}
    "html_tag": r"<[^>]+>[^<]*</[^>]+>",  # HTML/XML: <tag>content</tag>
    "markdown_link": r"\[([^\]]+)\]\([^)]+\)",  # Markdown: [text](url)
}


def validate_no_placeholder_breaks(
    text: str,
    segments: list[SplitSegment],
    *,
    placeholder_pattern: str,
) -> list[str]:
    """Validate that segment boundaries don't split placeholder regions.

    This function checks if any segment boundary falls within a placeholder
    or markup region. This helps ensure that markup isn't broken by sentence
    splitting.

    Args:
        text: Original input text containing markup/placeholders
        segments: List of segments from split_with_offsets()
        placeholder_pattern: Regex pattern matching placeholder regions.
            Must be a valid Python regex pattern.

    Returns:
        List of warning messages for any broken placeholders.
        Empty list if no issues found.

    Raises:
        ValueError: If placeholder_pattern is not a valid regex pattern

    Example - SSMD markup:
        >>> # SSMD uses [text]{lang="de"} syntax
        >>> text = "Hello [world]{lang='de'}. How are you?"
        >>> segments = split_with_offsets(text)
        >>> warnings = validate_no_placeholder_breaks(
        ...     text, segments,
        ...     placeholder_pattern=r"\\[([^\\]]+)\\]\\{[^}]+\\}"
        ... )
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")

    Example - Speech Markdown:
        >>> # Speech Markdown uses ((text)[key:"value";...]) syntax
        >>> text = "Hello ((world)[rate:'slow';]). How are you?"
        >>> segments = split_with_offsets(text)
        >>> warnings = validate_no_placeholder_breaks(
        ...     text, segments,
        ...     placeholder_pattern=r"\\(\\(([^)]+)\\)\\[[^\\]]+\\]\\)"
        ... )

    Example - HTML/XML tags:
        >>> text = "Hello <phoneme ph='wɝld'>world</phoneme>. Next sentence."
        >>> segments = split_with_offsets(text)
        >>> warnings = validate_no_placeholder_breaks(
        ...     text, segments,
        ...     placeholder_pattern=r"<[^>]+>[^<]*</[^>]+>"
        ... )

    Example - Using COMMON_PATTERNS:
        >>> from phrasplit.utils import COMMON_PATTERNS
        >>> text = "Text with [markup]{lang='en'}."
        >>> segments = split_with_offsets(text)
        >>> warnings = validate_no_placeholder_breaks(
        ...     text, segments,
        ...     placeholder_pattern=COMMON_PATTERNS["ssmd"]
        ... )
    """
    # Validate regex pattern
    try:
        placeholder_regex = re.compile(placeholder_pattern)
    except re.error as e:
        raise ValueError(
            f"Invalid regex pattern: {placeholder_pattern!r}. Error: {e}"
        ) from e

    warnings: list[str] = []

    # Find all placeholder regions in the text
    placeholders = list(placeholder_regex.finditer(text))

    if not placeholders:
        return warnings

    # For each placeholder, check if it's split across segments
    for match in placeholders:
        ph_start = match.start()
        ph_end = match.end()
        ph_text = match.group(0)

        # Find which segment(s) contain this placeholder
        containing_segments = []
        for seg in segments:
            # Check if segment overlaps with placeholder
            if (
                seg.char_start <= ph_start < seg.char_end
                or seg.char_start < ph_end <= seg.char_end
            ):
                containing_segments.append(seg)

        # If placeholder spans multiple segments, that's a problem
        if len(containing_segments) > 1:
            seg_ids = ", ".join(s.id for s in containing_segments)
            warnings.append(
                f"Placeholder {ph_text!r} at position {ph_start}-{ph_end} "
                f"is split across segments: {seg_ids}"
            )
        elif len(containing_segments) == 0:
            # Placeholder is in a gap between segments
            # (shouldn't happen with proper offsets)
            warnings.append(
                f"Placeholder {ph_text!r} at position {ph_start}-{ph_end} "
                f"is not contained in any segment"
            )

    return warnings


def suggest_splitting_mode(
    text: str,
    *,
    placeholder_pattern: str,
) -> str:
    """Suggest the safest splitting mode for text with markup/placeholders.

    Analyzes the text to determine if it contains placeholders and suggests
    an appropriate splitting mode to minimize placeholder breaks.

    Args:
        text: Input text containing markup/placeholders
        placeholder_pattern: Regex pattern matching placeholder regions.
            Must be a valid Python regex pattern.

    Returns:
        Suggested mode: "paragraph", "sentence", or "clause"

    Raises:
        ValueError: If placeholder_pattern is not a valid regex pattern

    Example - SSMD:
        >>> from phrasplit.utils import COMMON_PATTERNS
        >>> text = "Short [tag]{lang='en'}. Another [tag]{lang='de'}."
        >>> mode = suggest_splitting_mode(
        ...     text,
        ...     placeholder_pattern=COMMON_PATTERNS["ssmd"]
        ... )
        >>> segments = split_with_offsets(text, mode=mode)

    Example - Mustache templates:
        >>> text = "Hello {{name}}. Welcome to {{place}}."
        >>> mode = suggest_splitting_mode(
        ...     text,
        ...     placeholder_pattern=COMMON_PATTERNS["mustache"]
        ... )
    """
    # Validate regex pattern
    try:
        placeholder_regex = re.compile(placeholder_pattern)
    except re.error as e:
        raise ValueError(
            f"Invalid regex pattern: {placeholder_pattern!r}. Error: {e}"
        ) from e

    placeholders = list(placeholder_regex.finditer(text))

    if not placeholders:
        # No placeholders, any mode is safe
        return "sentence"

    # Count placeholders per line/sentence
    lines = text.split("\n")
    max_placeholders_per_line = max(
        (len(list(placeholder_regex.finditer(line))) for line in lines), default=0
    )

    if max_placeholders_per_line > 5:
        # Many placeholders per line, use paragraph mode to be safe
        return "paragraph"
    elif max_placeholders_per_line > 2:
        # Moderate placeholders, sentence mode should be okay
        return "sentence"
    else:
        # Few placeholders, clause mode is safe
        return "clause"
