"""Text splitting utilities using spaCy for NLP-based sentence and clause detection."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language  # type: ignore[import-not-found]

try:
    import spacy  # type: ignore[import-not-found]

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


# Cache for loaded spaCy model
_nlp_cache: dict[str, Language] = {}

# Placeholder for ellipsis during spaCy processing
_ELLIPSIS_PLACEHOLDER = "\u2026"  # Unicode ellipsis character


def _protect_ellipsis(text: str) -> str:
    """
    Replace ellipsis patterns with a placeholder to prevent sentence splitting.

    Handles:
    - Spaced ellipsis: ". . ." (dot-space-dot-space-dot)
    - Regular ellipsis: "..." (three or more consecutive dots)
    - Unicode ellipsis: U+2026 (single ellipsis character)

    All patterns are replaced with a Unicode ellipsis placeholder (U+2026)
    which is later restored to spaced ellipsis ". . ." format.
    """
    # Replace spaced ellipsis first (. . .)
    text = re.sub(r"\.\s+\.\s+\.", _ELLIPSIS_PLACEHOLDER, text)
    # Replace regular ellipsis (...)
    text = re.sub(r"\.{3,}", _ELLIPSIS_PLACEHOLDER, text)
    # Unicode ellipsis is already the placeholder, no action needed
    return text


def _restore_ellipsis(text: str) -> str:
    """Restore ellipsis placeholder back to spaced ellipsis."""
    return text.replace(_ELLIPSIS_PLACEHOLDER, ". . .")


def _get_nlp(language_model: str = "en_core_web_sm") -> Language:
    """Get or load a spaCy model (cached).

    Args:
        language_model: Name of the spaCy language model to load

    Returns:
        Loaded spaCy Language model

    Raises:
        ImportError: If spaCy is not installed
        OSError: If the specified language model is not found
    """
    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is required for this feature. Install with: pip install phrasplit"
        )

    if language_model not in _nlp_cache:
        try:
            # spacy is guaranteed to be not None here due to SPACY_AVAILABLE check above
            assert spacy is not None
            _nlp_cache[language_model] = spacy.load(language_model)
        except OSError:
            raise OSError(
                f"spaCy language model '{language_model}' not found. "
                f"Download with: python -m spacy download {language_model}"
            ) from None

    return _nlp_cache[language_model]


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs (separated by double newlines).

    Args:
        text: Input text

    Returns:
        List of paragraphs (non-empty, stripped)
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(
    text: str,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text: Input text
        language_model: spaCy language model to use

    Returns:
        List of sentences
    """
    nlp = _get_nlp(language_model)
    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[str] = []
    for para in paragraphs:
        # Protect ellipsis from being treated as sentence boundaries
        para = _protect_ellipsis(para)

        # Process paragraph into sentences
        doc = nlp(para)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        for sent in sentences:
            # Restore ellipsis in the sentence
            sent = _restore_ellipsis(sent)
            result.append(sent)

    return result


def _split_sentence_into_clauses(sentence: str) -> list[str]:
    """
    Split a sentence into comma-separated parts for audiobook creation.

    Splits only at commas, keeping the comma at the end of each part.
    This creates natural pause points for text-to-speech processing.

    Args:
        sentence: A single sentence

    Returns:
        List of comma-separated parts
    """
    # Pattern to split after comma followed by space
    # Using positive lookbehind to keep comma at end of clause
    parts = re.split(r"(?<=,)\s+", sentence)

    # Filter empty parts and strip whitespace
    clauses = [p.strip() for p in parts if p.strip()]

    return clauses if clauses else [sentence]


def split_clauses(
    text: str,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split text into comma-separated parts for audiobook creation.

    Uses spaCy for sentence detection, then splits each sentence at commas.
    The comma stays at the end of each part, creating natural pause points
    for text-to-speech processing.

    Args:
        text: Input text
        language_model: spaCy language model to use

    Returns:
        List of comma-separated parts

    Example:
        Input: "I do like coffee, and I like wine."
        Output: ["I do like coffee,", "and I like wine."]
    """
    nlp = _get_nlp(language_model)
    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[str] = []
    for para in paragraphs:
        # Protect ellipsis from being treated as sentence boundaries
        para = _protect_ellipsis(para)

        # Process paragraph into sentences
        doc = nlp(para)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Process each sentence into clauses
        for sent in sentences:
            # Restore ellipsis in the sentence
            sent = _restore_ellipsis(sent)

            # Split sentence at clause boundaries
            clauses = _split_sentence_into_clauses(sent)
            result.extend(clauses)

    return result


def _split_at_clauses(text: str, max_length: int) -> list[str]:
    """
    Split text at comma boundaries for audiobook creation.

    Args:
        text: Text to split
        max_length: Maximum line length

    Returns:
        List of lines
    """
    # Split at commas, keeping the comma with the preceding text
    parts = re.split(r"(?<=,)\s+", text)

    result: list[str] = []
    current_line = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if not current_line:
            current_line = part
        elif len(current_line) + 1 + len(part) <= max_length:
            current_line += " " + part
        else:
            if current_line:
                result.append(current_line)
            current_line = part

    if current_line:
        result.append(current_line)

    # If still too long, do hard split at word boundaries
    final_result: list[str] = []
    for line in result:
        if len(line) > max_length:
            final_result.extend(_hard_split(line, max_length))
        else:
            final_result.append(line)

    return final_result if final_result else [text]


def _hard_split(text: str, max_length: int) -> list[str]:
    """
    Hard split text at word boundaries when clause splitting isn't enough.

    Args:
        text: Text to split
        max_length: Maximum line length

    Returns:
        List of lines
    """
    words = text.split()
    result: list[str] = []
    current_line = ""

    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + 1 + len(word) <= max_length:
            current_line += " " + word
        else:
            result.append(current_line)
            current_line = word

    if current_line:
        result.append(current_line)

    return result if result else [text]


def _split_at_boundaries(text: str, max_length: int, nlp: Language) -> list[str]:
    """
    Split text at sentence/clause boundaries to fit within max_length.

    Args:
        text: Text to split
        max_length: Maximum line length
        nlp: spaCy language model

    Returns:
        List of lines
    """
    # Protect ellipsis before spaCy processing
    protected_text = _protect_ellipsis(text)

    # First, try splitting by sentences
    doc = nlp(protected_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    result: list[str] = []
    current_line = ""

    for sent in sentences:
        # Restore ellipsis in the sentence
        sent = _restore_ellipsis(sent)
        # If sentence itself exceeds max_length, split at clauses
        if len(sent) > max_length:
            # Flush current line first
            if current_line:
                result.append(current_line)
                current_line = ""
            # Split sentence at clause boundaries
            clause_lines = _split_at_clauses(sent, max_length)
            result.extend(clause_lines)
        elif not current_line:
            current_line = sent
        elif len(current_line) + 1 + len(sent) <= max_length:
            current_line += " " + sent
        else:
            result.append(current_line)
            current_line = sent

    if current_line:
        result.append(current_line)

    return result if result else [text]


def split_long_lines(
    text: str,
    max_length: int,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split lines exceeding max_length at clause/sentence boundaries.

    Strategy:
    1. First try to split at sentence boundaries
    2. If still too long, split at clause boundaries (commas, semicolons, etc.)
    3. If still too long, split at word boundaries

    Args:
        text: Input text
        max_length: Maximum line length in characters (must be positive)
        language_model: spaCy language model to use

    Returns:
        List of lines, each within max_length (except single words exceeding limit)

    Raises:
        ValueError: If max_length is less than 1
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")

    nlp = _get_nlp(language_model)

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        # Check if line is within limit
        if len(line) <= max_length:
            result.append(line)
            continue

        # Split the long line
        split_lines = _split_at_boundaries(line, max_length, nlp)
        result.extend(split_lines)

    return result
