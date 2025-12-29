"""Text splitting utilities using spaCy for NLP-based sentence and clause detection."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from phrasplit.abbreviations import get_abbreviations, get_sentence_starters

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

# Regex for hyphenated line breaks (e.g., "recom-\nmendation" -> "recommendation")
_HYPHENATED_LINEBREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")

# URL pattern for splitting
_URL_PATTERN = re.compile(r"(https?://\S+)")

# Pattern to detect abbreviation at end of sentence
# Matches: word ending with period, where word (without period) is in abbreviations
_ABBREV_END_PATTERN = re.compile(r"(\b[A-Za-z]+)\.\s*$")


def _fix_hyphenated_linebreaks(text: str) -> str:
    """
    Fix hyphenated line breaks commonly found in PDFs and OCR text.

    Joins words that were split across lines with a hyphen.
    Example: "recom-\\nmendation" -> "recommendation"

    Args:
        text: Input text

    Returns:
        Text with hyphenated line breaks fixed
    """
    return _HYPHENATED_LINEBREAK.sub(r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    """
    Normalize multiple whitespace characters to single spaces.

    Preserves paragraph breaks (double newlines) but normalizes
    other whitespace sequences.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # First preserve paragraph breaks by using a placeholder
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Normalize other whitespace (but not newlines in paragraph breaks)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


def _preprocess_text(text: str) -> str:
    """
    Apply preprocessing steps to clean up text before NLP processing.

    Steps:
    1. Fix hyphenated line breaks (common in PDFs)
    2. Normalize whitespace

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    text = _fix_hyphenated_linebreaks(text)
    text = _normalize_whitespace(text)
    return text


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


def _split_urls(sentences: list[str]) -> list[str]:
    """
    Split sentences that contain multiple URLs.

    URLs are often listed one per line in source text, but spaCy may merge them.
    This function splits sentences only when there are 2+ URLs present.

    Args:
        sentences: List of sentences from spaCy

    Returns:
        List of sentences with multiple URLs properly separated
    """
    result: list[str] = []

    for sent in sentences:
        # Check if sentence contains URLs
        if "http://" not in sent and "https://" not in sent:
            result.append(sent)
            continue

        # Count URLs in the sentence
        url_matches = list(_URL_PATTERN.finditer(sent))

        # Only split if there are multiple URLs
        if len(url_matches) < 2:
            result.append(sent)
            continue

        # Split at URL boundaries - each URL becomes its own "sentence"
        # along with any text that follows it until the next URL
        last_end = 0
        for i, match in enumerate(url_matches):
            # Text before this URL (only for first URL)
            if i == 0 and match.start() > 0:
                prefix = sent[: match.start()].strip()
                if prefix:
                    # Include prefix with first URL
                    next_url_start = (
                        url_matches[i + 1].start()
                        if i + 1 < len(url_matches)
                        else len(sent)
                    )
                    part = sent[:next_url_start].strip()
                    result.append(part)
                    last_end = next_url_start
                    continue

            # For subsequent URLs or if no prefix
            if match.start() >= last_end:
                next_url_start = (
                    url_matches[i + 1].start()
                    if i + 1 < len(url_matches)
                    else len(sent)
                )
                part = sent[match.start() : next_url_start].strip()
                if part:
                    result.append(part)
                last_end = next_url_start

    return result


def _merge_abbreviation_splits(
    sentences: list[str],
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Merge sentences that were incorrectly split after abbreviations.

    spaCy sometimes splits after abbreviations like "M.D." or "U.S." when
    followed by a name or continuation. This function merges such cases.

    Conservative approach: only merge if:
    1. Previous sentence ends with a known abbreviation + period
    2. Next sentence starts with a capital letter (likely a name/continuation)
    3. Next sentence does NOT start with a common sentence starter

    Args:
        sentences: List of sentences from spaCy
        language_model: spaCy language model name (for language-specific abbreviations)

    Returns:
        List of sentences with abbreviation splits merged
    """
    # Get language-specific abbreviations
    abbreviations = get_abbreviations(language_model)

    # If no abbreviations for this language, return unchanged
    if not abbreviations:
        return sentences

    if len(sentences) <= 1:
        return sentences

    # Get common sentence starters
    sentence_starters = get_sentence_starters()

    result: list[str] = []
    i = 0

    while i < len(sentences):
        current = sentences[i]

        # Check if we should merge with the next sentence
        if i + 1 < len(sentences):
            next_sent = sentences[i + 1]

            # Check if current sentence ends with an abbreviation
            match = _ABBREV_END_PATTERN.search(current)
            if match:
                abbrev = match.group(1)
                # Check if it's a known abbreviation for this language
                if abbrev in abbreviations:
                    # Check if next sentence starts with a word that's likely a name
                    # (capital letter, not a common sentence starter)
                    next_words = next_sent.split()
                    if next_words:
                        first_word = next_words[0]
                        # Merge if first word is capitalized but not a sentence starter
                        # and not all caps (which might be an acronym/heading)
                        if (
                            first_word[0].isupper()
                            and first_word not in sentence_starters
                            and not first_word.isupper()
                        ):
                            # Merge the sentences
                            merged = current + " " + next_sent
                            result.append(merged)
                            i += 2
                            continue

        result.append(current)
        i += 1

    return result


def _apply_corrections(
    sentences: list[str],
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Apply post-processing corrections to fix common spaCy segmentation errors.

    Corrections applied (in order):
    1. Merge sentences incorrectly split after abbreviations (reduces count)
    2. Split sentences containing multiple URLs (increases count)

    Args:
        sentences: List of sentences from spaCy
        language_model: spaCy language model name (for language-specific corrections)

    Returns:
        Corrected list of sentences
    """
    # First merge abbreviation splits (need to combine before URL split)
    sentences = _merge_abbreviation_splits(sentences, language_model)

    # Then split URLs (increases sentence count)
    sentences = _split_urls(sentences)

    return sentences


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

    Applies preprocessing to fix hyphenated line breaks and normalize whitespace.

    Args:
        text: Input text

    Returns:
        List of paragraphs (non-empty, stripped)
    """
    text = _preprocess_text(text)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(
    text: str,
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
) -> list[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text: Input text
        language_model: spaCy language model to use
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling).
            Default is True.

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

    # Apply post-processing corrections if enabled
    if apply_corrections:
        result = _apply_corrections(result, language_model)

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
