"""Regex-based text splitting utilities (no spaCy required).

This module provides simple, rule-based text splitting that doesn't require spaCy.
While less accurate than spaCy-based splitting for complex cases, it works well for
most common text and has no ML dependencies.

Performance vs. spaCy:
- Faster for short texts (no model loading overhead)
- Good accuracy (~85-90%) for well-formatted text
- May incorrectly split on some edge cases that spaCy handles correctly
- Recommended for: simple use cases, environments without spaCy, quick processing

For best accuracy with complex text, use the spaCy-based functions instead.
Install with: pip install phrasplit[nlp]
"""

from __future__ import annotations

import re
import warnings

from phrasplit.abbreviations import get_abbreviations

# Import preprocessing functions from main splitter
from phrasplit.splitter import (
    _preprocess_text,
    _protect_ellipsis,
    _restore_ellipsis,
    _split_sentence_into_clauses,
    split_paragraphs,
)

# Precompiled common patterns (English-focused, extended by language-specific patterns)
_WEBSITES = re.compile(
    r"\b(?:[\w\-]+\.)+(?:com|net|org|io|gov|edu|me|co|uk|de|fr|es|it)(?=\b|/)"
)
_DIGITS = re.compile(r"(?<=\d)\.(?=\d)")
_MULTIPLE_DOTS = re.compile(r"\.{3,}")
_INITIALS = re.compile(r"\b([A-Z])\.(?=[A-Z]\.)")
_SINGLE_INITIAL = re.compile(r"\b([A-Z])\.")

# Common sentence starters for English (used to identify sentence boundaries)
_SENTENCE_STARTERS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He|She|It|They|Their|Our|We|But|However|That|This|Wherever|"
    r"The|A|An|In|On|At|For|And|Or|So|Yet|As|What|When|Where|Who|Why|How)\b"
)

# Placeholder markers
_PROTECTED_PERIOD = "<prd>"
_SENTENCE_BOUNDARY = "<stop>"
_ELLIPSIS_MARKER = "<ellip>"


def _build_language_patterns(language_model: str) -> dict[str, re.Pattern[str]]:
    """
    Build language-specific regex patterns from abbreviations.

    Args:
        language_model: spaCy language model name (e.g., "en_core_web_sm")

    Returns:
        Dictionary of compiled regex patterns for the language
    """
    abbrevs = get_abbreviations(language_model)

    if not abbrevs:
        # No abbreviations for this language, return empty patterns
        return {
            "prefixes": re.compile(r"(?!)"),  # Never matches
            "suffixes": re.compile(r"(?!)"),
            "acronyms": re.compile(r"(?!)"),
        }

    # Separate abbreviations by type based on common patterns
    # Titles/Prefixes: typically 2-5 characters, often appear before names
    prefixes = {a for a in abbrevs if len(a) <= 5 and a[0].isupper()}

    # Company suffixes: Inc, Ltd, Co, Corp, etc.
    suffixes = {
        a
        for a in abbrevs
        if a
        in {
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "GmbH",
            "AG",
            "SA",
            "SpA",
            "Srl",
            "BV",
            "NV",
            "Cie",
            "Cía",
        }
    }

    # Acronyms: Multiple capital letters with periods (handled separately)
    # We'll use a generic pattern for this

    # Build regex patterns
    patterns = {}

    if prefixes:
        # Match whole word boundaries, case-sensitive
        prefix_pattern = r"\b(?:" + "|".join(re.escape(p) for p in prefixes) + r")\."
        patterns["prefixes"] = re.compile(prefix_pattern)
    else:
        patterns["prefixes"] = re.compile(r"(?!)")

    if suffixes:
        suffix_pattern = r"\b(?:" + "|".join(re.escape(s) for s in suffixes) + r")\."
        patterns["suffixes"] = re.compile(suffix_pattern)
    else:
        patterns["suffixes"] = re.compile(r"(?!)")

    # Acronyms: 2+ capital letters each followed by period
    patterns["acronyms"] = re.compile(r"\b(?:[A-Z]\.){2,}(?:[A-Z]\.?)?")

    return patterns


def split_sentences_simple(
    text: str,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split text into sentences using regex-based rules (no spaCy required).

    This is a simpler, faster alternative to the spaCy-based split_sentences().
    It works well for most common text but may be less accurate on complex cases.

    Algorithm:
    1. Preprocess text (fix hyphenated line breaks, normalize whitespace)
    2. Protect periods in abbreviations, URLs, numbers, etc. with placeholders
    3. Mark sentence boundaries after terminal punctuation (.!?)
    4. Split on boundaries and restore protected periods

    Args:
        text: Input text to split
        language_model: spaCy language model name (used only to determine language
            for abbreviation handling, e.g., "en_core_web_sm", "de_core_news_sm")

    Returns:
        List of sentences (non-empty, stripped)

    Example:
        >>> split_sentences_simple("Dr. Smith is here. She has a Ph.D.")
        ['Dr. Smith is here.', 'She has a Ph.D.']

    Note:
        This function does not apply post-processing corrections like the
        spaCy version (URL splitting, abbreviation merging). For best accuracy,
        use split_sentences() with spaCy installed.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    if not text.strip():
        return []

    try:
        # Apply preprocessing from main splitter (hyphenated line breaks, whitespace)
        text = _preprocess_text(text)

        # Pad text for easier pattern matching
        text = f" {text.strip()}  "

        # Protect ellipsis (reuse from main splitter)
        text = _protect_ellipsis(text)

        # Build language-specific patterns
        patterns = _build_language_patterns(language_model)

        # Protect periods in various contexts (replace with placeholder)

        # 1. Language-specific prefixes (Mr., Dr., Prof., etc.)
        text = patterns["prefixes"].sub(
            lambda m: m.group().replace(".", _PROTECTED_PERIOD), text
        )

        # 2. Language-specific suffixes (Inc., Ltd., Co., etc.)
        text = patterns["suffixes"].sub(
            lambda m: m.group().replace(".", _PROTECTED_PERIOD), text
        )

        # 3. Websites and URLs
        text = _WEBSITES.sub(lambda m: m.group().replace(".", _PROTECTED_PERIOD), text)

        # 4. Numbers with decimals (3.14 → 3<prd>14)
        text = _DIGITS.sub(_PROTECTED_PERIOD, text)

        # 5. Common multi-letter abbreviations (Ph.D., e.g., i.e., etc.)
        text = re.sub(r"\bPh\.D\.?", f"Ph{_PROTECTED_PERIOD}D{_PROTECTED_PERIOD}", text)
        text = re.sub(r"\be\.g\.", f"e{_PROTECTED_PERIOD}g{_PROTECTED_PERIOD}", text)
        text = re.sub(r"\bi\.e\.", f"i{_PROTECTED_PERIOD}e{_PROTECTED_PERIOD}", text)
        text = re.sub(r"\betc\.", f"etc{_PROTECTED_PERIOD}", text)
        text = re.sub(r"\bvs\.", f"vs{_PROTECTED_PERIOD}", text)
        text = re.sub(r"\bal\.", f"al{_PROTECTED_PERIOD}", text)  # et al.

        # 6. Acronyms (U.S.A., J.R.R., etc.)
        text = patterns["acronyms"].sub(
            lambda m: m.group().replace(".", _PROTECTED_PERIOD), text
        )

        # 7. Single initials (E. coli, J. Smith)
        text = _INITIALS.sub(rf"\1{_PROTECTED_PERIOD}", text)
        text = _SINGLE_INITIAL.sub(rf"\1{_PROTECTED_PERIOD}", text)

        # Fix double initials that got separated (J. R. → J.R.)
        text = re.sub(
            rf"\b([A-Z]){_PROTECTED_PERIOD} ([A-Z]){_PROTECTED_PERIOD}",
            rf"\1{_PROTECTED_PERIOD}\2{_PROTECTED_PERIOD}",
            text,
        )

        # Special case: Add sentence boundary after certain abbreviations
        # if followed by sentence starters
        # (Inc. The company → Inc.<stop> The company)
        pattern = rf"\b(?:Inc|Ltd|Jr|Sr|Co){_PROTECTED_PERIOD} "
        pattern += rf"(?={_SENTENCE_STARTERS.pattern})"
        text = re.sub(
            pattern,
            f"{_PROTECTED_PERIOD}{_SENTENCE_BOUNDARY} ",
            text,
        )

        # Add stop marker after protected abbreviations if followed by capital letter
        # (etc. The → etc.<stop> The)
        text = re.sub(
            rf"(e{_PROTECTED_PERIOD}g|i{_PROTECTED_PERIOD}e|etc){_PROTECTED_PERIOD}(?=\s+[A-Z])",
            rf"\1{_PROTECTED_PERIOD}{_SENTENCE_BOUNDARY}",
            text,
        )

        # IMPORTANT: Handle closing quotes after punctuation
        # Protect quotes that immediately follow sentence-ending punctuation
        # so they don't get a double boundary marker
        text = re.sub(r'([.!?])(["\']+)', r"\1<QUOTE>", text)

        # Mark sentence boundaries at terminal punctuation
        text = text.replace(".", f".{_SENTENCE_BOUNDARY}")
        text = text.replace("?", f"?{_SENTENCE_BOUNDARY}")
        text = text.replace("!", f"!{_SENTENCE_BOUNDARY}")

        # Restore quotes AFTER sentence boundaries
        # They should be with the previous sentence
        text = text.replace(
            f"{_SENTENCE_BOUNDARY}<QUOTE>", f"<QUOTE>{_SENTENCE_BOUNDARY}"
        )
        text = text.replace("<QUOTE>", '"')  # Normalize to double quotes

        # Restore protected periods
        text = text.replace(_PROTECTED_PERIOD, ".")

        # Restore ellipsis
        text = _restore_ellipsis(text)

        # Split on boundaries and clean up
        sentences = [
            re.sub(r"\s+", " ", s.strip())
            for s in text.split(_SENTENCE_BOUNDARY)
            if s.strip()
        ]

        return sentences

    except re.error as regex_err:
        warnings.warn(
            f"Regex error in sentence splitting: {regex_err}",
            stacklevel=2,
        )
        return [text.strip()]
    except Exception as e:
        warnings.warn(
            f"Unexpected error in sentence splitting: {e}",
            stacklevel=2,
        )
        return []


def split_clauses_simple(
    text: str,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split text into clauses using regex-based sentence detection (no spaCy required).

    Uses simple sentence splitting, then splits each sentence at commas.
    The comma stays at the end of each clause for natural pause points.

    Args:
        text: Input text
        language_model: spaCy language model name (for language detection)

    Returns:
        List of comma-separated clauses

    Example:
        >>> split_clauses_simple("I like coffee, and I like wine.")
        ['I like coffee,', 'and I like wine.']

    Note:
        This is less accurate than the spaCy version for complex sentence structures.
    """
    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[str] = []

    for para in paragraphs:
        # Get sentences for this paragraph
        sentences = split_sentences_simple(para, language_model)

        # Split each sentence into clauses
        for sent in sentences:
            clauses = _split_sentence_into_clauses(sent)
            result.extend(clauses)

    return result


def split_long_lines_simple(
    text: str,
    max_length: int,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split long lines at sentence/clause boundaries (no spaCy required).

    Uses regex-based sentence detection to split lines intelligently.

    Strategy:
    1. Split at sentence boundaries first
    2. If still too long, split at clause boundaries (commas)
    3. If still too long, split at word boundaries

    Args:
        text: Input text
        max_length: Maximum line length in characters (must be positive)
        language_model: spaCy language model name (for language detection)

    Returns:
        List of lines, each within max_length (except single words exceeding limit)

    Raises:
        ValueError: If max_length is less than 1

    Note:
        This is less accurate than the spaCy version for complex text structures.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        # Check if line is within limit
        if len(line) <= max_length:
            result.append(line)
            continue

        # Split the long line using simple sentence splitting
        sentences = split_sentences_simple(line, language_model)

        # Try to combine sentences up to max_length
        current_line = ""

        for sent in sentences:
            # If sentence itself exceeds max_length, split at clauses
            if len(sent) > max_length:
                # Flush current line first
                if current_line:
                    result.append(current_line)
                    current_line = ""

                # Split at clauses
                clause_lines = _split_at_clauses_simple(sent, max_length)
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

    return result


def _split_at_clauses_simple(text: str, max_length: int) -> list[str]:
    """
    Split text at comma boundaries to fit within max_length.

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
            final_result.extend(_hard_split_simple(line, max_length))
        else:
            final_result.append(line)

    return final_result if final_result else [text]


def _hard_split_simple(text: str, max_length: int) -> list[str]:
    """
    Hard split text at word boundaries when other splitting isn't enough.

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
