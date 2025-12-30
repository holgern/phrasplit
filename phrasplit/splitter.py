"""Text splitting utilities using spaCy for NLP-based sentence and clause detection."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from phrasplit.abbreviations import (
    get_abbreviations,
    get_sentence_ending_abbreviations,
    get_sentence_starters,
)


class Segment(NamedTuple):
    """A text segment with position information.

    Attributes:
        text: The text content of the segment
        paragraph: Paragraph index (0-based) within the document
        sentence: Sentence index (0-based) within the paragraph.
            None for paragraph mode.
    """

    text: str
    paragraph: int
    sentence: int | None = None


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

# Placeholders for ellipsis during spaCy processing
# We use Unicode private use area characters to avoid collision with real text
_ELLIPSIS_3_PLACEHOLDER = "\ue000"  # 3 dots: ...
_ELLIPSIS_4_PLACEHOLDER = "\ue001"  # 4 dots: ....
_ELLIPSIS_SPACED_PLACEHOLDER = "\ue002"  # Spaced: . . .
_ELLIPSIS_UNICODE_PLACEHOLDER = "\ue003"  # Unicode ellipsis: …
_ELLIPSIS_LONG_PREFIX = "\ue004"  # Prefix for 5+ dots (followed by count digit)

# Regex for hyphenated line breaks (e.g., "recom-\nmendation" -> "recommendation")
_HYPHENATED_LINEBREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")

# URL pattern for splitting
_URL_PATTERN = re.compile(r"(https?://\S+)")

# Pattern to detect abbreviation at end of sentence
# Matches: word ending with period, where word (without period) is in abbreviations
_ABBREV_END_PATTERN = re.compile(r"(\b[A-Za-z]+)\.\s*$")

# Default maximum chunk size for spaCy processing (will be capped by nlp.max_length)
_DEFAULT_MAX_CHUNK_SIZE = 500000

# Safety margin at chunk boundaries to avoid cutting sentences
_DEFAULT_SAFETY_MARGIN = 100


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
    Replace ellipsis patterns with placeholders to prevent sentence splitting.

    Handles:
    - Spaced ellipsis: ". . ." (dot-space-dot-space-dot)
    - Regular ellipsis: "..." (three consecutive dots)
    - Four dots: "...." (often used for ellipsis + period)
    - Five or more dots: "....." etc.
    - Unicode ellipsis: U+2026 (single ellipsis character)

    Each pattern is replaced with a unique placeholder that preserves information
    about the original format, allowing exact restoration later.
    """

    # Replace spaced ellipsis first (. . .) - must come before regular dots
    text = text.replace(". . .", _ELLIPSIS_SPACED_PLACEHOLDER)

    # Replace unicode ellipsis
    text = text.replace("\u2026", _ELLIPSIS_UNICODE_PLACEHOLDER)

    # Replace longer dot sequences first (5+ dots), encoding the count
    # Use offset of 0xE010 (private use area) to avoid control characters
    # chr(0) - chr(31) are control chars, chr(9) is tab, chr(10) is newline
    def replace_long_dots(match: re.Match[str]) -> str:
        count = len(match.group(0))
        # Encode count in private use area: U+E010 + count
        # This avoids control characters and whitespace
        return _ELLIPSIS_LONG_PREFIX + chr(0xE010 + count)

    text = re.sub(r"\.{5,}", replace_long_dots, text)

    # Replace 4 dots
    text = text.replace("....", _ELLIPSIS_4_PLACEHOLDER)

    # Replace 3 dots (must come after 4+ to avoid partial matches)
    text = text.replace("...", _ELLIPSIS_3_PLACEHOLDER)

    return text


def _restore_ellipsis(text: str) -> str:
    """Restore ellipsis placeholders back to their original format."""
    # Restore in reverse order of protection

    # Restore 3 dots
    text = text.replace(_ELLIPSIS_3_PLACEHOLDER, "...")

    # Restore 4 dots
    text = text.replace(_ELLIPSIS_4_PLACEHOLDER, "....")

    # Restore long dot sequences (5+)
    def restore_long_dots(match: re.Match[str]) -> str:
        # Decode count from private use area offset
        count = ord(match.group(1)) - 0xE010
        return "." * count

    # Use re.DOTALL so (.) matches any character including newline (chr(10))
    text = re.sub(
        _ELLIPSIS_LONG_PREFIX + r"(.)", restore_long_dots, text, flags=re.DOTALL
    )

    # Restore unicode ellipsis
    text = text.replace(_ELLIPSIS_UNICODE_PLACEHOLDER, "\u2026")

    # Restore spaced ellipsis
    text = text.replace(_ELLIPSIS_SPACED_PLACEHOLDER, ". . .")

    return text


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
    2. The abbreviation is NOT one that commonly ends sentences (etc., Inc., etc.)
    3. Next sentence starts with a capital letter (likely a name/continuation)
    4. Next sentence does NOT start with a common sentence starter

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

    # Get common sentence starters and sentence-ending abbreviations
    sentence_starters = get_sentence_starters()
    sentence_ending_abbrevs = get_sentence_ending_abbreviations()

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
                # BUT skip if it's an abbreviation that commonly ends sentences
                if abbrev in abbreviations and abbrev not in sentence_ending_abbrevs:
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


# Pattern to detect ellipsis followed by a new sentence
# Matches: 3+ dots OR spaced ellipsis, followed by whitespace,
# optional quotes, and capital letter
_ELLIPSIS_SENTENCE_BREAK = re.compile(
    r'(\.{3,}|\. \. \.)\s+(["\'\u201c\u201d\u2018\u2019]*[A-Z])',
)


def _split_after_ellipsis(sentences: list[str]) -> list[str]:
    """
    Split sentences that contain ellipsis followed by a new sentence.

    When text like "He was tired.... The next day" is processed, spaCy may not
    recognize the sentence boundary after the ellipsis. This function splits
    such cases by detecting ellipsis (3+ dots or ". . .") followed by whitespace
    and a capital letter (optionally preceded by quotes).

    Args:
        sentences: List of sentences from spaCy

    Returns:
        List of sentences with ellipsis boundaries properly handled
    """
    if not sentences:
        return sentences

    # Split sentences containing ellipsis followed by capital letter
    result: list[str] = []
    for sent in sentences:
        # Check if sentence contains ellipsis followed by capital letter
        match = _ELLIPSIS_SENTENCE_BREAK.search(sent)
        if not match:
            result.append(sent)
            continue

        # Split at the boundary (keep ellipsis with first part)
        # We need to handle multiple potential splits in one sentence
        remaining = sent
        while True:
            match = _ELLIPSIS_SENTENCE_BREAK.search(remaining)
            if not match:
                if remaining.strip():
                    result.append(remaining.strip())
                break

            # Split: everything up to and including ellipsis goes to first part
            # The capital letter starts the second part
            split_pos = match.end(1)  # End of ellipsis
            first_part = remaining[:split_pos].strip()
            remaining = remaining[split_pos:].strip()

            if first_part:
                result.append(first_part)

    return result


def _apply_corrections(
    sentences: list[str],
    language_model: str = "en_core_web_sm",
    split_on_colon: bool = True,
    nlp: Language | None = None,
) -> list[str]:
    """
    Apply post-processing corrections to fix common spaCy segmentation errors.

    Corrections applied (in order):
    1. Merge sentences incorrectly split after abbreviations (reduces count)
    2. Split sentences after ellipsis followed by capital letter (increases count)
    3. Split sentences containing multiple URLs (increases count)

    Note: Colon handling is minimal - we let spaCy handle colons naturally.
    The split_on_colon parameter is kept for API compatibility but currently
    has no effect (spaCy's default colon behavior is used).

    Args:
        sentences: List of sentences from spaCy
        language_model: spaCy language model name (for language-specific corrections)
        split_on_colon: Kept for API compatibility (currently unused)
        nlp: Optional spaCy language model (currently unused)

    Returns:
        Corrected list of sentences
    """
    # First merge abbreviation splits (need to combine before other splits)
    sentences = _merge_abbreviation_splits(sentences, language_model)

    # Split after ellipsis followed by new sentence
    sentences = _split_after_ellipsis(sentences)

    # Split URLs (increases sentence count)
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


def _extract_sentences(doc) -> list[str]:
    """Extract sentences from a spaCy Doc object.

    Args:
        doc: A spaCy Doc object

    Returns:
        List of sentence strings (stripped, non-empty)
    """
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _process_long_text(
    text: str,
    nlp: Language,
    max_chunk: int = _DEFAULT_MAX_CHUNK_SIZE,
    safety_margin: int = _DEFAULT_SAFETY_MARGIN,
) -> list[str]:
    """Process text that may exceed spaCy's max_length incrementally.

    Uses index-based tracking to extract sentences from long text without
    cutting sentences at chunk boundaries.

    Args:
        text: Input text (should be preprocessed, ellipsis protected)
        nlp: spaCy Language model
        max_chunk: Maximum characters to process at once
        safety_margin: Buffer at chunk end to avoid cutting sentences

    Returns:
        List of sentence strings (stripped, non-empty)
    """
    # Cap max_chunk to spaCy's limit minus safety margin
    effective_max = min(max_chunk, nlp.max_length - safety_margin)

    if len(text) <= effective_max:
        doc = nlp(text)
        return _extract_sentences(doc)

    sentences: list[str] = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + effective_max, len(text))
        chunk = text[start_idx:end_idx]
        doc = nlp(chunk)

        if end_idx >= len(text):
            # Last chunk - take all sentences
            sentences.extend(_extract_sentences(doc))
            break

        # Not the last chunk - keep only complete sentences
        last_complete_end = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text and sent.end_char < len(chunk) - safety_margin:
                sentences.append(sent_text)
                last_complete_end = sent.end_char

        # Move start index forward
        if last_complete_end > 0:
            start_idx += last_complete_end
        else:
            # No sentence boundary found - take all and move on
            sentences.extend(_extract_sentences(doc))
            start_idx = end_idx

        # Skip leading whitespace for next iteration
        while start_idx < len(text) and text[start_idx] in " \t\n\r":
            start_idx += 1

    return sentences


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
    split_on_colon: bool = True,
) -> list[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text: Input text
        language_model: spaCy language model to use
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling).
            Default is True.
        split_on_colon: Kept for API compatibility (currently unused).
            spaCy's default colon behavior is used. Default is True.

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

        # Process paragraph into sentences (handles long text)
        sentences = _process_long_text(para, nlp)

        for sent in sentences:
            # Restore ellipsis in the sentence
            sent = _restore_ellipsis(sent)
            result.append(sent)

    # Apply post-processing corrections if enabled
    if apply_corrections:
        result = _apply_corrections(result, language_model, split_on_colon, nlp)

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

        # Process paragraph into sentences (handles long text)
        sentences = _process_long_text(para, nlp)

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

    # Split into sentences (handles long text)
    sentences = _process_long_text(protected_text, nlp)

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


def split_text(
    text: str,
    mode: str = "sentence",
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
    split_on_colon: bool = True,
) -> list[Segment]:
    """
    Split text into segments with hierarchical position information.

    This function provides a unified interface for text splitting with different
    granularity levels, while preserving paragraph and sentence structure information.
    Useful for audiobook generation where different pause lengths are needed
    between paragraphs vs. sentences vs. clauses.

    Args:
        text: Input text to split
        mode: Splitting mode - one of:
            - "paragraph": Split into paragraphs only
            - "sentence": Split into sentences, grouped by paragraph
            - "clause": Split into clauses (comma-separated), with paragraph
              and sentence info
        language_model: spaCy language model to use (for sentence/clause modes)
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling).
            Default is True. Only applies to sentence/clause modes.
        split_on_colon: Kept for API compatibility (currently unused).
            spaCy's default colon behavior is used. Default is True.

    Returns:
        List of Segment namedtuples, each containing:
            - text: The segment text
            - paragraph: Paragraph index (0-based)
            - sentence: Sentence index within paragraph (0-based).
              None for paragraph mode.

    Raises:
        ValueError: If mode is not one of "paragraph", "sentence", "clause"

    Example:
        >>> segments = split_text("Hello world. How are you?\\n\\nNew paragraph.")
        >>> for seg in segments:
        ...     print(f"P{seg.paragraph} S{seg.sentence}: {seg.text}")
        P0 S0: Hello world.
        P0 S1: How are you?
        P1 S0: New paragraph.

        >>> # Detect paragraph changes for longer pauses
        >>> for i, seg in enumerate(segments):
        ...     if i > 0 and seg.paragraph != segments[i-1].paragraph:
        ...         print("--- paragraph break ---")
        ...     print(seg.text)
    """
    valid_modes = ("paragraph", "sentence", "clause")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[Segment] = []

    if mode == "paragraph":
        for para_idx, para in enumerate(paragraphs):
            result.append(Segment(text=para, paragraph=para_idx, sentence=None))
        return result

    # For sentence and clause modes, we need spaCy
    nlp = _get_nlp(language_model)

    for para_idx, para in enumerate(paragraphs):
        # Protect ellipsis from being treated as sentence boundaries
        protected_para = _protect_ellipsis(para)

        # Process paragraph into sentences (handles long text)
        sentences = _process_long_text(protected_para, nlp)

        # Restore ellipsis in sentences
        sentences = [_restore_ellipsis(sent) for sent in sentences]

        # Apply post-processing corrections if enabled
        if apply_corrections:
            sentences = _apply_corrections(
                sentences, language_model, split_on_colon, nlp
            )

        if mode == "sentence":
            for sent_idx, sent in enumerate(sentences):
                result.append(Segment(text=sent, paragraph=para_idx, sentence=sent_idx))

        elif mode == "clause":
            for sent_idx, sent in enumerate(sentences):
                clauses = _split_sentence_into_clauses(sent)
                for clause in clauses:
                    result.append(
                        Segment(text=clause, paragraph=para_idx, sentence=sent_idx)
                    )

    return result
