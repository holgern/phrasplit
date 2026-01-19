"""Tests for offset-preserving segmentation."""

import pytest

from phrasplit import (
    COMMON_PATTERNS,
    SplitSegment,
    iter_split_with_offsets,
    split_with_offsets,
    suggest_splitting_mode,
    validate_no_placeholder_breaks,
)


class TestSplitWithOffsets:
    """Tests for split_with_offsets function."""

    def test_basic_sentence_offsets(self) -> None:
        """Test that offsets correctly map to original text."""
        text = "Hello world. How are you?"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert len(segments) == 2

        # Verify offsets map correctly
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

        # Check specific values
        assert segments[0].text == "Hello world."
        assert segments[0].char_start == 0
        assert segments[1].text == "How are you?"

    def test_paragraph_mode_offsets(self) -> None:
        """Test paragraph mode with offsets."""
        text = "First paragraph.\n\nSecond paragraph."
        segments = split_with_offsets(text, mode="paragraph", use_spacy=False)

        assert len(segments) == 2

        # Verify offsets
        for seg in segments:
            # Note: paragraph text is stripped, so we check containment
            assert seg.text in text
            assert seg.paragraph_idx in (0, 1)
            assert seg.sentence_idx == 0
            assert seg.clause_idx is None

    def test_clause_mode_offsets(self) -> None:
        """Test clause mode with offsets."""
        text = "I like coffee, and I like tea."
        segments = split_with_offsets(text, mode="clause", use_spacy=False)

        # Should split on comma
        assert len(segments) >= 2

        # Verify offsets
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_multiline_text_offsets(self) -> None:
        """Test offsets with multiple paragraphs."""
        text = "Para 1 sent 1. Para 1 sent 2.\n\nPara 2 sent 1."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Verify all offsets
        for seg in segments:
            extracted = text[seg.char_start : seg.char_end]
            # Handle whitespace differences
            assert seg.text.strip() == extracted.strip()

    def test_whitespace_preservation(self) -> None:
        """Test that offsets preserve exact whitespace."""
        text = "  Hello world.  "
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert len(segments) == 1
        # Text should be the trimmed sentence, but offsets should be correct
        seg = segments[0]
        assert text[seg.char_start : seg.char_end].strip() == seg.text.strip()

    def test_punctuation_edge_cases(self) -> None:
        """Test offset accuracy with various punctuation."""
        text = 'She said, "Hello!" Then she left.'
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Verify offsets
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_abbreviations_offsets(self) -> None:
        """Test that abbreviations don't break offset calculation."""
        text = "Dr. Smith works at U.S.A. Inc. She is busy."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Should be 2 sentences
        assert len(segments) == 2

        # Verify offsets
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        segments = split_with_offsets("", mode="sentence", use_spacy=False)
        assert len(segments) == 0

        segments = split_with_offsets("   ", mode="sentence", use_spacy=False)
        assert len(segments) == 0

    def test_max_chars_splitting(self) -> None:
        """Test max_chars parameter splits long segments."""
        # Create a long sentence
        text = "word " * 50  # 250 characters
        segments = split_with_offsets(
            text, mode="sentence", use_spacy=False, max_chars=50
        )

        # Should be split into multiple segments
        assert len(segments) > 1

        # All segments should be <= max_chars
        for seg in segments:
            assert len(seg.text) <= 50

        # Verify offsets still work
        for seg in segments:
            extracted = text[seg.char_start : seg.char_end]
            assert seg.text.strip() == extracted.strip()

    def test_max_chars_respects_boundaries(self) -> None:
        """Test that max_chars splitting prefers word boundaries."""
        text = (
            "This is a very long sentence with many words "
            "that exceeds the maximum character limit."
        )
        segments = split_with_offsets(
            text, mode="sentence", use_spacy=False, max_chars=30
        )

        # Check that we don't have split words (no segment starts/ends mid-word)
        for seg in segments:
            # Segments should start and end with non-space or be at text boundaries
            if seg.char_start > 0:
                assert (
                    text[seg.char_start - 1].isspace()
                    or text[seg.char_start - 1] in ".,;!?"
                )

    def test_offsets_with_spacy(self) -> None:
        """Test offset accuracy with spaCy backend."""
        pytest.importorskip("spacy")

        text = "Hello world. How are you?\n\nNew paragraph."
        segments = split_with_offsets(text, mode="sentence", use_spacy=True)

        # Verify offsets
        for seg in segments:
            # Handle potential whitespace differences
            extracted = text[seg.char_start : seg.char_end]
            assert seg.text.strip() == extracted.strip()


class TestStableIDs:
    """Tests for stable ID generation."""

    def test_id_format_sentence(self) -> None:
        """Test ID format for sentence mode."""
        text = "Sentence one. Sentence two.\n\nParagraph two."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert segments[0].id == "p0s0"
        assert segments[1].id == "p0s1"
        assert segments[2].id == "p1s0"

    def test_id_format_clause(self) -> None:
        """Test ID format for clause mode."""
        text = "I like coffee, and tea."
        segments = split_with_offsets(text, mode="clause", use_spacy=False)

        # Should have clause indices
        for seg in segments:
            assert "c" in seg.id
            assert seg.clause_idx is not None

    def test_id_format_paragraph(self) -> None:
        """Test ID format for paragraph mode."""
        text = "Para 1.\n\nPara 2."
        segments = split_with_offsets(text, mode="paragraph", use_spacy=False)

        assert segments[0].id == "p0s0"
        assert segments[1].id == "p1s0"

    def test_hierarchical_indices(self) -> None:
        """Test that hierarchical indices are correct."""
        text = "P0 S0. P0 S1.\n\nP1 S0. P1 S1."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Check paragraph indices
        assert segments[0].paragraph_idx == 0
        assert segments[1].paragraph_idx == 0
        assert segments[2].paragraph_idx == 1
        assert segments[3].paragraph_idx == 1

        # Check sentence indices
        assert segments[0].sentence_idx == 0
        assert segments[1].sentence_idx == 1
        assert segments[2].sentence_idx == 0
        assert segments[3].sentence_idx == 1


class TestIterSplitWithOffsets:
    """Tests for streaming iterator API."""

    def test_iterator_yields_segments(self) -> None:
        """Test that iterator yields segments in order."""
        text = "First. Second. Third."
        segments_list = list(
            iter_split_with_offsets(text, mode="sentence", use_spacy=False)
        )

        assert len(segments_list) == 3
        assert all(isinstance(seg, SplitSegment) for seg in segments_list)

    def test_iterator_order(self) -> None:
        """Test that iterator yields in document order."""
        text = "S1. S2. S3."
        segments = list(iter_split_with_offsets(text, mode="sentence", use_spacy=False))

        # Check order by char_start
        for i in range(len(segments) - 1):
            assert segments[i].char_start < segments[i + 1].char_start

    def test_iterator_matches_list(self) -> None:
        """Test that iterator produces same results as list version."""
        text = "Test sentence one. Test sentence two.\n\nNew paragraph."

        list_segments = split_with_offsets(text, mode="sentence", use_spacy=False)
        iter_segments = list(
            iter_split_with_offsets(text, mode="sentence", use_spacy=False)
        )

        assert len(list_segments) == len(iter_segments)
        for ls, its in zip(list_segments, iter_segments, strict=False):
            assert ls.id == its.id
            assert ls.text == its.text
            assert ls.char_start == its.char_start
            assert ls.char_end == its.char_end


class TestSplitSegmentDataclass:
    """Tests for SplitSegment dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        seg = SplitSegment(
            id="p0s1",
            text="Test",
            char_start=5,
            char_end=9,
            paragraph_idx=0,
            sentence_idx=1,
            clause_idx=None,
            meta={"method": "spacy"},
        )

        d = seg.to_dict()
        assert d["id"] == "p0s1"
        assert d["text"] == "Test"
        assert d["char_start"] == 5
        assert d["char_end"] == 9
        assert d["paragraph_idx"] == 0
        assert d["sentence_idx"] == 1
        assert d["clause_idx"] is None
        assert d["meta"] == {"method": "spacy"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {
            "id": "p0s1c2",
            "text": "Test text",
            "char_start": 10,
            "char_end": 19,
            "paragraph_idx": 0,
            "sentence_idx": 1,
            "clause_idx": 2,
            "meta": {"custom": "value"},
        }

        seg = SplitSegment.from_dict(d)
        assert seg.id == "p0s1c2"
        assert seg.text == "Test text"
        assert seg.char_start == 10
        assert seg.char_end == 19
        assert seg.paragraph_idx == 0
        assert seg.sentence_idx == 1
        assert seg.clause_idx == 2
        assert seg.meta == {"custom": "value"}

    def test_validation_negative_start(self) -> None:
        """Test validation of negative char_start."""
        with pytest.raises(ValueError, match="char_start must be >= 0"):
            SplitSegment(
                id="p0s0",
                text="Test",
                char_start=-1,
                char_end=4,
                paragraph_idx=0,
                sentence_idx=0,
            )

    def test_validation_end_before_start(self) -> None:
        """Test validation of char_end < char_start."""
        with pytest.raises(ValueError, match="char_end .* must be >= char_start"):
            SplitSegment(
                id="p0s0",
                text="Test",
                char_start=10,
                char_end=5,
                paragraph_idx=0,
                sentence_idx=0,
            )


class TestPlaceholderValidation:
    """Tests for placeholder/markup validation utilities."""

    def test_no_breaks_detected(self) -> None:
        """Test that intact placeholders produce no warnings."""
        text = "Hello {{tag}}. Another sentence."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["mustache"]
        )
        assert len(warnings) == 0

    def test_placeholder_break_detected(self) -> None:
        """Test detection of broken placeholders."""
        # Manually create segments that split a placeholder
        text = "Hello {{placeholder}} world."
        # Create segments that would split the placeholder
        from phrasplit import SplitSegment

        segments = [
            SplitSegment(
                id="p0s0",
                text="Hello {{place",
                char_start=0,
                char_end=13,
                paragraph_idx=0,
                sentence_idx=0,
            ),
            SplitSegment(
                id="p0s1",
                text="holder}} world.",
                char_start=13,
                char_end=28,
                paragraph_idx=0,
                sentence_idx=1,
            ),
        ]

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["mustache"]
        )
        assert len(warnings) > 0
        assert "split across segments" in warnings[0]

    def test_custom_placeholder_pattern(self) -> None:
        """Test validation with custom placeholder pattern."""
        text = "Hello <TAG>world</TAG>. Another sentence."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Custom pattern for XML-style tags
        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=r"<[^>]+>[^<]*</[^>]+>"
        )
        assert len(warnings) == 0

    def test_ssmd_pattern(self) -> None:
        """Test validation with SSMD pattern."""
        text = "Hello [world]{lang='de'}. Another [sentence]{lang='en'}."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["ssmd"]
        )
        assert len(warnings) == 0

    def test_speechmarkdown_pattern(self) -> None:
        """Test validation with Speech Markdown pattern."""
        text = "Hello ((world)[rate:'slow';]). How are you?"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["speechmarkdown"]
        )
        assert len(warnings) == 0

    def test_invalid_pattern_raises_error(self) -> None:
        """Test that invalid regex pattern raises ValueError."""
        text = "Hello world."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            validate_no_placeholder_breaks(
                text, segments, placeholder_pattern=r"[invalid(regex"
            )

    def test_suggest_splitting_mode(self) -> None:
        """Test splitting mode suggestion."""
        # Few placeholders - should suggest clause mode
        text = "Short {{tag}}. Another {{tag}}."
        mode = suggest_splitting_mode(
            text, placeholder_pattern=COMMON_PATTERNS["mustache"]
        )
        assert mode in ("paragraph", "sentence", "clause")

    def test_suggest_splitting_mode_invalid_pattern(self) -> None:
        """Test that suggest_splitting_mode validates regex pattern."""
        text = "Hello world."

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            suggest_splitting_mode(text, placeholder_pattern=r"[invalid(regex")
