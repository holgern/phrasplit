"""Tests for phrasplit splitter module."""

import pytest

from phrasplit import split_clauses, split_long_lines, split_paragraphs, split_sentences


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_basic_sentences(self) -> None:
        """Test splitting of regular sentences with proper punctuation."""
        text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
        expected = ["Dr. Smith is here.", "She has a Ph.D. in Chemistry."]
        assert split_sentences(text) == expected

    def test_ellipses_handling(self) -> None:
        """Test handling of ellipses in sentence splitting.

        Note: spaCy doesn't split after ellipsis unless followed by sentence-ending punctuation.
        Ellipses are restored as '. . .' (spaced) after processing.
        """
        text = "Hello... Is it working? Yes... it is!"
        expected = ["Hello. . . Is it working?", "Yes. . . it is!"]
        assert split_sentences(text) == expected

    def test_common_abbreviations(self) -> None:
        """Test abbreviations like Mr., Prof., U.S.A. that shouldn't split sentences."""
        text = "Mr. Brown met Prof. Green. They discussed the U.S.A. case."
        expected = ["Mr. Brown met Prof. Green.", "They discussed the U.S.A. case."]
        assert split_sentences(text) == expected

    def test_acronyms_followed_by_sentences(self) -> None:
        """Test acronyms followed by normal sentences."""
        text = "U.S.A. is big. It has many states."
        expected = ["U.S.A. is big.", "It has many states."]
        assert split_sentences(text) == expected

    def test_website_urls(self) -> None:
        """Ensure website URLs like www.example.com are not split incorrectly."""
        text = "Visit www.example.com. Then send feedback."
        expected = ["Visit www.example.com.", "Then send feedback."]
        assert split_sentences(text) == expected

    def test_initials_and_titles(self) -> None:
        """Check titles and initials are handled gracefully without breaking sentence."""
        text = "Mr. J.R.R. Tolkien wrote many books. They were popular."
        expected = ["Mr. J.R.R. Tolkien wrote many books.", "They were popular."]
        assert split_sentences(text) == expected

    def test_single_letter_abbreviation(self) -> None:
        """Ensure single-letter abbreviations like 'E.' are not split."""
        text = "E. coli is a bacteria. Dr. E. Stone confirmed it."
        expected = ["E. coli is a bacteria.", "Dr. E. Stone confirmed it."]
        assert split_sentences(text) == expected

    def test_quotes_and_dialogue(self) -> None:
        """Test punctuation with quotation marks."""
        text = 'She said, "It works!" Then she smiled.'
        expected = ['She said, "It works!"', "Then she smiled."]
        assert split_sentences(text) == expected

    def test_suffix_abbreviations(self) -> None:
        """Test suffixes like Ltd., Co. don't break sentences prematurely."""
        text = "Smith & Co. Ltd. is closed. We're switching vendors."
        expected = ["Smith & Co. Ltd. is closed.", "We're switching vendors."]
        assert split_sentences(text) == expected

    def test_missing_terminal_punctuation(self) -> None:
        """Handle cases where no punctuation marks end the sentence."""
        text = "This is a sentence without trailing punctuation"
        expected = ["This is a sentence without trailing punctuation"]
        assert split_sentences(text) == expected

    def test_empty_text(self) -> None:
        """Test empty input returns empty list."""
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_multiple_paragraphs(self) -> None:
        """Test sentences across multiple paragraphs."""
        text = "First paragraph. Second sentence.\n\nSecond paragraph. Another one."
        result = split_sentences(text)
        assert len(result) == 4
        assert result[0] == "First paragraph."
        assert result[1] == "Second sentence."
        assert result[2] == "Second paragraph."
        assert result[3] == "Another one."


class TestSplitClauses:
    """Tests for split_clauses function."""

    def test_basic_clauses(self) -> None:
        """Test splitting at commas."""
        text = "I like coffee, and I like tea."
        expected = ["I like coffee,", "and I like tea."]
        assert split_clauses(text) == expected

    def test_semicolon_split(self) -> None:
        """Test splitting at semicolons."""
        text = "First clause; second clause."
        expected = ["First clause;", "second clause."]
        assert split_clauses(text) == expected

    def test_colon_split(self) -> None:
        """Test splitting at colons."""
        text = "Here is the list: apples and oranges."
        expected = ["Here is the list:", "apples and oranges."]
        assert split_clauses(text) == expected

    def test_multiple_delimiters(self) -> None:
        """Test splitting with multiple different delimiters."""
        text = "First, second; third: fourth."
        expected = ["First,", "second;", "third:", "fourth."]
        assert split_clauses(text) == expected

    def test_no_clause_delimiters(self) -> None:
        """Test text without clause delimiters."""
        text = "This is a simple sentence."
        expected = ["This is a simple sentence."]
        assert split_clauses(text) == expected

    def test_empty_text(self) -> None:
        """Test empty input returns empty list."""
        assert split_clauses("") == []


class TestSplitParagraphs:
    """Tests for split_paragraphs function."""

    def test_basic_paragraphs(self) -> None:
        """Test splitting by double newlines."""
        text = "First paragraph.\n\nSecond paragraph."
        expected = ["First paragraph.", "Second paragraph."]
        assert split_paragraphs(text) == expected

    def test_multiple_blank_lines(self) -> None:
        """Test multiple blank lines between paragraphs."""
        text = "First.\n\n\n\nSecond."
        expected = ["First.", "Second."]
        assert split_paragraphs(text) == expected

    def test_whitespace_only_lines(self) -> None:
        """Test blank lines with whitespace."""
        text = "First.\n  \n  \nSecond."
        expected = ["First.", "Second."]
        assert split_paragraphs(text) == expected

    def test_single_paragraph(self) -> None:
        """Test single paragraph without breaks."""
        text = "Single paragraph with no breaks."
        expected = ["Single paragraph with no breaks."]
        assert split_paragraphs(text) == expected

    def test_empty_text(self) -> None:
        """Test empty input returns empty list."""
        assert split_paragraphs("") == []
        assert split_paragraphs("\n\n") == []


class TestSplitLongLines:
    """Tests for split_long_lines function."""

    def test_short_line_unchanged(self) -> None:
        """Test lines under max_length are unchanged."""
        text = "Short line."
        result = split_long_lines(text, max_length=80)
        assert result == ["Short line."]

    def test_long_line_split(self) -> None:
        """Test long lines are split at sentence boundaries."""
        text = "This is a long sentence. This is another sentence that makes it longer."
        result = split_long_lines(text, max_length=30)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 30 or len(line.split()) == 1

    def test_very_long_word(self) -> None:
        """Test handling of words longer than max_length."""
        text = "Supercalifragilisticexpialidocious"
        result = split_long_lines(text, max_length=10)
        # Word is kept intact even if longer than max_length
        assert result == ["Supercalifragilisticexpialidocious"]

    def test_multiple_lines(self) -> None:
        """Test input with existing line breaks."""
        text = "Short line.\nAnother short one."
        result = split_long_lines(text, max_length=80)
        assert result == ["Short line.", "Another short one."]

    def test_clause_splitting_for_long_sentences(self) -> None:
        """Test that long sentences are split at clause boundaries."""
        text = "This is a very long sentence with many clauses, and it continues here, and it goes on further."
        result = split_long_lines(text, max_length=50)
        assert len(result) >= 2


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_unicode_text(self) -> None:
        """Test handling of unicode characters."""
        text = "Hello world. Bonjour le monde. Hallo Welt."
        result = split_sentences(text)
        assert len(result) == 3

    def test_newlines_in_paragraph(self) -> None:
        """Test single newlines within a paragraph."""
        text = "First line\nSecond line\n\nNew paragraph"
        result = split_paragraphs(text)
        assert len(result) == 2

    def test_special_characters(self) -> None:
        """Test text with special characters."""
        text = "Price is $100. Contact us at test@email.com."
        result = split_sentences(text)
        assert len(result) == 2
