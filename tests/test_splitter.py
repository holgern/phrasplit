"""Tests for phrasplit splitter module."""

import pytest

from phrasplit import split_clauses, split_long_lines, split_paragraphs, split_sentences
from phrasplit.splitter import (
    _hard_split,
    _protect_ellipsis,
    _restore_ellipsis,
    _split_at_clauses,
    _split_sentence_into_clauses,
)


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_basic_sentences(self) -> None:
        """Test splitting of regular sentences with proper punctuation."""
        text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
        expected = ["Dr. Smith is here.", "She has a Ph.D. in Chemistry."]
        assert split_sentences(text) == expected

    def test_ellipses_handling(self) -> None:
        """Test handling of ellipses in sentence splitting.

        Note: spaCy doesn't split after ellipsis unless followed by
        sentence-ending punctuation. Ellipses are restored as '. . .'
        (spaced) after processing.
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
        """Check titles and initials are handled without breaking sentence."""
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
    """Tests for split_clauses function - splits at commas for audiobook creation."""

    def test_basic_clauses(self) -> None:
        """Test splitting at commas."""
        text = "I like coffee, and I like tea."
        expected = ["I like coffee,", "and I like tea."]
        assert split_clauses(text) == expected

    def test_semicolon_no_split(self) -> None:
        """Test that semicolons do not cause splits."""
        text = "First clause; second clause."
        expected = ["First clause; second clause."]
        assert split_clauses(text) == expected

    def test_colon_no_split(self) -> None:
        """Test that colons do not cause splits."""
        text = "Here is the list: apples and oranges."
        expected = ["Here is the list: apples and oranges."]
        assert split_clauses(text) == expected

    def test_multiple_commas(self) -> None:
        """Test splitting with multiple commas."""
        text = "First, second, third, fourth."
        expected = ["First,", "second,", "third,", "fourth."]
        assert split_clauses(text) == expected

    def test_no_commas(self) -> None:
        """Test text without commas."""
        text = "This is a simple sentence."
        expected = ["This is a simple sentence."]
        assert split_clauses(text) == expected

    def test_empty_text(self) -> None:
        """Test empty input returns empty list."""
        assert split_clauses("") == []

    def test_em_dash_no_split(self) -> None:
        """Test that em dashes (—) do not cause splits."""
        text = "She was happy— he was not."
        expected = ["She was happy— he was not."]
        assert split_clauses(text) == expected

    def test_en_dash_no_split(self) -> None:
        """Test that en dashes (–) do not cause splits."""
        text = "The years 2020–2023 were difficult– we survived."
        expected = ["The years 2020–2023 were difficult– we survived."]
        assert split_clauses(text) == expected

    def test_complex_sentence_with_multiple_commas(self) -> None:
        """Test splitting a sentence with multiple comma-separated items."""
        text = "Apples, oranges, bananas, and grapes are fruits."
        expected = ["Apples,", "oranges,", "bananas,", "and grapes are fruits."]
        assert split_clauses(text) == expected

    def test_sentence_with_commas_and_other_punctuation(self) -> None:
        """Test sentence with commas and other punctuation (only splits at commas)."""
        text = "When I arrived, he said: 'Welcome home'; then we celebrated."
        expected = ["When I arrived,", "he said: 'Welcome home'; then we celebrated."]
        assert split_clauses(text) == expected

    def test_colon_with_comma_list(self) -> None:
        """Test colon introducing a list with commas (splits only at commas)."""
        text = "Buy these items: milk, bread, eggs."
        expected = ["Buy these items: milk,", "bread,", "eggs."]
        assert split_clauses(text) == expected

    def test_semicolon_with_commas(self) -> None:
        """Test semicolon with surrounding commas (splits only at commas)."""
        text = "The sun was setting, beautifully; the sky turned orange."
        expected = ["The sun was setting,", "beautifully; the sky turned orange."]
        assert split_clauses(text) == expected

    def test_comma_with_coordinating_conjunction(self) -> None:
        """Test comma before coordinating conjunctions (FANBOYS)."""
        text = "I wanted to go, but it was raining."
        expected = ["I wanted to go,", "but it was raining."]
        assert split_clauses(text) == expected

    def test_introductory_clause_with_comma(self) -> None:
        """Test introductory clause followed by main clause."""
        text = "After the meeting ended, everyone went home."
        expected = ["After the meeting ended,", "everyone went home."]
        assert split_clauses(text) == expected

    def test_appositive_with_commas(self) -> None:
        """Test appositive phrase set off by commas."""
        text = "My friend, a talented artist, won the competition."
        expected = ["My friend,", "a talented artist,", "won the competition."]
        assert split_clauses(text) == expected

    def test_mixed_punctuation_only_comma_splits(self) -> None:
        """Test complex sentence - only commas cause splits."""
        text = "First, I woke up; then, I made coffee: black, no sugar."
        expected = ["First,", "I woke up; then,", "I made coffee: black,", "no sugar."]
        assert split_clauses(text) == expected

    def test_quotes_with_comma(self) -> None:
        """Test handling of quoted text with commas.

        Note: Comma inside quotes like '"Hello,"' is not followed by space
        directly (the quote closes first), so spaCy treats it as one token.
        """
        text = '"Hello," she said, "how are you?"'
        expected = ['"Hello," she said,', '"how are you?"']
        assert split_clauses(text) == expected

    def test_direct_speech_with_comma(self) -> None:
        """Test direct speech attribution with comma.

        Note: Comma inside quotes like '"I am here,"' is part of the quoted
        text, so no split occurs there.
        """
        text = '"I am here," said John.'
        expected = ['"I am here," said John.']
        assert split_clauses(text) == expected

    def test_comma_outside_quotes(self) -> None:
        """Test comma outside quotes causes split."""
        text = 'He said "hello", then left.'
        expected = ['He said "hello",', "then left."]
        assert split_clauses(text) == expected

    def test_serial_comma_oxford(self) -> None:
        """Test sentence with Oxford/serial comma."""
        text = "We invited John, Mary, and Tom to the party."
        expected = ["We invited John,", "Mary,", "and Tom to the party."]
        assert split_clauses(text) == expected

    def test_parenthetical_with_commas(self) -> None:
        """Test parenthetical expression set off by commas."""
        text = "The book, which was published last year, became a bestseller."
        expected = [
            "The book,",
            "which was published last year,",
            "became a bestseller.",
        ]
        assert split_clauses(text) == expected

    def test_address_with_commas(self) -> None:
        """Test address or location with commas."""
        text = "She lives in Paris, France, near the Eiffel Tower."
        expected = ["She lives in Paris,", "France,", "near the Eiffel Tower."]
        assert split_clauses(text) == expected

    def test_date_with_commas(self) -> None:
        """Test date format with commas."""
        text = "On July 4, 1776, the Declaration was signed."
        expected = ["On July 4,", "1776,", "the Declaration was signed."]
        assert split_clauses(text) == expected

    def test_however_with_commas(self) -> None:
        """Test conjunctive adverb with commas."""
        text = "The weather was bad, however, we went outside."
        expected = ["The weather was bad,", "however,", "we went outside."]
        assert split_clauses(text) == expected

    def test_comma_after_interjection(self) -> None:
        """Test comma after interjection."""
        text = "Well, that was unexpected."
        expected = ["Well,", "that was unexpected."]
        assert split_clauses(text) == expected

    def test_compound_sentence_with_comma(self) -> None:
        """Test compound sentence joined by comma and conjunction."""
        text = "The cat slept, and the dog played outside."
        expected = ["The cat slept,", "and the dog played outside."]
        assert split_clauses(text) == expected


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
        text = (
            "This is a very long sentence with many clauses, "
            "and it continues here, and it goes on further."
        )
        result = split_long_lines(text, max_length=50)
        assert len(result) >= 2

    def test_max_length_validation_zero(self) -> None:
        """Test that max_length=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be at least 1"):
            split_long_lines("Some text", max_length=0)

    def test_max_length_validation_negative(self) -> None:
        """Test that negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be at least 1"):
            split_long_lines("Some text", max_length=-5)

    def test_max_length_one(self) -> None:
        """Test max_length=1 works (words kept intact)."""
        text = "a b c"
        result = split_long_lines(text, max_length=1)
        assert result == ["a", "b", "c"]

    def test_empty_line_preserved(self) -> None:
        """Test that empty lines in input are preserved."""
        text = "First line.\n\nThird line."
        result = split_long_lines(text, max_length=80)
        assert result == ["First line.", "", "Third line."]


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


class TestEllipsisHandling:
    """Tests for ellipsis protection and restoration functions."""

    def test_protect_regular_ellipsis(self) -> None:
        """Test protection of regular three-dot ellipsis."""
        text = "Hello... world"
        result = _protect_ellipsis(text)
        assert "..." not in result
        assert "\u2026" in result

    def test_protect_long_ellipsis(self) -> None:
        """Test protection of ellipsis with more than 3 dots."""
        text = "Hello..... world"
        result = _protect_ellipsis(text)
        assert "....." not in result
        assert "\u2026" in result

    def test_protect_spaced_ellipsis(self) -> None:
        """Test protection of spaced ellipsis (. . .)."""
        text = "Hello. . . world"
        result = _protect_ellipsis(text)
        assert ". . ." not in result
        assert "\u2026" in result

    def test_protect_unicode_ellipsis_unchanged(self) -> None:
        """Test that unicode ellipsis is already the placeholder."""
        text = "Hello\u2026 world"
        result = _protect_ellipsis(text)
        assert result == text  # No change, already placeholder

    def test_restore_ellipsis(self) -> None:
        """Test restoration of ellipsis placeholder to spaced format."""
        text = "Hello\u2026 world"
        result = _restore_ellipsis(text)
        assert result == "Hello. . . world"

    def test_protect_and_restore_roundtrip(self) -> None:
        """Test that protect then restore gives consistent output."""
        original = "Wait... what?"
        protected = _protect_ellipsis(original)
        restored = _restore_ellipsis(protected)
        assert restored == "Wait. . . what?"

    def test_multiple_ellipses(self) -> None:
        """Test handling multiple ellipses in same text."""
        text = "One... Two... Three..."
        protected = _protect_ellipsis(text)
        restored = _restore_ellipsis(protected)
        assert restored == "One. . . Two. . . Three. . ."


class TestHardSplit:
    """Tests for _hard_split internal function."""

    def test_hard_split_basic(self) -> None:
        """Test basic word splitting."""
        text = "one two three four"
        result = _hard_split(text, max_length=10)
        assert result == ["one two", "three four"]

    def test_hard_split_exact_fit(self) -> None:
        """Test words that fit exactly."""
        text = "ab cd"
        result = _hard_split(text, max_length=5)
        assert result == ["ab cd"]

    def test_hard_split_single_word_too_long(self) -> None:
        """Test single word exceeding max_length is kept intact."""
        text = "superlongword"
        result = _hard_split(text, max_length=5)
        assert result == ["superlongword"]

    def test_hard_split_empty_string(self) -> None:
        """Test empty string returns original."""
        result = _hard_split("", max_length=10)
        assert result == [""]

    def test_hard_split_whitespace_only(self) -> None:
        """Test whitespace-only string returns original."""
        result = _hard_split("   ", max_length=10)
        assert result == ["   "]

    def test_hard_split_single_word(self) -> None:
        """Test single word returns that word."""
        result = _hard_split("hello", max_length=10)
        assert result == ["hello"]


class TestSplitAtClauses:
    """Tests for _split_at_clauses internal function."""

    def test_split_at_clauses_basic(self) -> None:
        """Test basic clause splitting."""
        text = "First part, second part, third part."
        result = _split_at_clauses(text, max_length=30)
        assert len(result) >= 2

    def test_split_at_clauses_no_commas(self) -> None:
        """Test text without commas."""
        text = "Just a single clause without commas."
        result = _split_at_clauses(text, max_length=50)
        assert result == ["Just a single clause without commas."]

    def test_split_at_clauses_falls_back_to_hard_split(self) -> None:
        """Test fallback to hard split when clauses still too long."""
        text = "This is a very very very long clause without commas"
        result = _split_at_clauses(text, max_length=15)
        # Should be hard-split at word boundaries
        assert len(result) >= 3
        for line in result:
            # Each line should be <= max_length unless it's a single word
            assert len(line) <= 15 or len(line.split()) == 1


class TestSplitSentenceIntoClauses:
    """Tests for _split_sentence_into_clauses internal function."""

    def test_basic_split(self) -> None:
        """Test basic comma splitting."""
        sentence = "First, second, third."
        result = _split_sentence_into_clauses(sentence)
        assert result == ["First,", "second,", "third."]

    def test_no_commas(self) -> None:
        """Test sentence without commas."""
        sentence = "No commas here."
        result = _split_sentence_into_clauses(sentence)
        assert result == ["No commas here."]

    def test_empty_sentence(self) -> None:
        """Test empty sentence."""
        result = _split_sentence_into_clauses("")
        assert result == [""]

    def test_comma_at_end(self) -> None:
        """Test comma stays with preceding text."""
        sentence = "Hello, world."
        result = _split_sentence_into_clauses(sentence)
        assert result == ["Hello,", "world."]


class TestErrorConditions:
    """Tests for error handling and edge cases."""

    def test_split_long_lines_invalid_max_length_message(self) -> None:
        """Test error message includes the invalid value."""
        with pytest.raises(ValueError) as exc_info:
            split_long_lines("text", max_length=-10)
        assert "-10" in str(exc_info.value)

    def test_whitespace_only_paragraphs(self) -> None:
        """Test paragraphs that are only whitespace."""
        text = "   \n\n   \n\n   "
        assert split_paragraphs(text) == []

    def test_single_character_text(self) -> None:
        """Test single character text."""
        assert split_sentences("a") == ["a"]
        assert split_paragraphs("a") == ["a"]

    def test_only_punctuation(self) -> None:
        """Test text that is only punctuation."""
        result = split_sentences("...")
        # Should handle gracefully (result may vary based on spaCy)
        assert isinstance(result, list)

    def test_numeric_text(self) -> None:
        """Test handling of numeric text with periods."""
        text = "Version 3.14.159 is released. Update now."
        result = split_sentences(text)
        assert len(result) == 2

    def test_multiple_spaces(self) -> None:
        """Test text with multiple consecutive spaces."""
        text = "Hello    world.   Another    sentence."
        result = split_sentences(text)
        assert len(result) == 2

    def test_tabs_and_mixed_whitespace(self) -> None:
        """Test text with tabs and mixed whitespace."""
        text = "First paragraph.\n\t\n\nSecond paragraph."
        result = split_paragraphs(text)
        assert len(result) == 2
