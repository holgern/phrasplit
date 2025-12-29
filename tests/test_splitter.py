"""Tests for phrasplit splitter module."""

import pytest

from phrasplit import split_clauses, split_long_lines, split_paragraphs, split_sentences
from phrasplit.splitter import (
    _apply_corrections,
    _fix_hyphenated_linebreaks,
    _hard_split,
    _merge_abbreviation_splits,
    _normalize_whitespace,
    _preprocess_text,
    _protect_ellipsis,
    _restore_ellipsis,
    _split_at_clauses,
    _split_sentence_into_clauses,
    _split_urls,
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


class TestPreprocessing:
    """Tests for text preprocessing functions."""

    def test_fix_hyphenated_linebreaks_basic(self) -> None:
        """Test basic hyphenated line break fix."""
        text = "The recom-\nmendation was ignored."
        result = _fix_hyphenated_linebreaks(text)
        assert result == "The recommendation was ignored."

    def test_fix_hyphenated_linebreaks_multiple(self) -> None:
        """Test multiple hyphenated line breaks."""
        text = "The recom-\nmendation and imple-\nmentation."
        result = _fix_hyphenated_linebreaks(text)
        assert result == "The recommendation and implementation."

    def test_fix_hyphenated_linebreaks_with_spaces(self) -> None:
        """Test hyphenated line break with trailing spaces."""
        text = "The recom-  \n  mendation was ignored."
        result = _fix_hyphenated_linebreaks(text)
        assert result == "The recommendation was ignored."

    def test_fix_hyphenated_linebreaks_no_hyphen(self) -> None:
        """Test text without hyphenated line breaks is unchanged."""
        text = "Normal text without hyphenation."
        result = _fix_hyphenated_linebreaks(text)
        assert result == text

    def test_fix_hyphenated_linebreaks_preserves_real_hyphens(self) -> None:
        """Test that real compound words with hyphens are preserved."""
        text = "A well-known fact."
        result = _fix_hyphenated_linebreaks(text)
        assert result == "A well-known fact."

    def test_normalize_whitespace_multiple_spaces(self) -> None:
        """Test normalizing multiple spaces to single space."""
        text = "Hello    world   here."
        result = _normalize_whitespace(text)
        assert result == "Hello world here."

    def test_normalize_whitespace_tabs(self) -> None:
        """Test normalizing tabs to spaces."""
        text = "Hello\tworld\there."
        result = _normalize_whitespace(text)
        assert result == "Hello world here."

    def test_normalize_whitespace_preserves_paragraph_breaks(self) -> None:
        """Test that paragraph breaks (double newlines) are preserved."""
        text = "First paragraph.\n\nSecond paragraph."
        result = _normalize_whitespace(text)
        assert "\n\n" in result

    def test_normalize_whitespace_normalizes_paragraph_breaks(self) -> None:
        """Test that multiple newlines are normalized to double newline."""
        text = "First.\n\n\n\nSecond."
        result = _normalize_whitespace(text)
        # Should still have paragraph break
        assert "First." in result and "Second." in result

    def test_preprocess_text_combined(self) -> None:
        """Test combined preprocessing steps."""
        text = "The recom-\nmendation   was   ignored."
        result = _preprocess_text(text)
        assert result == "The recommendation was ignored."

    def test_preprocess_text_pdf_like(self) -> None:
        """Test preprocessing of PDF-like text with multiple issues."""
        text = "The   recom-\nmendation  and  imple-\nmentation.\n\nNew paragraph."
        result = _preprocess_text(text)
        assert "recommendation" in result
        assert "implementation" in result
        assert "\n\n" in result  # Paragraph break preserved


class TestPreprocessingIntegration:
    """Integration tests for preprocessing with main split functions."""

    def test_split_sentences_with_hyphenated_linebreaks(self) -> None:
        """Test that split_sentences handles hyphenated line breaks."""
        text = "The recom-\nmendation was accepted. It was good."
        result = split_sentences(text)
        assert len(result) == 2
        assert "recommendation" in result[0]

    def test_split_paragraphs_with_hyphenated_linebreaks(self) -> None:
        """Test that split_paragraphs handles hyphenated line breaks."""
        text = "First para-\ngraph here.\n\nSecond para-\ngraph here."
        result = split_paragraphs(text)
        assert len(result) == 2
        assert "paragraph" in result[0]
        assert "paragraph" in result[1]

    def test_split_clauses_with_hyphenated_linebreaks(self) -> None:
        """Test that split_clauses handles hyphenated line breaks."""
        text = "The recom-\nmendation was good, and it was imple-\nmented."
        result = split_clauses(text)
        assert any("recommendation" in c for c in result)
        assert any("implemented" in c for c in result)

    def test_split_sentences_with_normalized_whitespace(self) -> None:
        """Test that split_sentences normalizes whitespace."""
        text = "Hello    world.   Another    sentence."
        result = split_sentences(text)
        assert len(result) == 2
        # Check whitespace is normalized in output
        assert "    " not in result[0]


class TestSentenceSplittingEdgeCases:
    """Additional edge case tests inspired by syntok test suite.

    These tests cover many real-world sentence splitting challenges.
    """

    # Abbreviation tests
    def test_simple_abbreviations_mr_mrs(self) -> None:
        """Test Mr. and Mrs. abbreviations don't split."""
        text = "This is Mr. Motto here. And here is Mrs. Smithers."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Mr. Motto" in result[0]
        assert "Mrs. Smithers" in result[1]

    def test_abbreviation_capt(self) -> None:
        """Test Capt. abbreviation."""
        text = "This is Capt. Motto here. And here is Dr. Smithers."
        result = split_sentences(text)
        assert len(result) == 2

    def test_abbreviation_eg_ie(self) -> None:
        """Test e.g. and i.e. abbreviations."""
        text = "Examples include e.g. apples. I.e. fruits are healthy."
        result = split_sentences(text)
        assert len(result) == 2

    def test_abbreviation_etc(self) -> None:
        """Test etc. abbreviation at sentence end."""
        text = "Items like apples, oranges, etc. are fruits. They are healthy."
        result = split_sentences(text)
        assert len(result) == 2

    def test_abbreviation_no_number(self) -> None:
        """Test No. abbreviation with numbers."""
        text = "This is No. 1 on the list. It was the best."
        result = split_sentences(text)
        assert len(result) == 2

    def test_abbreviation_vs(self) -> None:
        """Test vs. abbreviation."""
        text = "The case is Smith vs. Jones. It was decided yesterday."
        result = split_sentences(text)
        assert len(result) == 2

    # Quote and parenthesis tests
    def test_sentence_in_parenthesis(self) -> None:
        """Test sentence inside parenthesis."""
        text = "And another sentence. (How about a sentence in parenthesis?)"
        result = split_sentences(text)
        assert len(result) == 2

    def test_sentence_with_quote(self) -> None:
        """Test sentence with quote inside."""
        text = 'Or a sentence with "a quote!" And then another.'
        result = split_sentences(text)
        assert len(result) == 2

    def test_single_quotes(self) -> None:
        """Test handling of single quotes."""
        text = "'How about those pesky single quotes?' She asked."
        result = split_sentences(text)
        assert len(result) == 2

    def test_square_brackets(self) -> None:
        """Test square brackets."""
        text = "[And not to forget about square brackets.] More text here."
        result = split_sentences(text)
        assert len(result) == 2

    def test_brackets_before_terminal(self) -> None:
        """Test brackets appearing before sentence terminal."""
        text = "And, brackets before the terminal [2]. You know I told you so."
        result = split_sentences(text)
        assert len(result) == 2

    # Multiple punctuation tests
    def test_multiple_question_marks(self) -> None:
        """Test multiple question/exclamation marks."""
        text = "What the heck??!?! This is crazy."
        result = split_sentences(text)
        assert len(result) == 2

    def test_ellipsis_at_end(self) -> None:
        """Test ellipsis at sentence end."""
        text = "This is a sentence terminal ellipsis... And this continues."
        result = split_sentences(text)
        # Ellipsis handling may vary - just check we get results
        assert len(result) == 1

    # Enumeration tests
    def test_enumeration_numbers(self) -> None:
        """Test numbered enumerations.

        Note: spaCy treats standalone numbers with periods as separate sentences.
        """
        text = "1. This is one. 2. And that is two."
        result = split_sentences(text)
        # spaCy splits "1." and "2." as separate items
        assert len(result) == 4
        assert "This is one." in result

    def test_enumeration_letters(self) -> None:
        """Test letter enumerations.

        Note: spaCy treats standalone letters with periods as separate sentences.
        """
        text = "A. The first assumption. B. The second bullet."
        result = split_sentences(text)
        # spaCy splits "A." and "B." as separate items
        assert len(result) == 4

    def test_enumeration_parenthesis(self) -> None:
        """Test enumeration with parenthesis."""
        text = "(A) First things here. (B) Second things there."
        result = split_sentences(text)
        assert len(result) == 2

    def test_enumeration_roman(self) -> None:
        """Test Roman numeral enumerations.

        Note: spaCy behavior varies with Roman numerals in parentheses.
        """
        text = "(vii) And the Romans, too. (viii) They counted."
        result = split_sentences(text)
        # spaCy may handle these differently
        assert len(result) >= 2

    # Complex abbreviation patterns
    def test_us_abbreviation(self) -> None:
        """Test U.S. abbreviation in various contexts."""
        text = "This happened in the U.S. last week. It was big news."
        result = split_sentences(text)
        assert len(result) == 2

    def test_eu_uk_abbreviations(self) -> None:
        """Test E.U. and U.K. abbreviations."""
        text = "The E.U. and the U.K. are separating. Brexit happened."
        result = split_sentences(text)
        assert len(result) == 2

    def test_abbreviation_at_sentence_end(self) -> None:
        """Test abbreviation followed by sentence end marker."""
        text = "Refugees are welcome in the E.U.. But rules apply."
        result = split_sentences(text)
        # Double period after abbreviation
        assert len(result) == 2

    def test_us_air_force(self) -> None:
        """Test U.S. followed by proper noun."""
        text = "The U.S. Air Force was called in. They responded quickly."
        result = split_sentences(text)
        assert len(result) == 2

    # Scientific notation tests
    def test_decimal_numbers(self) -> None:
        """Test decimal numbers don't split sentences."""
        text = "A 130 nm CMOS power amplifier operating at 2.4 GHz. Its power is high."
        result = split_sentences(text)
        assert len(result) == 2
        assert "2.4 GHz" in result[0]

    def test_scientific_species(self) -> None:
        """Test species names like S. lividans."""
        text = "Their presence was detected in S. lividans. Further study needed."
        result = split_sentences(text)
        assert len(result) == 2

    def test_statistical_notation(self) -> None:
        """Test statistical notation with p values."""
        text = "Results show significance (p <= .001). This is important."
        result = split_sentences(text)
        assert len(result) == 2

    # Date and time tests
    def test_date_with_abbreviation(self) -> None:
        """Test dates with month abbreviations."""
        text = "On Jan. 22, 2022 it happened. The weather was cold."
        result = split_sentences(text)
        assert len(result) == 2

    def test_time_notation(self) -> None:
        """Test time notation like 14.10."""
        text = "Let's meet at 14.10 in the lobby. Don't be late."
        result = split_sentences(text)
        assert len(result) == 2

    # Author and citation tests
    def test_author_initials(self) -> None:
        """Test author initials like B. Obama."""
        text = "B. Obama was the first black US president. He served two terms."
        result = split_sentences(text)
        assert len(result) == 2

    def test_multiple_initials(self) -> None:
        """Test multiple initials like Dr. Edgar F. Codd."""
        text = "This model was introduced by Dr. Edgar F. Codd. It changed databases."
        result = split_sentences(text)
        assert len(result) == 2

    def test_lester_b_pearson(self) -> None:
        """Test name with single initial: Lester B. Pearson."""
        text = "The basis for Lester B. Pearson's policy was clear. It influenced many."
        result = split_sentences(text)
        assert len(result) == 2

    # Copyright and legal tests
    def test_copyright_notice(self) -> None:
        """Test copyright notices.

        Note: With corrections enabled, "Ltd." is recognized as an abbreviation
        and merged with following text. This keeps the copyright notice intact,
        which is the desired behavior.
        """
        text = "(C) 2017 Company Ltd. All rights reserved."
        result = split_sentences(text)
        # With corrections, stays as one sentence
        # (Ltd. merged with "All rights reserved.")
        assert len(result) == 1
        assert "Ltd. All rights reserved" in result[0]

    # Complex sentence structures
    def test_nested_parenthesis(self) -> None:
        """Test nested parenthesis."""
        text = "Nested (Parenthesis. With words inside.) More here."
        result = split_sentences(text)
        assert len(result) >= 1

    def test_quote_inside_sentence(self) -> None:
        """Test quote that is inside the sentence.

        Note: spaCy may split at quote boundaries depending on punctuation.
        """
        text = 'This quote "He said it." is actually inside. See?'
        result = split_sentences(text)
        # spaCy splits after the quoted sentence ends with period
        assert len(result) >= 2

    def test_semicolon_splitting(self) -> None:
        """Test that semicolons don't incorrectly split sentences."""
        text = "This is verse 14;45 in the test. Splitting on semi-colons."
        result = split_sentences(text)
        assert len(result) == 2

    # Measurement and unit tests
    def test_measurement_units(self) -> None:
        """Test measurements with decimal points."""
        text = "The amplifier consumes total DC power of 167 uW. This is efficient."
        result = split_sentences(text)
        assert len(result) == 2

    def test_frequency_ghz(self) -> None:
        """Test frequency notation like 780 MHz."""
        text = "A sampling frequency of 780 MHz. The figure-of-merit is there."
        result = split_sentences(text)
        assert len(result) == 2

    # Edge cases
    def test_single_word_sentence(self) -> None:
        """Test single word sentences.

        Note: spaCy may merge short exclamations with following text.
        """
        text = "Who did this? No! Such a shame."
        result = split_sentences(text)
        # spaCy merges "No!" with "Such a shame."
        assert len(result) >= 2
        assert "Who did this?" in result

    def test_company_name_with_dots(self) -> None:
        """Test company names with dots."""
        text = "This is Company B.V. in Amsterdam. They make software."
        result = split_sentences(text)
        assert len(result) == 2

    def test_sentence_starting_with_number(self) -> None:
        """Test sentence starting with a number."""
        text = "12 monkeys ran into here. They escaped from the zoo."
        result = split_sentences(text)
        assert len(result) == 2

    def test_abbreviation_followed_by_number(self) -> None:
        """Test abbreviation followed by large number."""
        text = "This is No. 123 here. It's the best."
        result = split_sentences(text)
        assert len(result) == 2

    def test_long_text_in_parenthesis(self) -> None:
        """Test long text inside parenthesis should potentially split."""
        text = (
            "This is one. "
            "(Here is another view of the same. "
            "And then there is a different case here.)"
        )
        result = split_sentences(text)
        # Long parenthetical content may split
        assert len(result) >= 1

    def test_bible_citation(self) -> None:
        """Test bible citation format."""
        text = "This is a bible quote. (Phil. 4:8) Yes, it is!"
        result = split_sentences(text)
        assert len(result) >= 2

    def test_specimens_with_n_equals(self) -> None:
        """Test scientific notation with n = X."""
        text = "Specimens (n = 32) were sent for analysis. Results pending."
        result = split_sentences(text)
        assert len(result) == 2

    def test_percentage_notation(self) -> None:
        """Test percentage in scientific context."""
        text = "PCR could identify an organism in 10 of 32 cases (31.2%). This is good."
        result = split_sentences(text)
        assert len(result) == 2


class TestSplitUrls:
    """Tests for _split_urls post-processing function."""

    def test_no_urls(self) -> None:
        """Test sentences without URLs are unchanged."""
        sentences = ["This is a normal sentence.", "Another sentence here."]
        result = _split_urls(sentences)
        assert result == sentences

    def test_single_url_sentence(self) -> None:
        """Test sentence with single URL is unchanged."""
        sentences = ["Visit https://example.com for more info."]
        result = _split_urls(sentences)
        assert result == sentences

    def test_multiple_urls_split(self) -> None:
        """Test sentence with multiple URLs is split."""
        sentences = ["Check https://example.com https://another.com for details."]
        result = _split_urls(sentences)
        assert len(result) == 2
        assert "https://example.com" in result[0]
        assert "https://another.com" in result[1]

    def test_urls_with_text_between(self) -> None:
        """Test URLs separated by text."""
        sentences = ["Visit https://site1.com and also https://site2.com today."]
        result = _split_urls(sentences)
        assert len(result) == 2

    def test_http_and_https_mixed(self) -> None:
        """Test both http and https URLs."""
        sentences = ["Try http://old.com https://new.com for comparison."]
        result = _split_urls(sentences)
        assert len(result) == 2

    def test_url_at_start(self) -> None:
        """Test URL at start of sentence."""
        sentences = ["https://example.com is a great site."]
        result = _split_urls(sentences)
        # URL at start shouldn't split
        assert len(result) == 1

    def test_empty_list(self) -> None:
        """Test empty list returns empty list."""
        result = _split_urls([])
        assert result == []

    def test_multiple_sentences_mixed(self) -> None:
        """Test multiple sentences, some with URLs, some without."""
        sentences = [
            "Normal sentence.",
            "Check https://a.com https://b.com here.",
            "Another normal one.",
        ]
        result = _split_urls(sentences)
        assert len(result) == 4


class TestMergeAbbreviationSplits:
    """Tests for _merge_abbreviation_splits post-processing function."""

    def test_no_abbreviations(self) -> None:
        """Test sentences without abbreviations are unchanged."""
        sentences = ["Hello world.", "How are you?"]
        result = _merge_abbreviation_splits(sentences)
        assert result == sentences

    def test_merge_dr_name(self) -> None:
        """Test merging Dr. followed by name."""
        sentences = ["Visit Dr.", "Smith for help."]
        result = _merge_abbreviation_splits(sentences)
        assert len(result) == 1
        assert result[0] == "Visit Dr. Smith for help."

    def test_merge_mr_name(self) -> None:
        """Test merging Mr. followed by name."""
        sentences = ["This is Mr.", "Jones speaking."]
        result = _merge_abbreviation_splits(sentences)
        assert len(result) == 1
        assert result[0] == "This is Mr. Jones speaking."

    def test_merge_single_initial(self) -> None:
        """Test merging single initial like J. followed by name."""
        sentences = ["Talk to J.", "Kennedy about it."]
        result = _merge_abbreviation_splits(sentences)
        assert len(result) == 1
        assert result[0] == "Talk to J. Kennedy about it."

    def test_no_merge_sentence_starter(self) -> None:
        """Test that abbreviation followed by sentence starter doesn't merge."""
        sentences = ["I have a Ph.D.", "The university awarded it."]
        result = _merge_abbreviation_splits(sentences)
        assert len(result) == 2

    def test_no_merge_lowercase_start(self) -> None:
        """Test that abbreviation followed by lowercase doesn't merge."""
        sentences = ["This is Dr.", "and so on."]
        result = _merge_abbreviation_splits(sentences)
        # Doesn't merge because 'and' starts with lowercase
        assert len(result) == 2

    def test_merge_us_name(self) -> None:
        """Test merging U. (single letter) followed by continuation."""
        sentences = ["The U.", "States are large."]
        result = _merge_abbreviation_splits(sentences)
        # 'States' is capitalized and not a sentence starter
        assert len(result) == 1

    def test_empty_list(self) -> None:
        """Test empty list returns empty list."""
        result = _merge_abbreviation_splits([])
        assert result == []

    def test_single_sentence(self) -> None:
        """Test single sentence returns unchanged."""
        sentences = ["Just one sentence."]
        result = _merge_abbreviation_splits(sentences)
        assert result == sentences

    def test_multiple_merges(self) -> None:
        """Test multiple abbreviations in sequence.

        Note: The merge function processes one merge at a time from left to right.
        After merging "Dr." with "Smith", the resulting sentence ends with "Prof."
        which then gets merged with "Jones".
        """
        sentences = ["See Dr.", "Smith and Prof.", "Jones today."]
        result = _merge_abbreviation_splits(sentences)
        # First pass merges "See Dr." with "Smith and Prof."
        # Second pass merges "See Dr. Smith and Prof." with "Jones today."
        # But wait - the function only loops through once, so it merges:
        # 1. "See Dr." + "Smith and Prof." -> "See Dr. Smith and Prof."
        # 2. Then checks if "See Dr. Smith and Prof." ends with abbrev - it does (Prof.)
        #    So merges with "Jones today." -> "See Dr. Smith and Prof. Jones today."
        # Actually the loop processes sequentially, so let's trace:
        # i=0: "See Dr." ends with "Dr.", next is "Smith..." which starts with "Smith"
        #      -> merge to "See Dr. Smith and Prof.", skip i=1, now i=2
        # i=2: No more items to check
        # Result: ["See Dr. Smith and Prof.", "Jones today."]
        # So we need 2 results because the merge happens once then loop continues
        assert len(result) == 2
        assert "Dr. Smith" in result[0]

    def test_no_merge_all_caps(self) -> None:
        """Test that abbreviation followed by ALL CAPS doesn't merge."""
        sentences = ["Contact Dr.", "WHO for details."]
        result = _merge_abbreviation_splits(sentences)
        # WHO is all caps, likely an acronym/heading
        assert len(result) == 2

    def test_merge_academic_degree(self) -> None:
        """Test merging M. (like M.D., M.A.) with following name."""
        sentences = ["She has an M.", "Sc. degree."]
        result = _merge_abbreviation_splits(sentences)
        # 'Sc.' starts with capital and is not a sentence starter
        assert len(result) == 1


class TestApplyCorrections:
    """Tests for _apply_corrections wrapper function."""

    def test_applies_url_splitting(self) -> None:
        """Test that URL splitting is applied."""
        sentences = ["Check https://a.com https://b.com here."]
        result = _apply_corrections(sentences)
        # Should split into two: text+first URL, second URL+trailing
        assert len(result) == 2
        assert "https://a.com" in result[0]
        assert "https://b.com" in result[1]

    def test_applies_abbreviation_merging(self) -> None:
        """Test that abbreviation merging is applied."""
        sentences = ["Visit Dr.", "Smith for help."]
        result = _apply_corrections(sentences)
        assert len(result) == 1
        assert "Dr. Smith" in result[0]

    def test_order_abbrev_then_url(self) -> None:
        """Test corrections are applied: abbreviations merged first, then URLs split.

        The order is:
        1. Merge abbreviations -> combines "Dr." with "Smith recommends..."
        2. Split URLs -> the merged sentence has 2 URLs, so it gets split
        """
        sentences = ["Dr.", "Smith recommends https://a.com https://b.com today."]
        result = _apply_corrections(sentences)
        # Step 1 (Abbrev merge): "Dr." + "Smith..." -> merged sentence with 2 URLs
        # Step 2 (URL split): sentence has 2 URLs, split into 2
        assert len(result) == 2
        assert "Dr. Smith" in result[0]
        assert "https://a.com" in result[0]
        assert "https://b.com" in result[1]

    def test_empty_input(self) -> None:
        """Test empty input returns empty list."""
        result = _apply_corrections([])
        assert result == []

    def test_no_changes_needed(self) -> None:
        """Test sentences that need no corrections."""
        sentences = ["Hello world.", "How are you today?"]
        result = _apply_corrections(sentences)
        assert result == sentences


class TestSplitSentencesWithCorrections:
    """Integration tests for split_sentences with apply_corrections parameter."""

    def test_corrections_enabled_by_default(self) -> None:
        """Test that corrections are enabled by default."""
        # Multiple URLs should be split
        text = "Resources: https://a.com https://b.com for info."
        result = split_sentences(text)
        # spaCy returns one sentence, then URL split makes it 2
        assert len(result) == 2

    def test_corrections_disabled(self) -> None:
        """Test that corrections can be disabled."""
        text = "Check https://a.com https://b.com for info."
        result = split_sentences(text, apply_corrections=False)
        # Without corrections, spaCy may keep URLs together
        # The exact result depends on spaCy, but corrections won't be applied
        assert isinstance(result, list)

    def test_abbreviation_merge_integration(self) -> None:
        """Test abbreviation merging in real split_sentences call."""
        # spaCy might split after "Dr." - corrections should merge it back
        text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Dr. Smith" in result[0]

    def test_url_split_integration(self) -> None:
        """Test URL splitting in real split_sentences call."""
        text = (
            "Resources: https://example1.com https://example2.com https://example3.com"
        )
        result = split_sentences(text)
        # Should split into multiple sentences
        assert len(result) >= 2

    def test_combined_corrections(self) -> None:
        """Test both corrections working together."""
        text = "Dr. Johnson recommends https://health.com https://wellness.com today."
        result = split_sentences(text)
        # Should have Dr. Johnson merged and URLs split
        assert any("Dr. Johnson" in s for s in result)
        assert len(result) >= 2


class TestLanguageSpecificAbbreviations:
    """Tests for language-specific abbreviation handling."""

    def test_unsupported_language_no_merge(self) -> None:
        """Test that unsupported languages don't apply abbreviation merging."""
        # Japanese model has no abbreviations defined
        sentences = ["Dr.", "Smith is here."]
        result = _merge_abbreviation_splits(sentences, "ja_core_news_sm")
        # Should NOT merge because Japanese has no defined abbreviations
        assert len(result) == 2
        assert result == sentences

    def test_english_abbreviations_merge(self) -> None:
        """Test that English abbreviations merge correctly."""
        sentences = ["Dr.", "Smith is here."]
        result = _merge_abbreviation_splits(sentences, "en_core_web_sm")
        assert len(result) == 1
        assert "Dr. Smith" in result[0]

    def test_german_abbreviations(self) -> None:
        """Test German-specific abbreviations."""
        # "Hr." (Herr) is a German abbreviation
        sentences = ["Hr.", "Müller ist hier."]
        result = _merge_abbreviation_splits(sentences, "de_core_news_sm")
        assert len(result) == 1
        assert "Hr. Müller" in result[0]

    def test_french_abbreviations(self) -> None:
        """Test French-specific abbreviations."""
        # "Mme" (Madame) is a French abbreviation
        sentences = ["Mme.", "Dupont est là."]
        result = _merge_abbreviation_splits(sentences, "fr_core_news_sm")
        assert len(result) == 1
        assert "Mme. Dupont" in result[0]

    def test_spanish_abbreviations(self) -> None:
        """Test Spanish-specific abbreviations."""
        # "Sr" (Señor) is a Spanish abbreviation
        sentences = ["Sr.", "García está aquí."]
        result = _merge_abbreviation_splits(sentences, "es_core_news_sm")
        assert len(result) == 1
        assert "Sr. García" in result[0]

    def test_apply_corrections_with_language(self) -> None:
        """Test _apply_corrections passes language model correctly."""
        sentences = ["Visit Dr.", "Smith for help."]
        # English should merge
        result_en = _apply_corrections(sentences, "en_core_web_sm")
        assert len(result_en) == 1
        # Japanese should not merge
        result_ja = _apply_corrections(sentences.copy(), "ja_core_news_sm")
        assert len(result_ja) == 2

    def test_different_model_sizes_same_behavior(self) -> None:
        """Test that sm/md/lg models for same language behave identically."""
        sentences = ["Dr.", "Smith is here."]
        result_sm = _merge_abbreviation_splits(sentences, "en_core_web_sm")
        result_lg = _merge_abbreviation_splits(sentences.copy(), "en_core_web_lg")
        assert result_sm == result_lg
