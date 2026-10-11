"""
test_preprocessing.py - Unit Tests for the Preprocessing Module
================================================================

Tests ensure that each preprocessing step works correctly and
consistently. This is important because ANY change in preprocessing
between training and prediction would break the model.

Run tests with:
    python -m pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    preprocess_text,
    remove_html_tags,
    remove_urls,
    remove_punctuation,
    remove_special_characters,
)


class TestRemoveHTMLTags:
    """Test HTML tag removal."""

    def test_removes_br_tags(self):
        text = "Great movie!<br /><br />Loved it."
        result = remove_html_tags(text)
        assert "<br" not in result
        assert "Great movie!" in result

    def test_removes_paragraph_tags(self):
        text = "<p>This is a paragraph.</p>"
        result = remove_html_tags(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "This is a paragraph." in result

    def test_no_html_unchanged(self):
        text = "No HTML tags here."
        result = remove_html_tags(text)
        assert "No HTML tags here." == result.strip()

    def test_removes_nested_tags(self):
        text = "<div><b>Bold text</b></div>"
        result = remove_html_tags(text)
        assert "Bold text" in result
        assert "<" not in result


class TestRemoveURLs:
    """Test URL removal."""

    def test_removes_https_url(self):
        text = "Check out https://example.com for more."
        result = remove_urls(text)
        assert "https" not in result
        assert "example.com" not in result

    def test_removes_http_url(self):
        text = "Visit http://test.org today."
        result = remove_urls(text)
        assert "http" not in result

    def test_removes_www_url(self):
        text = "Go to www.example.com for info."
        result = remove_urls(text)
        assert "www" not in result

    def test_no_url_unchanged(self):
        text = "No URLs in this text."
        result = remove_urls(text)
        assert result.strip() == "No URLs in this text."


class TestRemovePunctuation:
    """Test punctuation removal."""

    def test_removes_exclamation(self):
        result = remove_punctuation("Amazing!!!")
        assert "!" not in result

    def test_removes_question_mark(self):
        result = remove_punctuation("Is this good?")
        assert "?" not in result

    def test_removes_commas_periods(self):
        result = remove_punctuation("Hello, world.")
        assert "," not in result
        assert "." not in result


class TestRemoveSpecialCharacters:
    """Test special character removal."""

    def test_removes_numbers(self):
        result = remove_special_characters("Product 123 is great")
        assert "123" not in result

    def test_removes_emojis(self):
        result = remove_special_characters("Great movie 😊👍")
        assert "😊" not in result
        assert "👍" not in result

    def test_keeps_letters(self):
        result = remove_special_characters("Hello World")
        assert "Hello" in result
        assert "World" in result


class TestPreprocessText:
    """Test the complete preprocessing pipeline."""

    def test_basic_preprocessing(self):
        text = "I REALLY loved this movie!!! It was amazing 😊"
        result = preprocess_text(text)
        # Should be lowercase
        assert result == result.lower()
        # Should not contain punctuation
        assert "!" not in result
        # Should not contain emojis
        assert "😊" not in result

    def test_html_removal(self):
        text = "Great film<br /><br />Really enjoyed it."
        result = preprocess_text(text)
        assert "<br" not in result

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_none_input(self):
        assert preprocess_text(None) == ""

    def test_whitespace_only(self):
        assert preprocess_text("   ") == ""

    def test_stop_words_removed(self):
        text = "This is a very good movie"
        result = preprocess_text(text)
        # Common stop words should be removed
        assert "this" not in result.split()
        # Content words should remain
        assert "good" in result
        assert "movie" in result

    def test_negation_preserved(self):
        """Negation words should NOT be removed as stop words."""
        text = "This movie is not good at all"
        result = preprocess_text(text)
        assert "not" in result

    def test_lemmatization(self):
        text = "The movies were absolutely amazing"
        result = preprocess_text(text)
        # "movies" should be lemmatized to "movie"
        assert "movie" in result

    def test_url_removal_in_pipeline(self):
        text = "Check https://example.com this is great"
        result = preprocess_text(text)
        assert "https" not in result
        assert "example" not in result

    def test_consistent_output(self):
        """Same input should always produce same output (deterministic)."""
        text = "This is a wonderful product!"
        result1 = preprocess_text(text)
        result2 = preprocess_text(text)
        assert result1 == result2

    def test_preserves_meaningful_content(self):
        """Important content words should survive preprocessing."""
        text = "The product quality was terrible and disappointing"
        result = preprocess_text(text)
        assert "terrible" in result
        assert "disappointing" in result


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_word(self):
        result = preprocess_text("Amazing")
        assert "amazing" in result

    def test_mixed_case(self):
        result = preprocess_text("GREAT Product")
        assert result == result.lower()

    def test_multiple_spaces(self):
        result = preprocess_text("too    many     spaces")
        # Should not have multiple consecutive spaces
        assert "  " not in result

    def test_numeric_input(self):
        result = preprocess_text("12345")
        # Pure numbers should result in empty or very short string
        assert len(result) <= 1
