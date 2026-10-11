"""
preprocessing.py - Text Preprocessing Module for Sentiment Analysis
=====================================================================

This module contains the complete NLP preprocessing pipeline used for
cleaning and normalizing text data before feeding it to ML models.

The SAME preprocessing function is used during both:
  1. Training (to preprocess the dataset)
  2. Prediction (to preprocess user input at inference time)

This ensures consistency between training and inference.

NLP Preprocessing Steps:
  1. Lowercase conversion - Standardizes text case
  2. HTML tag removal - Cleans web-scraped data
  3. URL removal - Removes irrelevant web links
  4. Punctuation removal - Strips punctuation marks
  5. Special character removal - Removes emojis, symbols, etc.
  6. Tokenization - Splits text into individual words (tokens)
  7. Stop word removal - Removes common words that don't carry sentiment
  8. Lemmatization - Reduces words to their dictionary base form
  9. Text reconstruction - Joins tokens back into a clean string

Why Lemmatization over Stemming?
  - Stemming (e.g., Porter Stemmer) aggressively chops word endings,
    sometimes producing non-real words (e.g., "studies" -> "studi").
  - Lemmatization uses vocabulary and morphological analysis to return
    the actual dictionary form (e.g., "studies" -> "study").
  - Lemmatization produces more meaningful features for the model.
"""

import re
import string
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources() -> None:
    """
    Download required NLTK data packages if not already present.
    
    Downloads:
      - punkt_tab: Tokenizer models for splitting text into words/sentences
      - stopwords: List of common words to filter out (the, is, at, etc.)
      - wordnet: Lexical database used by the lemmatizer
    """
    resources = ["punkt_tab", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# Download NLTK resources when this module is first imported
download_nltk_resources()

# Initialize the lemmatizer and stop words set once (reused across calls)
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# -------------------------------------------------------
# Important: We keep some negation words in the text
# because they carry critical sentiment information.
# For example:
#   "not good" vs "good" have opposite sentiments.
#   Removing "not" would lose this distinction.
# -------------------------------------------------------
_negation_words = {
    "not", "no", "nor", "neither", "never", "nobody",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "don", "dont", "doesn", "doesnt", "didn", "didnt",
    "won", "wont", "wouldn", "wouldnt", "shouldn", "shouldnt",
    "couldn", "couldnt", "isn", "isnt", "aren", "arent",
    "wasn", "wasnt", "weren", "werent", "hasn", "hasnt",
    "haven", "havent", "hadn", "hadnt",
}

# Remove negation words from the stop words set so they are preserved
_stop_words = _stop_words - _negation_words


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Many review datasets (like IMDb) contain HTML markup such as <br />,
    <p>, <b>, etc. These need to be stripped before analysis.
    
    Args:
        text: Raw text potentially containing HTML tags.
    
    Returns:
        Text with all HTML tags removed.
    
    Example:
        >>> remove_html_tags("Great movie!<br /><br />Loved it.")
        'Great movie!  Loved it.'
    """
    clean = re.compile(r"<.*?>")
    return re.sub(clean, " ", text)


def remove_urls(text: str) -> str:
    """
    Remove URLs (http, https, www) from text.
    
    URLs don't contribute to sentiment and add noise to feature extraction.
    
    Args:
        text: Text potentially containing URLs.
    
    Returns:
        Text with URLs removed.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return re.sub(url_pattern, " ", text)


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from text.
    
    Punctuation like !, ?, ., , etc. are removed because:
      - They don't carry consistent sentiment meaning.
      - They add noise to TF-IDF features.
      - The model learns from words, not punctuation.
    
    Args:
        text: Text containing punctuation.
    
    Returns:
        Text with punctuation replaced by spaces.
    """
    # Create translation table that maps each punctuation char to a space
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def remove_special_characters(text: str) -> str:
    """
    Remove special characters, numbers, and emojis - keep only alphabetic
    characters and spaces.
    
    Args:
        text: Text potentially containing special characters.
    
    Returns:
        Text containing only letters and spaces.
    """
    # Keep only letters (a-z, A-Z) and whitespace
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def preprocess_text(text: str) -> str:
    """
    Complete NLP preprocessing pipeline for sentiment analysis.
    
    This is the MAIN function that applies all preprocessing steps in order.
    It is used during both training and prediction to ensure consistency.
    
    Pipeline:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove punctuation
        5. Remove special characters
        6. Tokenize
        7. Remove stop words (keeping negation words)
        8. Lemmatize
        9. Reconstruct text
    
    Args:
        text: Raw input text/review.
    
    Returns:
        Cleaned and normalized text string.
    
    Example:
        >>> preprocess_text("I REALLY loved this movie!!! It was amazing 😊")
        'really love movie amazing'
    """
    # Handle edge cases
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Step 1: Convert to lowercase
    # This ensures "Good", "GOOD", "good" are all treated the same
    text = text.lower()

    # Step 2: Remove HTML tags (common in web-scraped reviews)
    text = remove_html_tags(text)

    # Step 3: Remove URLs
    text = remove_urls(text)

    # Step 4: Remove punctuation
    text = remove_punctuation(text)

    # Step 5: Remove special characters, numbers, emojis
    text = remove_special_characters(text)

    # Step 6: Tokenize - split text into individual word tokens
    # word_tokenize is more sophisticated than str.split() because
    # it handles contractions and edge cases properly
    tokens = word_tokenize(text)

    # Step 7: Remove stop words (common words like "the", "is", "at")
    # Stop words don't carry sentiment meaning and increase feature space
    # NOTE: We keep negation words (not, never, etc.) as they flip sentiment
    tokens = [token for token in tokens if token not in _stop_words]

    # Step 8: Lemmatize - reduce words to their base dictionary form
    # "loved" -> "love", "movies" -> "movie", "running" -> "running"
    # This reduces vocabulary size and groups related words together
    tokens = [_lemmatizer.lemmatize(token) for token in tokens]

    # Step 9: Remove very short tokens (single characters) that are noise
    tokens = [token for token in tokens if len(token) > 1]

    # Step 10: Reconstruct the cleaned text from tokens
    cleaned_text = " ".join(tokens)

    return cleaned_text


def preprocess_dataframe(
    df,
    text_column: str = "review",
    label_column: Optional[str] = "sentiment",
) -> "tuple":
    """
    Preprocess an entire DataFrame of reviews.
    
    Applies the preprocessing pipeline to every row in the text column
    and optionally encodes sentiment labels to numeric values.
    
    Args:
        df: pandas DataFrame containing the review data.
        text_column: Name of the column containing review text.
        label_column: Name of the column containing sentiment labels.
                      If None, only text preprocessing is performed.
    
    Returns:
        Tuple of (processed_texts: list[str], labels: list[int] or None)
    """
    import pandas as pd

    print(f"Preprocessing {len(df)} reviews...")
    print("This may take a few minutes for large datasets.\n")

    # Apply preprocessing to each review
    processed_texts = df[text_column].apply(preprocess_text).tolist()

    # Encode labels if label column is provided
    labels = None
    if label_column and label_column in df.columns:
        # Map string labels to integers: positive=1, negative=0
        label_map = {"positive": 1, "negative": 0}
        if df[label_column].dtype == object:
            labels = df[label_column].str.lower().map(label_map).tolist()
        else:
            labels = df[label_column].tolist()

    # Count processed results
    non_empty = sum(1 for t in processed_texts if len(t) > 0)
    print(f"Successfully preprocessed {non_empty}/{len(df)} reviews.")

    return processed_texts, labels
