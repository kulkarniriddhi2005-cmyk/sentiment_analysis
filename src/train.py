"""
train.py - Model Training Module for Sentiment Analysis
=========================================================

This module handles the complete training pipeline:
  1. Load the IMDb dataset (download if needed)
  2. Preprocess the text data
  3. Extract features using TF-IDF vectorization
  4. Train multiple ML models (Logistic Regression, Naive Bayes, SVM)
  5. Compare model performance
  6. Save the best model and vectorizer for later use

Why TF-IDF (Term Frequency - Inverse Document Frequency)?
  - TF-IDF converts text into numerical feature vectors that ML models
    can understand.
  - TF (Term Frequency): How often a word appears in a document.
    Words that appear frequently in a review are likely important.
  - IDF (Inverse Document Frequency): How rare a word is across ALL
    documents. Words that appear in many documents (like "the") get
    lower weight, while rare but informative words get higher weight.
  - Together, TF-IDF highlights words that are both frequent in a
    specific document AND rare across the corpus — these are the most
    discriminative features for classification.
  - TF-IDF is preferred over simple Bag-of-Words because it penalizes
    common words that don't carry sentiment information.

Why Compare Multiple Models?
  - Different algorithms make different assumptions about the data.
  - Logistic Regression: Works well with high-dimensional sparse data,
    provides probability scores, and is highly interpretable.
  - Multinomial Naive Bayes: Fast, works well with word counts/TF-IDF,
    assumes feature independence (naive assumption).
  - Linear SVM: Effective in high-dimensional spaces, robust to
    overfitting in text classification tasks.
  - Comparing multiple models helps us pick the best one and
    demonstrates understanding of the bias-variance tradeoff.
"""

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.evaluate import evaluate_model, generate_comparison_table, plot_confusion_matrix
from src.preprocessing import preprocess_text


# ============================================================
# Project paths - using pathlib for cross-platform compatibility
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"


def load_imdb_dataset(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load the IMDb Movie Reviews dataset.
    
    This function first checks for a local CSV file. If not found,
    it downloads the dataset using the Hugging Face `datasets` library.
    
    The IMDb dataset contains:
      - 25,000 training reviews
      - 25,000 test reviews
      - Binary labels: positive / negative
    
    Args:
        max_samples: Maximum number of samples to use (for faster
                     experimentation). If None, uses the full dataset.
    
    Returns:
        DataFrame with columns ['review', 'sentiment']
    """
    local_csv = DATA_RAW_DIR / "imdb_reviews.csv"

    if local_csv.exists():
        print(f"Loading dataset from local file: {local_csv}")
        df = pd.read_csv(local_csv)
    else:
        print("Local dataset not found. Downloading IMDb dataset...")
        print("(This is a one-time download, ~84 MB)\n")

        try:
            from datasets import load_dataset

            # Download the IMDb dataset from Hugging Face
            dataset = load_dataset("stanfordnlp/imdb")

            # Combine train and test splits into a single DataFrame
            train_df = pd.DataFrame(dataset["train"])
            test_df = pd.DataFrame(dataset["test"])
            df = pd.concat([train_df, test_df], ignore_index=True)

            # Rename columns to our standard names
            df.columns = ["review", "sentiment"]

            # Convert numeric labels to string labels
            df["sentiment"] = df["sentiment"].map({1: "positive", 0: "negative"})

            # Save locally for future runs
            DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(local_csv, index=False)
            print(f"Dataset saved to: {local_csv}")

        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to download the IMDb dataset.\n"
                "Install it with: pip install datasets\n"
                "Or manually place 'imdb_reviews.csv' in data/raw/"
            )

    # Subsample if requested (useful for quick experimentation)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using {max_samples} samples (subsampled for speed).")

    print(f"\nDataset loaded: {len(df)} reviews")
    print(f"  Positive: {(df['sentiment'] == 'positive').sum()}")
    print(f"  Negative: {(df['sentiment'] == 'negative').sum()}")

    return df


def preprocess_dataset(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """
    Apply the NLP preprocessing pipeline to the entire dataset.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns.
    
    Returns:
        Tuple of (preprocessed_texts, numeric_labels)
        where labels are: positive=1, negative=0
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATASET")
    print("=" * 60)

    start_time = time.time()

    # Apply preprocessing to each review
    texts = []
    total = len(df)
    for i, review in enumerate(df["review"]):
        texts.append(preprocess_text(review))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{total} reviews...")

    # Convert string labels to numeric: positive=1, negative=0
    label_map = {"positive": 1, "negative": 0}
    labels = df["sentiment"].str.lower().map(label_map).tolist()

    elapsed = time.time() - start_time
    print(f"\nPreprocessing complete in {elapsed:.1f} seconds.")

    return texts, labels


def build_tfidf_vectorizer(
    texts: list[str],
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Build and fit a TF-IDF Vectorizer on the training texts.
    
    TF-IDF Parameters Explained:
      - max_features: Maximum vocabulary size. Limits the number of unique
        terms to keep (top N by TF-IDF score). Prevents memory issues and
        reduces overfitting on very rare words.
      - ngram_range: (1, 2) means we use both unigrams ("good") and bigrams
        ("not good"). Bigrams capture word order and negation patterns that
        unigrams miss. "not good" is very different from "good"!
      - min_df: Minimum document frequency. Words appearing in fewer than
        min_df documents are discarded. Removes very rare misspellings.
      - max_df: Maximum document frequency (as proportion). Words appearing
        in more than 95% of documents are too common to be discriminative.
      - sublinear_tf: Applies logarithmic TF scaling (1 + log(tf)).
        Prevents very long documents from dominating just because they
        have more word occurrences.
    
    Args:
        texts: List of preprocessed text strings.
        max_features: Maximum vocabulary size.
        ngram_range: Range of n-gram sizes to extract.
        min_df: Minimum document frequency for a term.
        max_df: Maximum document frequency for a term.
        sublinear_tf: Whether to apply sublinear TF scaling.
    
    Returns:
        Fitted TfidfVectorizer object.
    """
    print("\n" + "=" * 60)
    print("BUILDING TF-IDF FEATURES")
    print("=" * 60)
    print(f"  max_features  = {max_features}")
    print(f"  ngram_range   = {ngram_range}")
    print(f"  min_df        = {min_df}")
    print(f"  max_df        = {max_df}")
    print(f"  sublinear_tf  = {sublinear_tf}")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )

    vectorizer.fit(texts)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"\nVocabulary size: {vocab_size} features")

    return vectorizer


def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict[str, dict[str, Any]]:
    """
    Train and evaluate multiple ML models for comparison.
    
    Models Trained:
      1. Logistic Regression
         - Works well with high-dimensional sparse features (TF-IDF).
         - Outputs calibrated probabilities via sigmoid function.
         - Highly interpretable: coefficients show feature importance.
         - L2 regularization prevents overfitting.
      
      2. Multinomial Naive Bayes
         - Specifically designed for text classification.
         - Very fast to train, works well with TF-IDF features.
         - Based on Bayes' theorem with the "naive" assumption that
           features (words) are independent given the class.
         - Often a strong baseline for text classification.
      
      3. Linear SVM (Support Vector Machine)
         - Finds the optimal hyperplane separating positive/negative.
         - Effective in high-dimensional spaces.
         - Robust to overfitting, especially with many features.
         - Does NOT output probabilities natively (no predict_proba).
    
    Args:
        X_train: TF-IDF feature matrix for training.
        y_train: Training labels.
        X_test: TF-IDF feature matrix for testing.
        y_test: Test labels.
    
    Returns:
        Dictionary mapping model names to their results
        (model object, metrics, training time).
    """
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    # Define the models to train
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,       # Ensure convergence
            C=1.0,               # Regularization strength (inverse)
            solver="lbfgs",      # Efficient solver for L2 penalty
            random_state=42,
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0,           # Laplace smoothing parameter
        ),
        "Linear SVM": LinearSVC(
            C=1.0,               # Regularization parameter
            max_iter=2000,       # Maximum iterations
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        start_time = time.time()

        # Train the model
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, model_name=name)

        # Store results
        results[name] = {
            "model": model,
            "metrics": metrics,
            "train_time": train_time,
        }

        print(f"  Training time: {train_time:.2f}s")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    return results


def select_best_model(results: dict) -> tuple[str, Any]:
    """
    Select the best model based on F1-score.
    
    F1-score is used because it balances both precision and recall.
    In sentiment analysis, we want to minimize both false positives
    (predicting positive when actually negative) and false negatives
    (predicting negative when actually positive).
    
    Args:
        results: Dictionary of model results from train_models().
    
    Returns:
        Tuple of (best_model_name, best_model_object).
    """
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    best_model = results[best_name]["model"]

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name}")
    print(f"  F1 Score: {results[best_name]['metrics']['f1_score']:.4f}")
    print(f"{'=' * 60}")

    return best_name, best_model


def save_artifacts(
    model,
    vectorizer: TfidfVectorizer,
    model_name: str,
    metrics: dict,
    num_train_samples: int,
    num_features: int,
) -> None:
    """
    Save the trained model, vectorizer, and metadata using joblib.
    
    Why joblib?
      - joblib is optimized for serializing large NumPy arrays,
        which is exactly what TF-IDF matrices and model weights are.
      - Faster and more space-efficient than pickle for numeric data.
      - Standard choice in scikit-learn ecosystem.
    
    Args:
        model: Trained scikit-learn model object.
        vectorizer: Fitted TfidfVectorizer.
        model_name: Human-readable name of the model.
        metrics: Dictionary of evaluation metrics.
        num_train_samples: Number of training samples used.
        num_features: Number of TF-IDF features.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_path = MODELS_DIR / "sentiment_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save the TF-IDF vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")

    # Save metadata (model info, metrics, training details)
    metadata = {
        "model_name": model_name,
        "num_train_samples": num_train_samples,
        "num_features": num_features,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def main(max_samples: int | None = None) -> None:
    """
    Main training pipeline - orchestrates the entire workflow.
    
    Steps:
      1. Load dataset
      2. Preprocess text
      3. Split into train/test sets
      4. Build TF-IDF features
      5. Train multiple models
      6. Compare and select best model
      7. Save artifacts
    
    Args:
        max_samples: Limit dataset size for faster experimentation.
                     Use None for the full dataset.
    """
    print("=" * 60)
    print("  SENTIMENT ANALYSIS - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # ---- Step 1: Load the dataset ----
    df = load_imdb_dataset(max_samples=max_samples)

    # ---- Step 2: Preprocess all reviews ----
    texts, labels = preprocess_dataset(df)

    # ---- Step 3: Split into training and testing sets ----
    # stratify=labels ensures both classes are proportionally represented
    # in both train and test sets (important for balanced evaluation)
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,       # 80% train, 20% test
        random_state=42,      # Reproducibility
        stratify=labels,      # Maintain class balance in both splits
    )
    print(f"  Training samples: {len(X_train_text)}")
    print(f"  Testing samples:  {len(X_test_text)}")

    # ---- Step 4: Build TF-IDF features ----
    vectorizer = build_tfidf_vectorizer(X_train_text)

    # Transform texts to TF-IDF feature matrices
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Feature matrix shape: {X_train.shape}")

    # ---- Step 5: Train models ----
    results = train_models(X_train, y_train, X_test, y_test)

    # ---- Step 6: Compare and select best model ----
    comparison_table = generate_comparison_table(results)
    print(f"\n{comparison_table}")

    best_name, best_model = select_best_model(results)

    # ---- Step 7: Generate visualizations ----
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        best_model,
        X_test,
        y_test,
        model_name=best_name,
        save_path=VISUALIZATIONS_DIR / "confusion_matrix.png",
    )

    # ---- Step 8: Save the best model and vectorizer ----
    save_artifacts(
        model=best_model,
        vectorizer=vectorizer,
        model_name=best_name,
        metrics=results[best_name]["metrics"],
        num_train_samples=len(X_train_text),
        num_features=X_train.shape[1],
    )

    # Save processed data for potential later use
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df = pd.DataFrame({
        "processed_text": texts,
        "sentiment": labels,
    })
    processed_path = DATA_PROCESSED_DIR / "processed_reviews.csv"
    processed_df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to: {processed_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest Model: {best_name}")
    print(f"Accuracy:   {results[best_name]['metrics']['accuracy']:.4f}")
    print(f"\nTo start the web app, run:")
    print(f"  streamlit run app/app.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment analysis models")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for quick testing). "
             "Default: None (use full dataset).",
    )
    args = parser.parse_args()

    main(max_samples=args.max_samples)
