"""
predict.py - Prediction Module for Sentiment Analysis
========================================================

This module handles loading the saved model and vectorizer, and
making predictions on new text input.

How the Complete Prediction Pipeline Works:
  1. User provides raw text (review).
  2. The SAME preprocessing pipeline used during training is applied
     to clean the text (lowercase, remove HTML/URLs/punctuation,
     tokenize, remove stop words, lemmatize).
  3. The saved TF-IDF vectorizer transforms the cleaned text into
     a feature vector using the SAME vocabulary learned during training.
  4. The saved model predicts the sentiment (0=Negative, 1=Positive).
  5. If the model supports it, probability scores are also returned
     to show confidence in the prediction.

IMPORTANT: The preprocessing and vectorization must be IDENTICAL to
what was used during training. Any difference would mean the features
don't align, leading to garbage predictions.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_text


# Path to the saved models directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class SentimentPredictor:
    """
    Sentiment Prediction class that encapsulates the complete
    prediction pipeline.
    
    Usage:
        predictor = SentimentPredictor()
        result = predictor.predict("This movie was amazing!")
        print(result)
        # {'sentiment': 'Positive', 'confidence': 0.964, ...}
    
    Attributes:
        model: The loaded scikit-learn classifier.
        vectorizer: The loaded TF-IDF vectorizer.
        metadata: Training metadata (model name, metrics, etc.)
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the predictor by loading the saved model and vectorizer.
        
        Args:
            models_dir: Path to the directory containing saved model files.
                        Defaults to the project's models/ directory.
        
        Raises:
            FileNotFoundError: If model files are not found.
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

        model_path = self.models_dir / "sentiment_model.pkl"
        vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
        metadata_path = self.models_dir / "model_metadata.json"

        # Validate that model files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please train the model first by running:\n"
                "  python -m src.train"
            )
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {vectorizer_path}\n"
                "Please train the model first by running:\n"
                "  python -m src.train"
            )

        # Load the saved model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Load metadata if available
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

        print(f"Model loaded: {self.metadata.get('model_name', 'Unknown')}")

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text input.
        
        The complete prediction pipeline:
          1. Preprocess the text (same pipeline as training)
          2. Transform to TF-IDF features (same vectorizer as training)
          3. Predict class label (0=Negative, 1=Positive)
          4. Get confidence score (probability, if available)
        
        Args:
            text: Raw text/review to analyze.
        
        Returns:
            Dictionary containing:
              - sentiment: "Positive" or "Negative"
              - confidence: float between 0.0 and 1.0
              - processed_text: cleaned version of input
              - label: numeric label (0 or 1)
        """
        # Step 1: Preprocess the input text using the same pipeline
        processed = preprocess_text(text)

        # Step 2: Transform to TF-IDF feature vector
        # The vectorizer uses the SAME vocabulary from training
        features = self.vectorizer.transform([processed])

        # Step 3: Predict the sentiment class
        prediction = self.model.predict(features)[0]

        # Step 4: Get confidence/probability score
        confidence = self._get_confidence(features, prediction)

        # Map numeric label to human-readable sentiment
        sentiment = "Positive" if prediction == 1 else "Negative"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "processed_text": processed,
            "label": int(prediction),
        }

    def _get_confidence(self, features, prediction: int) -> float:
        """
        Get the confidence score (probability) for the prediction.
        
        Not all models support predict_proba():
          - Logistic Regression: Yes (sigmoid probability)
          - Naive Bayes: Yes (posterior probability)
          - Linear SVM: No (uses decision_function instead)
        
        For SVM, we use the decision function value and convert it to
        a pseudo-probability using the sigmoid function.
        
        Args:
            features: TF-IDF feature vector.
            prediction: Predicted class label.
        
        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if hasattr(self.model, "predict_proba"):
            # Models like LogisticRegression and NaiveBayes
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        elif hasattr(self.model, "decision_function"):
            # SVM - convert decision function to pseudo-probability
            decision = self.model.decision_function(features)[0]
            # Apply sigmoid function: 1 / (1 + e^(-x))
            confidence = 1.0 / (1.0 + np.exp(-abs(decision)))
        else:
            confidence = 0.0

        return float(round(confidence, 4))

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of raw text strings.
        
        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(text) for text in texts]

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "review") -> pd.DataFrame:
        """
        Predict sentiment for a DataFrame column and return results.
        
        This is used for the batch CSV upload feature in the Streamlit app.
        
        Args:
            df: DataFrame containing text data.
            text_column: Name of the column with text to analyze.
        
        Returns:
            DataFrame with additional columns:
              - Predicted Sentiment
              - Confidence
              - Processed Text
        """
        results = []
        total = len(df)

        for i, text in enumerate(df[text_column]):
            if pd.isna(text) or str(text).strip() == "":
                results.append({
                    "sentiment": "Unknown",
                    "confidence": 0.0,
                    "processed_text": "",
                })
            else:
                results.append(self.predict(str(text)))

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total}")

        result_df = df.copy()
        result_df["Predicted Sentiment"] = [r["sentiment"] for r in results]
        result_df["Confidence"] = [r["confidence"] for r in results]
        result_df["Processed Text"] = [r["processed_text"] for r in results]

        return result_df

    def get_model_info(self) -> dict:
        """
        Return metadata about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.metadata.get("model_name", "Unknown"),
            "num_train_samples": self.metadata.get("num_train_samples", "N/A"),
            "num_features": self.metadata.get("num_features", "N/A"),
            "metrics": self.metadata.get("metrics", {}),
        }


def main():
    """
    Interactive command-line prediction demo.
    
    Allows users to type reviews and get instant sentiment predictions.
    """
    print("=" * 50)
    print("  SENTIMENT ANALYSIS - INTERACTIVE PREDICTOR")
    print("=" * 50)
    print("Type a review and press Enter to analyze sentiment.")
    print("Type 'quit' to exit.\n")

    try:
        predictor = SentimentPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    while True:
        text = input("\nEnter review: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            print("Please enter some text.")
            continue

        result = predictor.predict(text)
        emoji = "😊" if result["sentiment"] == "Positive" else "😞"
        print(f"\n  Sentiment:      {result['sentiment']} {emoji}")
        print(f"  Confidence:     {result['confidence'] * 100:.1f}%")
        print(f"  Processed Text: \"{result['processed_text']}\"")


if __name__ == "__main__":
    main()
