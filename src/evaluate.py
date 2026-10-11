"""
evaluate.py - Model Evaluation Module for Sentiment Analysis
==============================================================

This module provides functions for evaluating trained ML models:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix generation and visualization
  - Model comparison table
  - Performance visualization plots

Understanding Evaluation Metrics:
  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Proportion of all predictions that are correct.
    - Can be misleading with imbalanced datasets.
  
  Precision = TP / (TP + FP)
    - Of all predicted positives, how many are actually positive?
    - Important when false positives are costly.
    - Example: Marking a legitimate review as spam.
  
  Recall = TP / (TP + FN)
    - Of all actual positives, how many did we correctly identify?
    - Important when false negatives are costly.
    - Example: Missing a negative review about a product defect.
  
  F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    - Harmonic mean of precision and recall.
    - Balances both metrics, especially useful when classes are imbalanced.
    - We use F1 as our primary selection metric.

  Confusion Matrix:
    - 2x2 matrix showing:
      [[True Negatives,  False Positives],
       [False Negatives, True Positives]]
    - Helps visualize WHERE the model makes mistakes.
    - Example: If many negative reviews are classified as positive,
      the model may struggle with sarcasm or subtle negativity.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server/script use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
) -> dict[str, float]:
    """
    Evaluate a trained model on test data and return all metrics.
    
    Args:
        model: Trained scikit-learn model with .predict() method.
        X_test: Test feature matrix (TF-IDF transformed).
        y_test: True test labels.
        model_name: Human-readable name for logging.
    
    Returns:
        Dictionary with keys: accuracy, precision, recall, f1_score.
    """
    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics using 'weighted' average for multi-class support
    # (though binary here, this makes the code future-proof)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    return metrics


def print_classification_report(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
) -> str:
    """
    Print and return a detailed classification report.
    
    The classification report shows per-class precision, recall, and F1:
      - Class 0 (Negative): How well the model identifies negative reviews
      - Class 1 (Positive): How well the model identifies positive reviews
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        model_name: Model name for display.
    
    Returns:
        Classification report as a formatted string.
    """
    y_pred = model.predict(X_test)
    target_names = ["Negative", "Positive"]

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\n{'=' * 50}")
    print(f"Classification Report: {model_name}")
    print(f"{'=' * 50}")
    print(report)

    return report


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate and save a confusion matrix heatmap.
    
    The confusion matrix is a key visualization that shows:
      - Top-left (TN): Correctly predicted negative reviews
      - Top-right (FP): Negative reviews incorrectly predicted as positive
      - Bottom-left (FN): Positive reviews incorrectly predicted as negative
      - Bottom-right (TP): Correctly predicted positive reviews
    
    A good model has high values on the diagonal and low off-diagonal values.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        model_name: Model name for the title.
        save_path: Path to save the plot image. If None, just displays.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5,
        square=True,
        annot_kws={"size": 14},
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nConfusion matrix saved to: {save_path}")

    plt.close()


def plot_model_comparison(
    results: dict[str, dict[str, Any]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Create a bar chart comparing all models across metrics.
    
    This visualization makes it easy to see which model performs best
    overall and where specific models excel or struggle.
    
    Args:
        results: Dictionary from train_models() with model results.
        save_path: Path to save the plot.
    """
    model_names = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    # Extract metric values for each model
    metric_values = {
        metric: [results[name]["metrics"][metric] for name in model_names]
        for metric in metrics_names
    }

    x = np.arange(len(model_names))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, (metric, display) in enumerate(zip(metrics_names, display_names)):
        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            metric_values[metric],
            width,
            label=display,
            color=colors[i],
            alpha=0.85,
        )
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.003,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        multiplier += 1

    ax.set_xlabel("Models", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.75, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison chart saved to: {save_path}")

    plt.close()


def plot_class_distribution(
    labels: list[int],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the distribution of sentiment classes in the dataset.
    
    This is important to check for class imbalance. If one class
    significantly outnumbers the other, the model may be biased
    toward the majority class.
    
    Args:
        labels: List of numeric labels (0=Negative, 1=Positive).
        save_path: Path to save the plot.
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ["Negative" if u == 0 else "Positive" for u in unique]

    plt.figure(figsize=(8, 5))
    colors = ["#EF5350", "#66BB6A"]
    bars = plt.bar(class_names, counts, color=colors, edgecolor="white", linewidth=1.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 200,
            f"{count:,}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    plt.title("Sentiment Class Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=13)
    plt.ylabel("Number of Reviews", fontsize=13)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Class distribution plot saved to: {save_path}")

    plt.close()


def generate_comparison_table(results: dict[str, dict[str, Any]]) -> str:
    """
    Generate a formatted comparison table of all models.
    
    Args:
        results: Dictionary from train_models() with model results.
    
    Returns:
        Formatted table string.
    """
    header = f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'Time (s)':>10}"
    separator = "-" * len(header)

    lines = ["\n" + separator, header, separator]

    for name, data in results.items():
        m = data["metrics"]
        t = data["train_time"]
        lines.append(
            f"{name:<30} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {t:>10.2f}"
        )

    lines.append(separator)
    return "\n".join(lines)
