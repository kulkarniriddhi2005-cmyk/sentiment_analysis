# 🎭 Sentiment Analysis using NLP & Machine Learning

A production-quality **Sentiment Analysis** web application that predicts whether a given text/review expresses a **Positive** or **Negative** sentiment. Built with traditional NLP techniques and explainable Machine Learning - no black-box LLMs.

---

## 📌 Project Description

This project implements a complete **Natural Language Processing (NLP)** pipeline for binary sentiment classification. Given a piece of text (e.g., a movie review), the system:

1. **Preprocesses** the text (cleaning, tokenization, lemmatization)
2. **Extracts features** using TF-IDF vectorization
3. **Classifies** the sentiment using a trained ML model
4. **Displays** the result with a confidence score in a web interface

**Example:**

| Input | Output |
|-------|--------|
| "I absolutely loved this product. The quality is amazing!" | Sentiment: **Positive** 😊 · Confidence: **96.4%** |
| "The product quality is terrible and it broke within a week." | Sentiment: **Negative** 😞 · Confidence: **93.2%** |

---

## ✨ Features

- ✅ **Complete NLP Pipeline** — Text cleaning, tokenization, stop-word removal, lemmatization
- ✅ **TF-IDF Feature Extraction** — Unigrams + Bigrams for capturing word order
- ✅ **3 ML Models Compared** — Logistic Regression, Naive Bayes, Linear SVM
- ✅ **Best Model Auto-Selection** — Based on F1-score performance
- ✅ **Streamlit Web App** — Clean, interactive UI with confidence scores
- ✅ **Batch Prediction** — Upload CSV, get predictions, download results
- ✅ **Example Reviews** — Quick-test buttons for positive/negative examples
- ✅ **Model Metrics Dashboard** — View accuracy, precision, recall, F1 in sidebar
- ✅ **Preprocessed Text View** — See how the NLP pipeline transforms your input
- ✅ **Visualizations** — Confusion matrix, model comparison chart, class distribution
- ✅ **Reproducible** — Fixed random seeds, saved models, documented pipeline

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| NLP | NLTK |
| ML | scikit-learn |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | joblib |
| Dataset | IMDb Movie Reviews (50K reviews) |

---

## 🔧 NLP Preprocessing Pipeline

The preprocessing pipeline is applied identically during both training and prediction:

```
Raw Text → Lowercase → Remove HTML → Remove URLs → Remove Punctuation
→ Remove Special Characters → Tokenize → Remove Stop Words
→ Lemmatize → Reconstruct Text
```

**Example:**

```
Original:  "I REALLY loved this movie!!! It was amazing 😊"
Processed: "really love movie amazing"
```

**Key Design Decisions:**
- **Lemmatization over Stemming**: Produces real words ("studies" → "study" vs "studi")
- **Negation words preserved**: Words like "not", "never" are kept because they flip sentiment
- **Bigrams captured**: "not good" is treated differently from "good" via n-grams

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── data/
│   ├── raw/                      # Raw dataset (auto-downloaded)
│   └── processed/                # Preprocessed data
│
├── notebooks/
│   └── (optional Jupyter notebooks)
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── preprocessing.py          # NLP text preprocessing pipeline
│   ├── train.py                  # Model training & comparison
│   ├── evaluate.py               # Evaluation metrics & visualizations
│   └── predict.py                # Prediction class for inference
│
├── models/
│   ├── sentiment_model.pkl       # Saved best model (after training)
│   ├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
│   └── model_metadata.json       # Model info & metrics
│
├── app/
│   └── app.py                    # Streamlit web application
│
├── tests/
│   └── test_preprocessing.py     # Unit tests for preprocessing
│
├── visualizations/               # Generated plots
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── class_distribution.png
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

---

## 📊 Dataset

**IMDb Movie Reviews Dataset** — 50,000 movie reviews labeled as positive or negative.

- **Source**: [IMDb via Hugging Face Datasets](https://huggingface.co/datasets/imdb)
- **Size**: 25,000 training + 25,000 test reviews
- **Labels**: Positive (1) / Negative (0) — balanced dataset
- **Download**: Automatic on first training run (via `datasets` library)

The dataset is automatically downloaded and saved to `data/raw/imdb_reviews.csv` when you run the training script for the first time.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Activate it:**

- **Windows PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

- **Windows CMD:**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ How to Train the Model

From the project root directory:

```bash
python -m src.train
```

**What this does:**
1. Downloads the IMDb dataset (first run only, ~84 MB)
2. Preprocesses all 50,000 reviews
3. Splits data into 80% training / 20% testing
4. Builds TF-IDF features (unigrams + bigrams)
5. Trains 3 models: Logistic Regression, Naive Bayes, Linear SVM
6. Compares performance and selects the best model
7. Saves the model, vectorizer, and metadata to `models/`
8. Generates visualizations in `visualizations/`

**Quick test (faster, uses fewer samples):**

```bash
python -m src.train --max-samples 5000
```

---

## 🌐 How to Run the Streamlit App

After training the model:

```bash
streamlit run app/app.py
```

The app opens in your browser at `http://localhost:8501`.

**Features in the app:**
- Type/paste any review and click "Analyze Sentiment"
- Try built-in positive/negative example reviews
- Upload a CSV file for batch analysis
- View model details and metrics in the sidebar

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📈 Model Evaluation

Three models are trained and compared:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model, outputs calibrated probabilities, highly interpretable |
| **Multinomial Naive Bayes** | Probabilistic model based on Bayes' theorem, fast to train |
| **Linear SVM** | Finds optimal separating hyperplane, robust in high dimensions |

After training, a comparison table with **actual** metrics is printed:

```
Model                          Accuracy  Precision    Recall  F1 Score
----------------------------------------------------------------------
Logistic Regression              0.9064     0.9066    0.9064    0.9064
Multinomial Naive Bayes          0.8864     0.8865    0.8864    0.8864
Linear SVM                       0.9106     0.9108    0.9106    0.9106
```

**Best Model: Linear SVM** with **91.06% accuracy** and **0.9106 F1-score**.

A confusion matrix visualization is saved to `visualizations/confusion_matrix.png`.

---

## 🔮 Example Predictions

After training, you can also use the interactive CLI predictor:

```bash
python -m src.predict
```

```
Enter review: This movie was absolutely fantastic, I loved every moment!
  Sentiment:      Positive 😊
  Confidence:     95.3%
  Processed Text: "movie absolutely fantastic love every moment"

Enter review: Terrible acting and a boring plot. Complete waste of time.
  Sentiment:      Negative 😞
  Confidence:     92.7%
  Processed Text: "terrible acting boring plot complete waste time"
```

---

## 🚀 Future Improvements

1. **Word Embeddings** — Use Word2Vec or GloVe for semantic word representations
2. **Deep Learning** — Implement LSTM/GRU recurrent neural networks
3. **Transformer Models** — Fine-tune BERT or DistilBERT for state-of-the-art accuracy
4. **Multi-class Sentiment** — Extend to Positive / Neutral / Negative
5. **Aspect-based Analysis** — Identify sentiment toward specific product aspects
6. **Real-time Streaming** — Analyze live Twitter/Reddit data
7. **Explainability** — Highlight which words influenced the prediction (LIME/SHAP)
8. **Docker Deployment** — Containerize for cloud deployment
9. **REST API** — Add Flask/FastAPI endpoint for programmatic access
10. **Cross-domain Transfer** — Test on Amazon, Yelp, and other review datasets

---

## 🎓 Interview Preparation Topics

This project demonstrates understanding of:

| # | Topic | Where in Code |
|---|-------|--------------|
| 1 | What is NLP? | `src/preprocessing.py` — complete text processing pipeline |
| 2 | What is Sentiment Analysis? | `src/train.py` — binary classification of text |
| 3 | Why TF-IDF? | `src/train.py` — `build_tfidf_vectorizer()` comments |
| 4 | How TF-IDF works | `src/train.py` — parameter explanations |
| 5 | Why Logistic Regression? | `src/train.py` — `train_models()` model descriptions |
| 6 | Tokenization | `src/preprocessing.py` — `word_tokenize()` step |
| 7 | Stop word removal | `src/preprocessing.py` — negation words discussion |
| 8 | Stemming vs Lemmatization | `src/preprocessing.py` — module docstring |
| 9 | N-grams | `src/train.py` — `ngram_range=(1,2)` explanation |
| 10 | Confusion Matrix | `src/evaluate.py` — `plot_confusion_matrix()` |
| 11 | Metrics (Acc/Prec/Rec/F1) | `src/evaluate.py` — module docstring |
| 12 | Why compare models? | `src/train.py` — `train_models()` docstring |
| 13 | Complete pipeline | `src/predict.py` — `SentimentPredictor.predict()` |
| 14 | Future improvements | This README — Future Improvements section |

---

## 👤 Author

**Riddhi**

- GitHub: [github.com/yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
