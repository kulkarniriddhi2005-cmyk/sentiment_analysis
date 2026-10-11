"""
app.py - Streamlit Web Application for Sentiment Analysis
============================================================

This is the web-based user interface for the Sentiment Analysis project.
Built with Streamlit, it provides:
  1. Single text analysis with sentiment & confidence display
  2. Example reviews for quick testing
  3. Batch CSV upload and download
  4. Model information and metrics display
  5. Preprocessed text visualization

Streamlit runs this file as:
  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Ensure the project root is on the Python path so 'src' is importable ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import SentimentPredictor


# ==========================================================================
# Page Configuration
# ==========================================================================
st.set_page_config(
    page_title="Sentiment Analysis using NLP",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Custom CSS for a Premium Look
# ==========================================================================
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }

    .result-positive {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    }
    .result-negative {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }

    .sentiment-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .confidence-label {
        font-size: 1.3rem;
        font-weight: 500;
        opacity: 0.85;
    }

    /* Info cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
        margin-bottom: 0.8rem;
    }
    .info-card h4 {
        margin: 0 0 0.3rem 0;
        color: #333;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.7;
    }
    .info-card p {
        margin: 0;
        color: #111;
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* Example buttons */
    .example-section {
        background: #f8f9fe;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    /* Processed text display */
    .processed-text {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #555;
        border: 1px solid #e0e3e8;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Load Model (cached so it loads only once)
# ==========================================================================
@st.cache_resource
def load_predictor():
    """Load the sentiment predictor. Cached to avoid reloading on every interaction."""
    try:
        return SentimentPredictor()
    except FileNotFoundError:
        return None


predictor = load_predictor()


# ==========================================================================
# Header Section
# ==========================================================================
st.markdown("""
<div class="main-header">
    <h1>🎭 Sentiment Analysis using NLP</h1>
    <p>Analyze the sentiment of any text or review using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================================
# Check if model is loaded
# ==========================================================================
if predictor is None:
    st.error(
        "⚠️ **Model not found!** Please train the model first by running:\n\n"
        "```bash\npython -m src.train\n```\n\n"
        "This will download the dataset, train the models, and save the best one."
    )
    st.stop()


# ==========================================================================
# Sidebar - Model Information
# ==========================================================================
with st.sidebar:
    st.markdown("## 📊 Model Information")
    model_info = predictor.get_model_info()

    st.markdown(f"""
    <div class="info-card">
        <h4>Model Used</h4>
        <p>{model_info['model_name']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <h4>Training Samples</h4>
        <p>{model_info['num_train_samples']:,}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <h4>TF-IDF Features</h4>
        <p>{model_info['num_features']:,}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display evaluation metrics
    metrics = model_info.get("metrics", {})
    if metrics:
        st.markdown("---")
        st.markdown("## 📈 Evaluation Metrics")

        metric_display = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
        }
        for key, display_name in metric_display.items():
            if key in metrics:
                value = metrics[key]
                st.markdown(f"""
                <div class="info-card">
                    <h4>{display_name}</h4>
                    <p>{value:.4f} ({value * 100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown(
        "This project uses a traditional **NLP + ML pipeline** "
        "with **TF-IDF** features and a trained classifier to "
        "predict sentiment from text.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Built with **scikit-learn**, **NLTK**, and **Streamlit**."
    )


# ==========================================================================
# Main Content - Tabs
# ==========================================================================
tab1, tab2 = st.tabs(["✍️ Single Analysis", "📁 Batch Analysis"])


# ==========================================================================
# Tab 1: Single Text Analysis
# ==========================================================================
with tab1:
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown("### Enter Your Review")
        user_text = st.text_area(
            label="Review Text",
            placeholder="Type or paste your review here...\n\nExample: I absolutely loved this product. The quality is amazing!",
            height=200,
            label_visibility="collapsed",
            key="review_input",
        )

        # Analyze button
        analyze_clicked = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
        )

        # Example Reviews Section
        st.markdown("### 💡 Try an Example")
        example_col1, example_col2 = st.columns(2)

        with example_col1:
            if st.button("😊 Positive Example", use_container_width=True):
                st.session_state["example_text"] = (
                    "This product is absolutely fantastic! "
                    "The quality exceeded my expectations and the delivery was super fast. "
                    "I would definitely recommend this to everyone."
                )
                st.rerun()

        with example_col2:
            if st.button("😞 Negative Example", use_container_width=True):
                st.session_state["example_text"] = (
                    "The product quality is terrible and it broke within a week. "
                    "Customer service was unhelpful and rude. "
                    "Total waste of money, do not buy this."
                )
                st.rerun()

        # Check for example text
        if "example_text" in st.session_state:
            user_text = st.session_state.pop("example_text")
            analyze_clicked = True

    with col_result:
        st.markdown("### Result")

        if analyze_clicked and user_text.strip():
            # Run prediction
            result = predictor.predict(user_text)

            sentiment = result["sentiment"]
            confidence = result["confidence"]
            processed = result["processed_text"]

            emoji = "😊" if sentiment == "Positive" else "😞"
            card_class = "result-positive" if sentiment == "Positive" else "result-negative"

            # Display sentiment result
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="sentiment-label">{sentiment} {emoji}</div>
                <div class="confidence-label">Confidence: {confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Confidence progress bar
            st.markdown("**Confidence Score**")
            st.progress(confidence)

            # Processed text
            st.markdown("")
            with st.expander("🔧 View Processed Text", expanded=False):
                st.markdown(f"""
                <div class="processed-text">
                    {processed}
                </div>
                """, unsafe_allow_html=True)
                st.caption(
                    "This is the cleaned text after applying the NLP preprocessing pipeline: "
                    "lowercasing, HTML/URL/punctuation removal, tokenization, stop-word removal, "
                    "and lemmatization."
                )

        elif analyze_clicked:
            st.warning("Please enter some text to analyze.")
        else:
            st.info("Enter a review and click **Analyze Sentiment** to see the result.")


# ==========================================================================
# Tab 2: Batch Analysis (CSV Upload)
# ==========================================================================
with tab2:
    st.markdown("### 📁 Batch Sentiment Analysis")
    st.markdown(
        "Upload a CSV file containing a column of text/reviews. "
        "The app will predict the sentiment for each row and let you download the results."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="The CSV must contain a column with text data.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df)} rows** with columns: `{', '.join(df.columns)}`")

            # Let user pick the text column
            text_column = st.selectbox(
                "Select the column containing text/reviews:",
                options=df.columns.tolist(),
            )

            if st.button("🚀 Run Batch Analysis", type="primary"):
                with st.spinner(f"Analyzing {len(df)} reviews..."):
                    result_df = predictor.predict_dataframe(df, text_column=text_column)

                st.success(f"✅ Completed! Analyzed {len(result_df)} reviews.")

                # Show summary statistics
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                positive_count = (result_df["Predicted Sentiment"] == "Positive").sum()
                negative_count = (result_df["Predicted Sentiment"] == "Negative").sum()
                avg_confidence = result_df["Confidence"].mean()

                with summary_col1:
                    st.metric("😊 Positive", f"{positive_count} ({positive_count/len(result_df)*100:.1f}%)")
                with summary_col2:
                    st.metric("😞 Negative", f"{negative_count} ({negative_count/len(result_df)*100:.1f}%)")
                with summary_col3:
                    st.metric("📊 Avg Confidence", f"{avg_confidence*100:.1f}%")

                # Show results table
                st.dataframe(
                    result_df[[text_column, "Predicted Sentiment", "Confidence"]],
                    use_container_width=True,
                    height=400,
                )

                # Download button
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
