"""
app.py
Streamlit UI Module for Fake News Detection.
Implements Chapter 19 of the project documentation.
Public entry point: streamlit run app.py
"""

import streamlit as st
import os
import sys
import json
import subprocess
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict
from history import log_query, get_history, export_history
from preprocessing import preprocess
from db import init_db

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
init_db()
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

MODEL_TIERS = {
    "Basic": ["naive_bayes", "decision_tree", "logistic_regression", "albert"],
    "Intermediate": ["random_forest", "svm", "electra"],
    "Hardcore": ["lstm", "bert", "distilbert", "roberta"],
    "Ensemble (Majority Vote)": ["naive_bayes", "decision_tree", "logistic_regression", "random_forest", "svm", "lstm", "bert", "distilbert", "roberta", "albert", "electra"]
}

def load_metrics(model_name):
    metrics_path = os.path.join(BASE_DIR, "models", f"{model_name}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home (Check News)", "History", "Admin Dashboard"])

    if page == "Home (Check News)":
        st.title("📰 Fake News Detection System")
        st.write("Enter a news article or headline below to check if it's real or fake.")

        # Input
        raw_text = st.text_area("News Content", height=200, placeholder="Paste news content here...")
        
        tier = st.selectbox("Model Tier", list(MODEL_TIERS.keys()))
        models_to_run = MODEL_TIERS[tier]

        check_button = st.button("Check News", disabled=len(raw_text.strip()) == 0)

        if check_button:
            if not raw_text.strip():
                st.warning("Please enter some text to check.")
            else:
                st.subheader(f"Results for {tier} Tier")
                
                if tier == "Ensemble (Majority Vote)":
                    real_votes = 0
                    fake_votes = 0
                    results_data = []
                    
                    with st.spinner(f"Running all {len(models_to_run)} models concurrently..."):
                        for m_name in models_to_run:
                            result = predict(raw_text, m_name)
                            if "error" not in result:
                                label = result["label"]
                                if label == "Real":
                                    real_votes += 1
                                else:
                                    fake_votes += 1
                                results_data.append({"Model": m_name.replace('_', ' ').title(), "Prediction": label, "Confidence": f"{result['confidence']}%"})
                                
                                # Log query
                                cleaned_text = preprocess(raw_text)
                                log_query(raw_text, cleaned_text, m_name, label, result["confidence"])
                    
                    # Final verdict
                    if real_votes > fake_votes:
                        st.success(f"### 🎯 FINAL VERDICT: REAL NEWS ({real_votes} out of {len(models_to_run)} models agree)")
                    else:
                        st.error(f"### 🛑 FINAL VERDICT: FAKE NEWS ({fake_votes} out of {len(models_to_run)} models agree)")
                        
                    st.write("#### Detailed Breakdown:")
                    st.dataframe(results_data, use_container_width=True)
                    
                else:
                    cols = st.columns(len(models_to_run))
                    for i, m_name in enumerate(models_to_run):
                        with cols[i]:
                            st.write(f"**{m_name.replace('_', ' ').title()}**")
                            with st.spinner("Analyzing..."):
                                result = predict(raw_text, m_name)
                                if "error" in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    label = result["label"]
                                    confidence = result["confidence"]
                                    if label == "Real":
                                        st.success(f"Prediction: {label}")
                                    else:
                                        st.error(f"Prediction: {label}")
                                    st.progress(min(int(confidence) / 100.0, 1.0))
                                    st.write(f"Confidence: {confidence}%")
                                    # Log query
                                    cleaned_text = preprocess(raw_text)
                                    log_query(raw_text, cleaned_text, m_name, label, confidence)
                st.caption("Caveat: This is a decision-support tool, not a final verdict. Always verify facts independently.")

    elif page == "History":
        st.title("📜 Prediction History")
        df = get_history(limit=100)
        if not df.empty:
            st.dataframe(df)
            
            # Export CSV
            export_path = os.path.join(BASE_DIR, "data", "history_export.csv")
            if st.button("Prepare CSV for Download"):
                export_history(export_path)
                if os.path.exists(export_path):
                    with open(export_path, "rb") as f:
                        st.download_button("Download CSV", data=f, file_name="history.csv", mime="text/csv")
        else:
            st.info("No history found.")

    elif page == "Admin Dashboard":
        st.title("⚙️ Admin & Model Comparison Dashboard")
        
        # Simple Authentication
        if "admin_authenticated" not in st.session_state:
            st.session_state.admin_authenticated = False
            
        if not st.session_state.admin_authenticated:
            st.write("Please enter the admin password to access this dashboard.")
            pwd = st.text_input("Password", type="password")
            if st.button("Login"):
                if pwd == "admin123":
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
            return

        
        st.subheader("Model Performance")
        
        # Load metrics for all models
        all_metrics = []
        for tier, models in MODEL_TIERS.items():
            for m in models:
                metrics = load_metrics(m)
                if metrics:
                    metrics["model"] = m
                    metrics["tier"] = tier
                    all_metrics.append(metrics)
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            # Reorder columns
            metrics_df = metrics_df[["tier", "model", "accuracy", "precision", "recall", "f1"]]
            st.dataframe(metrics_df)
        else:
            st.warning("No metrics found. Have the models been trained?")

        st.subheader("Confusion Matrices")
        matrix_model = st.selectbox("Select model to view confusion matrix", [m["model"] for m in all_metrics] if all_metrics else [])
        if matrix_model:
            img_path = os.path.join(BASE_DIR, "models", f"{matrix_model}_confusion_matrix.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{matrix_model} Confusion Matrix")
            else:
                st.info(f"No confusion matrix image found for {matrix_model}.")

        st.subheader("Word Clouds (Fake vs Real News)")
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            true_wc_path = os.path.join(BASE_DIR, "models", "wordcloud_true.png")
            if os.path.exists(true_wc_path):
                st.image(true_wc_path, caption="Real News Word Cloud", use_container_width=True)
            else:
                st.info("True news word cloud not found.")
        with col_wc2:
            fake_wc_path = os.path.join(BASE_DIR, "models", "wordcloud_fake.png")
            if os.path.exists(fake_wc_path):
                st.image(fake_wc_path, caption="Fake News Word Cloud", use_container_width=True)
            else:
                st.info("Fake news word cloud not found.")

        st.subheader("Retrain Models")
        st.write("Add labeled data and trigger retraining.")
        if st.button("Trigger Retraining"):
            with st.spinner("Retraining basic models in progress..."):
                process = subprocess.run(
                    [sys.executable, os.path.join(BASE_DIR, "src", "train.py"), "--tier", "all"],
                    cwd=BASE_DIR, capture_output=True, text=True
                )
                if process.returncode == 0:
                    st.success("Retraining completed successfully.")
                else:
                    st.error("Retraining failed. Check that valid training data and dependencies are installed.")
                    st.code(process.stderr or process.stdout)

if __name__ == "__main__":
    main()
