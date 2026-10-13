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
    page = st.sidebar.radio("Go to", ["Home (Check News)", "History"])

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



if __name__ == "__main__":
    main()
