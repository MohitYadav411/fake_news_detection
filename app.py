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
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict
from history import log_query, get_history, export_history
from preprocessing import preprocess

# Configuration
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

MODEL_TIERS = {
    "Basic": ["naive_bayes", "decision_tree", "logistic_regression"],
    "Intermediate": ["random_forest", "svm"],
    "Hardcore": ["lstm", "bert"]
}

def load_metrics(model_name):
    metrics_path = os.path.join("models", f"{model_name}_metrics.json")
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
        
        col1, col2 = st.columns(2)
        with col1:
            tier = st.selectbox("Model Tier", list(MODEL_TIERS.keys()))
        with col2:
            model_name = st.selectbox("Specific Model", MODEL_TIERS[tier])

        check_button = st.button("Check News", disabled=len(raw_text.strip()) == 0)

        if check_button:
            if not raw_text.strip():
                st.warning("Please enter some text to check.")
            else:
                with st.spinner(f"Analyzing with {model_name}..."):
                    result = predict(raw_text, model_name)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        label = result["label"]
                        confidence = result["confidence"]
                        
                        # Display Result
                        st.subheader("Analysis Result")
                        if label == "Real":
                            st.success(f"**Prediction: {label}**")
                        else:
                            st.error(f"**Prediction: {label}**")
                            
                        # Use min to ensure we don't pass > 1.0 to st.progress
                        st.progress(min(int(confidence) / 100.0, 1.0))
                        st.write(f"**Confidence Score:** {confidence}%")
                        st.caption("Caveat: This is a decision-support tool, not a final verdict. Always verify facts independently.")

                        # Log query
                        cleaned_text = preprocess(raw_text)
                        log_query(raw_text, cleaned_text, model_name, label, confidence)

    elif page == "History":
        st.title("📜 Prediction History")
        df = get_history(limit=100)
        if not df.empty:
            st.dataframe(df)
            
            # Export CSV
            export_path = os.path.join("data", "history_export.csv")
            if st.button("Prepare CSV for Download"):
                export_history(export_path)
                if os.path.exists(export_path):
                    with open(export_path, "rb") as f:
                        st.download_button("Download CSV", data=f, file_name="history.csv", mime="text/csv")
        else:
            st.info("No history found.")

    elif page == "Admin Dashboard":
        st.title("⚙️ Admin & Model Comparison Dashboard")
        
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
            img_path = os.path.join("models", f"{matrix_model}_confusion_matrix.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{matrix_model} Confusion Matrix")
            else:
                st.info(f"No confusion matrix image found for {matrix_model}.")

        st.subheader("Retrain Models")
        st.write("Add labeled data and trigger retraining.")
        if st.button("Trigger Retraining"):
            with st.spinner("Retraining basic models in progress..."):
                os.system(f"{sys.executable} src/train.py")
                st.success("Retraining triggered successfully! (Basic/Intermediate models updated)")

if __name__ == "__main__":
    main()
