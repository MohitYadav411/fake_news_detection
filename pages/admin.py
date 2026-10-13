import streamlit as st
import pandas as pd
import os
import json
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    st.stop()

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
