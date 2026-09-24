# Person A Handoff Document

This document serves as the official handoff from **Person A (Data, Preprocessing & Model Training)** to **Person B (Backend/Persistence)** and **Person C (Frontend/QA)**.

## 1. For Person B (Backend)
All required artifacts have been successfully built, trained, and saved to the `models/` directory.

### **Available Models**
You can load the following models using `joblib.load()`:
- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/svm.joblib`

For the Hardcore tier, the following are available:
- `models/lstm.pt` (PyTorch state dict)
- `models/bert/` (Hugging Face pretrained directory containing model and tokenizer)

### **Vectorizer**
The fitted TF-IDF vectorizer (15,000 max features) is saved at:
- `models/tfidf_vectorizer.joblib`

### **Preprocessing Contract**
Before running any user text through the vectorizer at inference time, you **must** pass it through the preprocessing pipeline exactly as it was done during training. 

Import the signature from `src/preprocessing.py`:
```python
from src.preprocessing import preprocess

def preprocess(raw_text: str) -> str:
    # Returns space-separated string of cleaned, tokenized, lemmatized text without stopwords.
    # Returns "insufficient content to analyze" if the text is empty after cleaning.
    ...
```

---

## 2. For Person C (Frontend & QA)
All evaluation metrics have been generated and serialized so you can dynamically load them into the Streamlit dashboard and the README.

### **Metrics & Visualizations**
For every model trained, you will find a corresponding JSON file containing exact `accuracy`, `precision`, `recall`, and `f1` scores, alongside a `.png` of the confusion matrix. 
- Example: `models/random_forest_metrics.json`
- Example: `models/random_forest_confusion_matrix.png`

### **EDA Artifacts**
- `notebooks/eda.ipynb` contains the exploratory data analysis.
- Visualizations like `notebooks/plots/text_length.png` are available for the analytics page.

### **Documented Known Limitations (For Chapter 25)**
When writing the QA and Limitations section, please note:
1. **Satire Misclassification:** Because the preprocessing pipeline aggressively strips URLs, HTML, and punctuation, highly stylized text (like satire or sarcasm) will likely lose its contextual nuance and be misclassified.
2. **Hardcore Tier Sample Size:** Due to hardware constraints (CPU-only environment), the LSTM and BERT models were trained on a 2% aggressively downsampled dataset. Their metrics should be treated as a proof-of-concept rather than production-ready indicators. The classical machine learning models (Random Forest, Logistic Regression) trained on the full dataset and are highly performant (~99%).
