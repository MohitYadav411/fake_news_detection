# Fake News Detection System

This project, titled "Fake News Detection," presents a data science and machine learning based solution to automatically classify a given piece of news content — a headline, article, or social media post — as either **Real** or **Fake**, along with a confidence score indicating the model's certainty.

## Setup & Installation

1. **Clone the repository** and navigate to the project root.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Initialize the Database**:
   ```bash
   python src/db.py
   ```

## Dataset Source
The model is trained on the **Kaggle "Fake and Real News Dataset"** (or an equivalent labeled dataset). To reproduce training, place the raw CSV files (`True.csv` and `Fake.csv`) into the `data/` directory before running the training scripts. Do not commit large datasets to version control.

## Training Models

Models are divided into three tiers: **Basic**, **Intermediate**, and **Hardcore**.

1. **Train Basic & Intermediate Models** (Naive Bayes, Decision Tree, Logistic Regression, Random Forest, SVM):
   ```bash
   python src/train.py
   ```
   This will train the models, save their `.joblib` artifacts to `models/`, and generate metrics/confusion matrices.

2. **Train Hardcore Models** (LSTM, BERT):
   ```bash
   python src/train_hardcore.py
   ```
   *Note: Deep learning models (especially BERT) require a GPU for reasonable training times.*

## Running the Application

To launch the Streamlit UI, run:
```bash
streamlit run app.py
```
This will open the application in your browser where you can:
- Check news via text input using different model tiers.
- View your prediction history and export it to CSV.
- Access the Admin Dashboard to compare model metrics and view confusion matrices.

## Model Performance Metrics

The following metrics are derived directly from the trained models evaluated on the test set:

| Tier | Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Basic | Naive Bayes | 95.55% | 95.68% | 95.81% | 95.74% |
| Basic | Decision Tree | 99.61% | 99.81% | 99.43% | 99.62% |
| Basic | Logistic Regression | 99.26% | 99.32% | 99.28% | 99.30% |
| Intermediate | Random Forest | 99.63% | 99.77% | 99.52% | 99.64% |
| Intermediate | SVM | 99.41% | 99.42% | 99.45% | 99.43% |
| Hardcore | LSTM | 53.70% | 54.71% | 65.96% | 59.81% |
| Hardcore | BERT | 99.26% | 99.29% | 99.29% | 99.29% |

*(Note: The LSTM model was trained for a very short duration or on a small subset for demonstration, leading to lower performance. For production, train on a GPU for more epochs.)*

## Known Limitations

- **Satire and Irony**: The model analyzes lexical features and struggles to distinguish between malicious fake news and intentional satire (e.g., The Onion).
- **Out of Domain Data**: The model is trained on a specific snapshot of historical political and world news. It may perform poorly on completely novel topics (e.g., breaking scientific discoveries) or non-English content.
- **Not a Final Verdict**: This is a **decision-support tool**, not an arbiter of absolute truth. It detects *patterns* associated with fake news, but cannot independently verify facts in real-time against external sources. Always verify claims independently.
- **Length Constraint**: Extremely short inputs ("ok", "yes") lack sufficient context to analyze.

---
*Built as a final-year Bachelor's (BCA) data science project submitted to Gujarat Technological University.*
*By Mohit, Shubham, Karan*
