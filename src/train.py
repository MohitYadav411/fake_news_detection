import argparse
import json
import joblib
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import sys
import time

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluate import compute_metrics, plot_confusion_matrix
from src.features import load_vectorizer, transform

def train_model(model_name: str, X_train, y_train) -> object:
    """
    Initializes and trains the specified model.
    """
    # Initialize model based on name
    if model_name == "naive_bayes":
        model = MultinomialNB(alpha=0.1)
    elif model_name == "logistic_regression":
        model = LogisticRegression(C=10, max_iter=1000, class_weight='balanced')
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(max_depth=20, min_samples_split=5)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
    elif model_name == "svm":
        model = LinearSVC(C=1.0)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    # Train
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    return model

def save_model(model: object, path: str) -> None:
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, path)

def run_training_pipeline(tier: str = None) -> dict:
    """
    Trains models for the specified tier ('basic', 'intermediate', 'hardcore', or 'all').
    Returns a dictionary of all computed metrics.
    """
    print("Loading preprocessed datasets and vectorizer...")
    train_df = pd.read_csv('data/processed/train.csv').dropna(subset=['cleaned_text'])
    test_df = pd.read_csv('data/processed/test.csv').dropna(subset=['cleaned_text'])
    
    vectorizer = load_vectorizer('models/tfidf_vectorizer.joblib')
    
    X_train = transform(vectorizer, train_df['cleaned_text'].tolist())
    y_train = train_df['label'].values
    
    X_test = transform(vectorizer, test_df['cleaned_text'].tolist())
    y_test = test_df['label'].values
    
    models_to_train = []
    if tier == 'basic':
        models_to_train = ["naive_bayes", "logistic_regression"]
    elif tier == 'intermediate':
        models_to_train = ["decision_tree", "random_forest", "svm"]
    elif tier == 'hardcore':
        # Handled separately due to deep learning requirement
        print("Hardcore tier requires LSTM/BERT which are handled in specialized scripts due to hardware requirements.")
        return {}
    elif tier == 'all':
        models_to_train = ["naive_bayes", "logistic_regression", "decision_tree", "random_forest", "svm"]
    else:
        print(f"Unknown tier: {tier}")
        return {}
        
    all_metrics = {}
    
    for model_name in models_to_train:
        # Train
        model = train_model(model_name, X_train, y_train)
        
        # Save model
        model_path = f"models/{model_name}.joblib"
        save_model(model, model_path)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = compute_metrics(y_test, y_pred)
        all_metrics[model_name] = metrics
        
        # Save metrics to json
        metrics_path = f"models/{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Plot and save confusion matrix
        fig = plot_confusion_matrix(y_test, y_pred)
        fig.savefig(f"models/{model_name}_confusion_matrix.png")
        
        print(f"Results for {model_name}: {metrics}")
        
    return all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection Models")
    parser.add_argument("--tier", type=str, choices=["basic", "intermediate", "hardcore", "all"], default="basic",
                        help="Which tier of models to train (basic, intermediate, hardcore, all)")
    
    args = parser.parse_args()
    run_training_pipeline(args.tier)
