import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess
from src.features import fit_vectorizer, save_vectorizer

def main():
    print("Loading raw data...")
    fake_df = pd.read_csv('data/raw/Fake.csv')
    true_df = pd.read_csv('data/raw/True.csv')
    
    fake_df['label'] = 1  # 1 for Fake
    true_df['label'] = 0  # 0 for Real
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # 70/30 stratified split (as per Phase 3 docs)
    print("Splitting into train and test sets (70/30 stratified)...")
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Preprocess text
    # We apply preprocess to the 'text' column. (Depending on the dataset, 'title' and 'text' could be merged, but usually 'text' is sufficient. Let's merge them for better representation).
    print("Preprocessing training text (this may take a few minutes)...")
    df_train = df_train.copy()
    df_train['full_text'] = df_train['title'].fillna('') + " " + df_train['text'].fillna('')
    df_train['cleaned_text'] = df_train['full_text'].apply(preprocess)
    
    print("Preprocessing testing text...")
    df_test = df_test.copy()
    df_test['full_text'] = df_test['title'].fillna('') + " " + df_test['text'].fillna('')
    df_test['cleaned_text'] = df_test['full_text'].apply(preprocess)
    
    # Filter out empty results
    df_train = df_train[df_train['cleaned_text'] != 'insufficient content to analyze']
    df_test = df_test[df_test['cleaned_text'] != 'insufficient content to analyze']
    
    # Save processed datasets
    print("Saving processed datasets to data/processed/ ...")
    df_train[['cleaned_text', 'label']].to_csv('data/processed/train.csv', index=False)
    df_test[['cleaned_text', 'label']].to_csv('data/processed/test.csv', index=False)
    
    # Fit vectorizer
    print("Fitting TF-IDF Vectorizer...")
    # Based on EDA, vocabulary is huge. We use max_features=15000, ngram_range=(1,2), min_df=2
    tfidf_kwargs = {
        'max_features': 15000,
        'ngram_range': (1, 2),
        'min_df': 2
    }
    vectorizer = fit_vectorizer(df_train['cleaned_text'].tolist(), **tfidf_kwargs)
    
    # Save vectorizer
    print("Saving vectorizer to models/tfidf_vectorizer.joblib ...")
    save_vectorizer(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    vocab_size = len(vectorizer.vocabulary_)
    print(f"Vectorizer successfully fitted and saved. Actual vocabulary size: {vocab_size}")

if __name__ == "__main__":
    main()
