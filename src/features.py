import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

def fit_vectorizer(corpus: list[str], **tfidf_kwargs) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectorizer on the given corpus.
    """
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    vectorizer.fit(corpus)
    return vectorizer

def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """
    Saves the fitted vectorizer to disk using joblib.
    """
    joblib.dump(vectorizer, path)

def load_vectorizer(path: str) -> TfidfVectorizer:
    """
    Loads a fitted vectorizer from disk.
    """
    return joblib.load(path)

def transform(vectorizer: TfidfVectorizer, texts: list[str]) -> scipy.sparse.csr_matrix:
    """
    Transforms a list of texts into a sparse TF-IDF matrix using the fitted vectorizer.
    """
    return vectorizer.transform(texts)
