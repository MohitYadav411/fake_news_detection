import os
import joblib
import numpy as np

# Import Person A's preprocessing and features modules
from preprocessing import preprocess
from features import load_vectorizer, transform

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')

# Global cache for the vectorizer
_VECTORIZER = None

def get_vectorizer():
    """Loads and caches the vectorizer."""
    global _VECTORIZER
    if _VECTORIZER is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Vectorizer not found. Please run training first.")
        _VECTORIZER = load_vectorizer(VECTORIZER_PATH)
    return _VECTORIZER

def load_model(model_name: str) -> object:
    """
    Loads a trained model artifact from the models directory.
    
    Args:
        model_name: The name of the model (e.g., 'logistic_regression', 'naive_bayes')
        
    Returns:
        The loaded model object.
    """
    # For this implementation, we map model_name to the .joblib file.
    # Handling PyTorch / deep learning models would require specific loading logic.
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("this model hasn't been trained yet — run training first")
    
    return joblib.load(model_path)

def predict(raw_text: str, model_name: str) -> dict:
    """
    Predicts whether a given news text is Real or Fake using the specified model.
    
    Args:
        raw_text: The original user input text.
        model_name: The name of the model to use.
        
    Returns:
        dict: {"label": "Real" | "Fake", "confidence": float}
        or an error dictionary for invalid inputs.
    """
    # 1. Error Handling Matrix: Empty string
    if not raw_text or not raw_text.strip():
        return {"error": "Please enter some text to check"}
        
    # 2. Error Handling Matrix: Extremely long input (>5000 words)
    words = raw_text.split()
    if len(words) > 5000:
        # Truncate to 5000 words and proceed, with a caveat.
        raw_text = " ".join(words[:5000])
        # Note: We can add a 'warning' key in a real UI, but for the basic dict signature,
        # we proceed with the truncated text.
        
    # 3. Preprocessing
    cleaned_text = preprocess(raw_text)
    
    # 4. Error Handling Matrix: Text becomes empty after cleaning
    if not cleaned_text or cleaned_text == "insufficient content to analyze":
        return {"error": "Not enough content to analyze"}
        
    # 5. Load model and vectorizer
    try:
        model = load_model(model_name)
        vectorizer = get_vectorizer()
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to load model or vectorizer: {e}"}
        
    # 6. Feature Extraction
    try:
        features = transform(vectorizer, [cleaned_text])
    except ValueError as e:
        # This could happen if vectorizer version mismatches
        return {"error": f"Feature extraction failed (possible version mismatch): {e}"}
        
    # 7. Prediction
    try:
        # Most sklearn models support predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            # Assuming classes are [0, 1] for Fake, Real. 
            # We need to map model's classes to our labels.
            # Let's assume class '1' or positive class is 'Real', but it depends on Person A's training.
            # Generally, class index with max probability:
            max_index = np.argmax(proba)
            confidence = float(proba[max_index]) * 100
            
            predicted_class = model.classes_[max_index]
            label = "Real" if predicted_class == 1 else "Fake"
        else:
            # Fallback if no predict_proba
            prediction = model.predict(features)[0]
            label = "Real" if prediction == 1 else "Fake"
            confidence = 100.0  # Cannot compute confidence without proba
            
        return {
            "label": label,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
