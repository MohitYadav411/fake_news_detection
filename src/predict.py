import os
import joblib
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import Person A's preprocessing and features modules
from preprocessing import preprocess
from features import load_vectorizer, transform
from train_hardcore import LSTMClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')

# Global cache
_VECTORIZER = None
_MODELS = {}
_TOKENIZERS = {}

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
    """
    global _MODELS, _TOKENIZERS
    if model_name in _MODELS:
        return _MODELS[model_name]

    if model_name == "lstm":
        model_path = os.path.join(MODELS_DIR, "lstm.pt")
        vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.json")
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError("this model hasn't been trained yet — run training first")
        
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
            
        model = LSTMClassifier(len(vocab))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        _MODELS[model_name] = {"model": model, "vocab": vocab}
        return _MODELS[model_name]
        
    elif model_name in ["bert", "distilbert", "roberta", "albert", "electra"]:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError("this model hasn't been trained yet — run training first")
            
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        _MODELS[model_name] = {"model": model, "tokenizer": tokenizer}
        return _MODELS[model_name]

    # scikit-learn models
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("this model hasn't been trained yet — run training first")
    
    _MODELS[model_name] = joblib.load(model_path)
    return _MODELS[model_name]

def predict(raw_text: str, model_name: str) -> dict:
    if not raw_text or not raw_text.strip():
        return {"error": "Please enter some text to check"}
        
    words = raw_text.split()
    if len(words) > 5000:
        raw_text = " ".join(words[:5000])
        
    cleaned_text = preprocess(raw_text)
    
    if not cleaned_text or cleaned_text == "insufficient content to analyze":
        return {"error": "Not enough content to analyze"}
        
    try:
        model_obj = load_model(model_name)
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}
        
    if model_name == "lstm":
        model = model_obj["model"]
        vocab = model_obj["vocab"]
        
        seq = [vocab.get(word, vocab['<UNK>']) for word in cleaned_text.split()]
        max_len = 100
        if len(seq) < max_len:
            seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
            
        input_tensor = torch.tensor([seq], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = float(output)
            
        label = "Fake" if prob >= 0.5 else "Real"
        confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
        
        return {
            "label": label,
            "confidence": round(confidence, 2)
        }
        
    elif model_name in ["bert", "distilbert", "roberta", "albert", "electra"]:
        model = model_obj["model"]
        tokenizer = model_obj["tokenizer"]
        
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
        prob_fake = float(probs[1])
        prob_real = float(probs[0])
        
        label = "Fake" if prob_fake >= 0.5 else "Real"
        confidence = prob_fake * 100 if prob_fake >= 0.5 else prob_real * 100
        
        return {
            "label": label,
            "confidence": round(confidence, 2)
        }
    
    # scikit-learn models
    try:
        vectorizer = get_vectorizer()
        features = transform(vectorizer, [cleaned_text])
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}
        
    try:
        model = model_obj
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            max_index = np.argmax(proba)
            confidence = float(proba[max_index]) * 100
            predicted_class = model.classes_[max_index]
            # fake_df['label'] = 1, true_df['label'] = 0 -> 1 is Fake, 0 is Real
            label = "Fake" if predicted_class == 1 else "Real"
        else:
            prediction = model.predict(features)[0]
            label = "Fake" if prediction == 1 else "Real"
            confidence = 100.0
            
        return {
            "label": label,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
