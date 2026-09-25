import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK datasets are downloaded silently
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(raw_text: str) -> str:
    """
    Lowercases the text, removes HTML tags, URLs, and punctuation.
    Returns the cleaned string.
    """
    if not isinstance(raw_text, str):
        return ""
        
    text = raw_text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(cleaned_text: str) -> list[str]:
    """
    Tokenizes text into a list of words.
    """
    if not cleaned_text.strip():
        return []
    return word_tokenize(cleaned_text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes common English stopwords.
    """
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatizes tokens (converts to base dictionary form).
    """
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess(raw_text: str) -> str:
    """
    Single public entry point chaining the preprocessing steps.
    Returns the final preprocessed text as a space-separated string, 
    or a fallback message if the resulting content is empty.
    """
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmatized = lemmatize(tokens_no_stop)
    
    result = " ".join(lemmatized)
    
    if not result.strip():
        return "insufficient content to analyze"
        
    return result
