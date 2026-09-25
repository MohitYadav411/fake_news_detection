# Handoff to Person C (UI Engineer)

## What is Ready for You
Person B (Backend Engineer) has finished the database, persistence, and prediction modules for the Fake News Detection project. The codebase is now ready for you to build the Streamlit frontend.

### The API Contracts

You can import these functions directly in your Streamlit app:

#### 1. Making Predictions
```python
import sys
sys.path.append("src")
from predict import predict

# To run a prediction (it handles loading the model and vectorizer automatically)
result = predict("Text from the UI goes here...", "naive_bayes")
# Output: {"label": "Real", "confidence": 85.68} 
# Note: if there is an error (e.g. empty text), the output is {"error": "Please enter some text to check"}
```

#### 2. Logging & History
```python
from history import log_query, get_history, export_history

# Log a query after prediction
log_query(raw_text, cleaned_text, "naive_bayes", result["label"], result["confidence"])

# Get the history for the admin dashboard (returns a Pandas DataFrame)
df = get_history(limit=100)

# Export history to CSV
csv_path = export_history("data/history_export.csv")
```

### Notes for UI
- **Error Handling**: The `predict()` function will never crash the app. It returns a dictionary with an `"error"` key if something goes wrong. Ensure your Streamlit app checks for `if "error" in result:` and shows a `st.warning()` or `st.error()`.
- **Supported Models**: The `predict` function accepts model names that exist in the `models/` directory (e.g., `"naive_bayes"`, `"logistic_regression"`, `"random_forest"`). You can use a dropdown in Streamlit to let the user select these models.
- **Database**: The SQLite database (`data/app.db`) initializes automatically when `history.py` or `db.py` is called. It tracks history properly across application restarts.

Good luck! You can use the `git_push.bat` script to push your updates directly to GitHub once you're done.
