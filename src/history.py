import pandas as pd
from datetime import datetime
from db import get_connection

def log_query(raw_text: str, cleaned_text: str, model_name: str, label: str, confidence: float) -> int:
    """
    Logs a prediction query and its result to the database.
    
    Args:
        raw_text: The original user input text.
        cleaned_text: The text after preprocessing.
        model_name: The name of the model used for prediction.
        label: The predicted label ('Real' or 'Fake').
        confidence: The confidence score of the prediction (0-100).
        
    Returns:
        int: The ID of the newly created history log.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        now = datetime.utcnow()
        
        # 1. Ensure model exists in MODEL table or fetch its ID
        cursor.execute("SELECT model_id FROM MODEL WHERE name = ?", (model_name,))
        model_row = cursor.fetchone()
        if model_row:
            model_id = model_row[0]
        else:
            # For simplicity, if model is not registered, insert it
            cursor.execute("INSERT INTO MODEL (name, tier) VALUES (?, ?)", (model_name, 'unknown'))
            model_id = cursor.lastrowid
            
        # 2. Insert into NEWS_QUERY
        # Truncating raw_text for safety if it's too long, but spec says "truncated for display", 
        # storing the full text here unless it's extremely long, but let's just store it as is for now.
        cursor.execute(
            "INSERT INTO NEWS_QUERY (raw_text, cleaned_text, submitted_at) VALUES (?, ?, ?)",
            (raw_text, cleaned_text, now)
        )
        query_id = cursor.lastrowid
        
        # 3. Insert into PREDICTION_RESULT
        cursor.execute(
            "INSERT INTO PREDICTION_RESULT (query_id, model_id, predicted_label, confidence) VALUES (?, ?, ?, ?)",
            (query_id, model_id, label, confidence)
        )
        result_id = cursor.lastrowid
        
        # 4. Insert into HISTORY_LOG
        cursor.execute(
            "INSERT INTO HISTORY_LOG (result_id, logged_at) VALUES (?, ?)",
            (result_id, now)
        )
        log_id = cursor.lastrowid
        
        conn.commit()
        return log_id
    except Exception as e:
        conn.rollback()
        # Logging failure shouldn't crash the app, but returning -1 to indicate failure
        print(f"Error logging query: {e}")
        return -1
    finally:
        conn.close()

def get_history(limit: int = 100) -> pd.DataFrame:
    """
    Retrieves the prediction history.
    
    Args:
        limit: The maximum number of recent logs to retrieve.
        
    Returns:
        pandas.DataFrame: The retrieved history log containing query, label, confidence, model, and timestamp.
    """
    conn = get_connection()
    
    query = '''
        SELECT 
            nq.raw_text,
            pr.predicted_label,
            pr.confidence,
            m.name AS model_name,
            hl.logged_at
        FROM HISTORY_LOG hl
        JOIN PREDICTION_RESULT pr ON hl.result_id = pr.result_id
        JOIN NEWS_QUERY nq ON pr.query_id = nq.query_id
        JOIN MODEL m ON pr.model_id = m.model_id
        ORDER BY hl.logged_at DESC
        LIMIT ?
    '''
    
    try:
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def export_history(path: str) -> str:
    """
    Exports the entire history to a CSV file.
    
    Args:
        path: The file path to save the CSV to.
        
    Returns:
        str: The path to the written CSV.
    """
    df = get_history(limit=-1) # -1 means no limit or we just fetch all
    if df.empty:
        # If there's no data or an error occurred, we should still write an empty file or headers
        df = pd.DataFrame(columns=['raw_text', 'predicted_label', 'confidence', 'model_name', 'logged_at'])
    
    df.to_csv(path, index=False)
    return path
