import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'app.db')

def get_connection():
    """Returns a connection to the SQLite database."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database schema if it does not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # USER table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS USER (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    
    # Insert a default admin user if not exists
    cursor.execute("SELECT COUNT(*) FROM USER WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO USER (username, role) VALUES ('admin', 'admin')")

    # NEWS_QUERY table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NEWS_QUERY (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT NOT NULL,
            cleaned_text TEXT,
            submitted_at DATETIME NOT NULL,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES USER(user_id)
        )
    ''')

    # FEATURE_VECTOR table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FEATURE_VECTOR (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            vector_ref TEXT,
            FOREIGN KEY (query_id) REFERENCES NEWS_QUERY(query_id)
        )
    ''')

    # MODEL table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS MODEL (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            tier TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            trained_at DATETIME,
            artifact_path TEXT
        )
    ''')

    # PREDICTION_RESULT table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PREDICTION_RESULT (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (query_id) REFERENCES NEWS_QUERY(query_id),
            FOREIGN KEY (model_id) REFERENCES MODEL(model_id)
        )
    ''')

    # HISTORY_LOG table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS HISTORY_LOG (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL,
            logged_at DATETIME NOT NULL,
            FOREIGN KEY (result_id) REFERENCES PREDICTION_RESULT(result_id)
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
