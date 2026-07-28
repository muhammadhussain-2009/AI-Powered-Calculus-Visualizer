import sqlite3
import os

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./calculus_vis.db").replace("sqlite:///", "")

def get_secure_db_connection():
    """
    Returns a secure SQLite database connection with parameterized query safety.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes database schema securely.
    """
    conn = get_secure_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
