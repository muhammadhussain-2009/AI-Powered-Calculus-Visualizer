import sqlite3
import os
from typing import List, Dict, Any, Optional
from security.rls import RLSEngine, SecurityContext, SecurityRole

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

def log_user_usage(session_id: str, prompt: str, context: Optional[SecurityContext] = None):
    """
    Logs user prompt usage under Role-Level Security (RLS) policies.
    """
    sec_ctx = context or SecurityContext(session_id=session_id)
    record = RLSEngine.apply_insert_policy(
        "usage_logs", sec_ctx, {"session_id": session_id, "prompt": prompt}
    )
    conn = get_secure_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO usage_logs (session_id, prompt) VALUES (?, ?)",
        (record["session_id"], record["prompt"])
    )
    conn.commit()
    conn.close()

def get_user_usage_logs(session_id: str, limit: int = 50, context: Optional[SecurityContext] = None) -> List[Dict[str, Any]]:
    """
    Retrieves user usage logs restricted strictly to the user's session_id via RLS policy.
    """
    sec_ctx = context or SecurityContext(session_id=session_id)
    base_sql = "SELECT * FROM usage_logs ORDER BY timestamp DESC LIMIT ?"
    scoped_sql, params = RLSEngine.apply_read_policy("usage_logs", sec_ctx, base_sql, [limit])
    
    conn = get_secure_db_connection()
    cursor = conn.cursor()
    cursor.execute(scoped_sql, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
