import aiosqlite
import os
import time
from typing import List, Dict, Any, Optional

DB_FILE = os.getenv("DATABASE_FILE", "calculus_vis.db")

async def get_db():
    """Context manager / generator for async SQLite connection."""
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON;")
        yield db

async def init_db_async():
    """Initializes the database schema asynchronously."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS visualization_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL,
                expressions_count INTEGER DEFAULT 0,
                processing_time_ms REAL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rate_limit_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_ip TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await db.commit()

async def log_visualization_request(
    session_id: str,
    prompt: str,
    status: str,
    expressions_count: int = 0,
    processing_time_ms: float = 0.0,
    error_message: Optional[str] = None
) -> int:
    """Logs a visualization request to the database asynchronously."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            """
            INSERT INTO visualization_logs (session_id, prompt, status, expressions_count, processing_time_ms, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, prompt, status, expressions_count, processing_time_ms, error_message)
        )
        await db.commit()
        return cursor.lastrowid

async def get_visualization_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent visualization logs asynchronously."""
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM visualization_logs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
