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

from security.rls import RLSEngine, SecurityContext, SecurityRole

async def log_visualization_request(
    session_id: str,
    prompt: str,
    status: str,
    expressions_count: int = 0,
    processing_time_ms: float = 0.0,
    error_message: Optional[str] = None,
    context: Optional[SecurityContext] = None
) -> int:
    """Logs a visualization request to the database asynchronously under RLS write policy."""
    sec_ctx = context or SecurityContext(session_id=session_id)
    record = RLSEngine.apply_insert_policy(
        "visualization_logs",
        sec_ctx,
        {
            "session_id": session_id,
            "prompt": prompt,
            "status": status,
            "expressions_count": expressions_count,
            "processing_time_ms": processing_time_ms,
            "error_message": error_message
        }
    )

    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            """
            INSERT INTO visualization_logs (session_id, prompt, status, expressions_count, processing_time_ms, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record["session_id"],
                record["prompt"],
                record["status"],
                record["expressions_count"],
                record["processing_time_ms"],
                record["error_message"]
            )
        )
        await db.commit()
        return cursor.lastrowid

async def get_visualization_logs(
    session_id: Optional[str] = None,
    limit: int = 50,
    context: Optional[SecurityContext] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves visualization logs asynchronously under Role-Level Security (RLS) policy.
    Users can only access their own session logs unless system admin role is present.
    """
    sec_ctx = context or SecurityContext(session_id=session_id or "anonymous")
    base_sql = "SELECT * FROM visualization_logs ORDER BY created_at DESC LIMIT ?"
    scoped_sql, params = RLSEngine.apply_read_policy(
        "visualization_logs", sec_ctx, base_sql, [limit]
    )

    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(scoped_sql, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
