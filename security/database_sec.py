import os
import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy import text, select
from security.rls import RLSEngine, SecurityContext, SecurityRole
from backend.database import get_session_factory, get_engine, UsageLog, ENCRYPTION_KEY, init_db_async, is_postgres_active

async def init_db_sec_async():
    """Initializes security usage schema asynchronously in PostgreSQL."""
    await init_db_async()

def init_db():
    """Synchronous wrapper for schema initialization."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(init_db_sec_async())
    finally:
        loop.close()

async def log_user_usage_async(
    session_id: str,
    prompt: str,
    jwt_token: Optional[str] = None,
    context: Optional[SecurityContext] = None
) -> int:
    """
    Logs user prompt usage in PostgreSQL under Role-Level Security (RLS) policies.
    Encrypts JWT token / sensitive data at rest using pgcrypto.
    """
    sec_ctx = context or SecurityContext(session_id=session_id)
    record = RLSEngine.apply_insert_policy(
        "usage_logs", sec_ctx, {"session_id": session_id, "prompt": prompt}
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        async with session.begin():
            if is_postgres_active():
                await session.execute(
                    text("SET LOCAL app.current_session_id = :session_id"),
                    {"session_id": sec_ctx.session_id}
                )
                await session.execute(
                    text("SET LOCAL app.is_admin = :is_admin"),
                    {"is_admin": "true" if sec_ctx.is_admin() else "false"}
                )

                if jwt_token:
                    query = text("""
                        INSERT INTO usage_logs (session_id, prompt, encrypted_jwt_token)
                        VALUES (:session_id, :prompt, pgp_sym_encrypt(:jwt_token, :enc_key))
                        RETURNING id
                    """)
                    params = {
                        "session_id": record["session_id"],
                        "prompt": record["prompt"],
                        "jwt_token": jwt_token,
                        "enc_key": ENCRYPTION_KEY
                    }
                else:
                    query = text("""
                        INSERT INTO usage_logs (session_id, prompt)
                        VALUES (:session_id, :prompt)
                        RETURNING id
                    """)
                    params = {
                        "session_id": record["session_id"],
                        "prompt": record["prompt"]
                    }
                result = await session.execute(query, params)
                log_id = result.scalar()
                return log_id or 0
            else:
                log_item = UsageLog(
                    session_id=record["session_id"],
                    prompt=record["prompt"],
                    encrypted_jwt_token=jwt_token.encode() if jwt_token else None
                )
                session.add(log_item)
                await session.flush()
                return log_item.id

def log_user_usage(
    session_id: str,
    prompt: str,
    jwt_token: Optional[str] = None,
    context: Optional[SecurityContext] = None
):
    """Synchronous wrapper for log_user_usage_async."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(log_user_usage_async(session_id, prompt, jwt_token, context))
    finally:
        loop.close()

async def get_user_usage_logs_async(
    session_id: str,
    limit: int = 50,
    context: Optional[SecurityContext] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves user usage logs from PostgreSQL restricted strictly to the user's session_id via RLS policy.
    """
    sec_ctx = context or SecurityContext(session_id=session_id)
    session_factory = get_session_factory()

    async with session_factory() as session:
        if is_postgres_active():
            await session.execute(
                text("SET LOCAL app.current_session_id = :session_id"),
                {"session_id": sec_ctx.session_id}
            )
            await session.execute(
                text("SET LOCAL app.is_admin = :is_admin"),
                {"is_admin": "true" if sec_ctx.is_admin() else "false"}
            )

            if sec_ctx.is_admin():
                query = text("SELECT id, session_id, prompt, timestamp FROM usage_logs ORDER BY timestamp DESC LIMIT :limit")
                params = {"limit": limit}
            else:
                query = text("SELECT id, session_id, prompt, timestamp FROM usage_logs WHERE session_id = :session_id ORDER BY timestamp DESC LIMIT :limit")
                params = {"session_id": sec_ctx.session_id, "limit": limit}

            result = await session.execute(query, params)
            rows = result.mappings().all()
            return [dict(row) for row in rows]
        else:
            stmt = select(UsageLog)
            if not sec_ctx.is_admin():
                stmt = stmt.where(UsageLog.session_id == sec_ctx.session_id)
            stmt = stmt.order_by(UsageLog.timestamp.desc()).limit(limit)

            res = await session.execute(stmt)
            logs = res.scalars().all()
            return [
                {
                    "id": log.id,
                    "session_id": log.session_id,
                    "prompt": log.prompt,
                    "timestamp": str(log.timestamp) if log.timestamp else None
                }
                for log in logs
            ]

def get_user_usage_logs(
    session_id: str,
    limit: int = 50,
    context: Optional[SecurityContext] = None
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for get_user_usage_logs_async."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_user_usage_logs_async(session_id, limit, context))
    finally:
        loop.close()
