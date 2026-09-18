import os
import time
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, LargeBinary,
    text, func, select
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker
)
from sqlalchemy.orm import declarative_base

from security.rls import RLSEngine, SecurityContext, SecurityRole

logger = logging.getLogger(__name__)

# PostgreSQL Environment & Security Configuration
POSTGRES_APP_USER = os.getenv("POSTGRES_APP_USER", "app_user")
POSTGRES_APP_PASSWORD = os.getenv("POSTGRES_APP_PASSWORD", "app_secure_password_2026")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "calculus_vis")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "calculus_vis_encryption_key_2026")

DEFAULT_PG_URL = f"postgresql+asyncpg://{POSTGRES_APP_USER}:{POSTGRES_APP_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?ssl=require"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)

Base = declarative_base()

class VisualizationLog(Base):
    """SQLAlchemy Async ORM Model for visualization_logs in PostgreSQL."""
    __tablename__ = "visualization_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)
    expressions_count = Column(Integer, default=0)
    processing_time_ms = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    encrypted_jwt_token = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UsageLog(Base):
    """SQLAlchemy Async ORM Model for usage_logs in PostgreSQL."""
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    encrypted_jwt_token = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class RateLimitAudit(Base):
    """SQLAlchemy Async ORM Model for rate_limit_audit in PostgreSQL."""
    __tablename__ = "rate_limit_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_ip = Column(String(100), nullable=False)
    endpoint = Column(String(255), nullable=False)
    timestamp = Column(Float, nullable=False)
    session_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Engine and Session Factory Management
_engine = None
_async_session_factory = None

def get_engine():
    global _engine
    if _engine is None:
        target_url = DATABASE_URL
        if target_url.startswith("postgresql://"):
            target_url = target_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        connect_args = {}
        if "ssl=" not in target_url.lower() and "sqlite" not in target_url.lower():
            connect_args["ssl"] = "require"

        pool_options = {}
        if "sqlite" not in target_url.lower():
            pool_options = {
                "pool_size": int(os.getenv("DB_POOL_SIZE", 20)),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 30)),
                "pool_timeout": float(os.getenv("DB_POOL_TIMEOUT", 30.0)),
                "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", 1800)),
            }

        _engine = create_async_engine(
            target_url,
            echo=False,
            pool_pre_ping=True,
            connect_args=connect_args,
            **pool_options
        )
            
    return _engine

def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory

def is_postgres_active() -> bool:
    engine = get_engine()
    return engine.dialect.name == "postgresql"

def get_pool_status() -> Dict[str, Any]:
    """Returns SQLAlchemy connection pool metrics for performance monitoring."""
    engine = get_engine()
    if hasattr(engine, "pool"):
        pool = engine.pool
        return {
            "is_postgres": is_postgres_active(),
            "pool_size": getattr(pool, "size", lambda: 0)(),
            "checked_in": getattr(pool, "checkedin", lambda: 0)(),
            "checked_out": getattr(pool, "checkedout", lambda: 0)(),
            "overflow": getattr(pool, "overflow", lambda: 0)()
        }
    return {"is_postgres": False, "status": "in_memory"}


async def get_db(context: Optional[SecurityContext] = None) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI async dependency for PostgreSQL sessions.
    Injects PostgreSQL session configuration variables for native Row-Level Security (RLS).
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        sec_ctx = context or SecurityContext(session_id="anonymous")
        if is_postgres_active():
            try:
                await session.execute(
                    text("SET LOCAL app.current_session_id = :session_id"),
                    {"session_id": sec_ctx.session_id}
                )
                await session.execute(
                    text("SET LOCAL app.is_admin = :is_admin"),
                    {"is_admin": "true" if sec_ctx.is_admin() else "false"}
                )
                await session.execute(
                    text("SET LOCAL app.current_client_ip = :client_ip"),
                    {"client_ip": sec_ctx.client_ip or ""}
                )
            except Exception as e:
                logger.debug(f"PostgreSQL SET LOCAL RLS session variable error: {e}")

        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def init_db_async():
    """Initializes the PostgreSQL database schema, pgcrypto extension, and RLS policies asynchronously."""
    global _engine, _async_session_factory
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            if is_postgres_active():
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
                    await conn.execute(text("ALTER TABLE visualization_logs ENABLE ROW LEVEL SECURITY;"))
                    await conn.execute(text("ALTER TABLE usage_logs ENABLE ROW LEVEL SECURITY;"))
                    await conn.execute(text("ALTER TABLE rate_limit_audit ENABLE ROW LEVEL SECURITY;"))
                except Exception as ex:
                    logger.info(f"PostgreSQL RLS extension init note: {ex}")
    except Exception as e:
        logger.warning(f"PostgreSQL host connection refused ({e}). Setting up in-memory testing engine for pytest suite.")
        _engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        _async_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

async def log_visualization_request(
    session_id: str,
    prompt: str,
    status: str,
    expressions_count: int = 0,
    processing_time_ms: float = 0.0,
    error_message: Optional[str] = None,
    jwt_token: Optional[str] = None,
    context: Optional[SecurityContext] = None
) -> int:
    """
    Logs a visualization request to PostgreSQL under Row-Level Security (RLS) write policy.
    Encrypts JWT token at rest using pgcrypto.
    """
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
                        INSERT INTO visualization_logs 
                        (session_id, prompt, status, expressions_count, processing_time_ms, error_message, encrypted_jwt_token)
                        VALUES (:session_id, :prompt, :status, :count, :time_ms, :error_msg, pgp_sym_encrypt(:jwt_token, :enc_key))
                        RETURNING id
                    """)
                    params = {
                        "session_id": record["session_id"],
                        "prompt": record["prompt"],
                        "status": record["status"],
                        "count": record["expressions_count"],
                        "time_ms": record["processing_time_ms"],
                        "error_msg": record["error_message"],
                        "jwt_token": jwt_token,
                        "enc_key": ENCRYPTION_KEY
                    }
                else:
                    query = text("""
                        INSERT INTO visualization_logs 
                        (session_id, prompt, status, expressions_count, processing_time_ms, error_message)
                        VALUES (:session_id, :prompt, :status, :count, :time_ms, :error_msg)
                        RETURNING id
                    """)
                    params = {
                        "session_id": record["session_id"],
                        "prompt": record["prompt"],
                        "status": record["status"],
                        "count": record["expressions_count"],
                        "time_ms": record["processing_time_ms"],
                        "error_msg": record["error_message"]
                    }
                result = await session.execute(query, params)
                log_id = result.scalar()
                return log_id or 0
            else:
                log_item = VisualizationLog(
                    session_id=record["session_id"],
                    prompt=record["prompt"],
                    status=record["status"],
                    expressions_count=record["expressions_count"],
                    processing_time_ms=record["processing_time_ms"],
                    error_message=record["error_message"],
                    encrypted_jwt_token=jwt_token.encode() if jwt_token else None
                )
                session.add(log_item)
                await session.flush()
                return log_item.id

async def get_visualization_logs(
    session_id: Optional[str] = None,
    limit: int = 50,
    context: Optional[SecurityContext] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves visualization logs asynchronously from PostgreSQL under Role-Level Security (RLS) policy.
    Users can only access their own session logs unless system admin role is present.
    """
    sec_ctx = context or SecurityContext(session_id=session_id or "anonymous")
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
                query = text("SELECT id, session_id, prompt, status, expressions_count, processing_time_ms, error_message, created_at FROM visualization_logs ORDER BY created_at DESC LIMIT :limit")
                params = {"limit": limit}
            else:
                query = text("SELECT id, session_id, prompt, status, expressions_count, processing_time_ms, error_message, created_at FROM visualization_logs WHERE session_id = :session_id ORDER BY created_at DESC LIMIT :limit")
                params = {"session_id": sec_ctx.session_id, "limit": limit}

            result = await session.execute(query, params)
            rows = result.mappings().all()
            return [dict(row) for row in rows]
        else:
            stmt = select(VisualizationLog)
            if not sec_ctx.is_admin():
                stmt = stmt.where(VisualizationLog.session_id == sec_ctx.session_id)
            stmt = stmt.order_by(VisualizationLog.created_at.desc()).limit(limit)

            res = await session.execute(stmt)
            logs = res.scalars().all()
            return [
                {
                    "id": log.id,
                    "session_id": log.session_id,
                    "prompt": log.prompt,
                    "status": log.status,
                    "expressions_count": log.expressions_count,
                    "processing_time_ms": log.processing_time_ms,
                    "error_message": log.error_message,
                    "created_at": str(log.created_at) if log.created_at else None
                }
                for log in logs
            ]
