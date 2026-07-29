"""
Security Module: Rate Limiting System (SlowAPI Integration)

This module encapsulates all rate-limiting configuration, custom key extraction functions,
and custom exception handlers for the Calculus Visualizer FastAPI application.

It enforces strict request throttling on AI API endpoints (max 3 requests per 60 seconds)
per user session or client IP, regardless of overall daily usage quotas.
"""

import os
import json
import jwt
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Secret key and algorithm used to decode JWT bearer session tokens for rate limit keying
JWT_SECRET = os.getenv("JWT_SECRET", "calculus-visualizer-secret-key-change-in-prod-2026")
JWT_ALGORITHM = "HS256"

# Rate Limit Definition for AI API Endpoints (3 requests in 60 seconds)
AI_API_RATE_LIMIT = "3/60seconds"

def get_user_rate_limit_key(request: Request) -> str:
    """
    Extracts a unique rate-limiting key for the incoming request.
    
    Priority:
    1. User Session ID derived from JWT Bearer Authorization header.
    2. Fallback to client remote IP address (get_remote_address) if unauthenticated.
    
    This session-aware keying prevents users from bypassing IP limits by rotating IPs
    or opening new browser tabs, and ensures rate limits are enforced strictly per user.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
            session_id = payload.get("session_id")
            if session_id:
                return f"user_session:{session_id}"
        except Exception:
            # On invalid/expired token, fallback gracefully to client IP
            pass

    return get_remote_address(request)


# Initialize SlowAPI Limiter instance with session-aware key function
limiter = Limiter(
    key_func=get_user_rate_limit_key,
    default_limits=["60 per minute"]
)


async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom exception handler for RateLimitExceeded exceptions across all endpoints.
    
    Blocks or throttles requests exceeding 3 requests in 60 seconds on AI endpoints and returns a clear,
    informative rate-limit message.
    
    Supports:
    - Standard JSON response (status code 429) for REST endpoints.
    - Server-Sent Events (SSE) stream (status code 429) for streaming endpoints (/api/visualize/stream).
    """
    # Explicit human-readable rate limit message requested by system specification
    error_msg = (
        "Rate limit exceeded: You have submitted more than 3 requests in 60 seconds. "
        "Please wait 60 seconds before submitting another request."
    )

    # Deliver SSE structured payload if the request targets streaming endpoints or accepts text/event-stream
    if "/api/visualize/stream" in request.url.path or request.headers.get("accept") == "text/event-stream":
        async def sse_rate_limit_generator():
            event_data = json.dumps({
                "type": "error",
                "error": error_msg,
                "detail": str(exc.detail),
                "retry_after_seconds": 60
            })
            yield f"data: {event_data}\n\n"

        return StreamingResponse(
            sse_rate_limit_generator(),
            media_type="text/event-stream",
            status_code=429,
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": "3",
                "X-RateLimit-Window": "60s"
            }
        )

    # Standard JSON 429 Response for REST clients
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": error_msg,
            "detail": str(exc.detail),
            "retry_after_seconds": 60
        },
        headers={
            "Retry-After": "60",
            "X-RateLimit-Limit": "3",
            "X-RateLimit-Window": "60s"
        }
    )
