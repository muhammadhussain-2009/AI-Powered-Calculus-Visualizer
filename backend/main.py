import time
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.schemas import (
    HealthCheckResponse,
    VisualizeRequest,
    VisualizeAPIResponse,
    LLMVisualizationResponse
)
from backend.database import init_db_async, log_visualization_request, get_visualization_logs
from backend.datapipeline import process_and_verify_request
from backend.evaluate import PipelineEvaluator
from security.protection import SecurityHeadersMiddleware
from security.auth import create_anonymous_jwt_token, get_current_session

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for startup & shutdown."""
    await init_db_async()
    yield

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["60 per minute"])

app = FastAPI(
    title="Calculus Visualizer API",
    description="Backend API for generating Desmos visualizations of calculus concepts via LLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,  # Security: Disable redoc admin endpoint
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
@limiter.limit("30/minute")
async def health_check(request: Request):
    """
    Health check endpoint to verify backend status and service dependencies.
    """
    google_key_configured = bool(os.getenv("GOOGLE_API_KEY"))
    desmos_key_configured = bool(os.getenv("DESMOS_API_KEY"))
    
    return HealthCheckResponse(
        status="healthy",
        service="Calculus Visualizer API",
        version="1.0.0",
        timestamp=time.time(),
        dependencies={
            "google_api_key": "configured" if google_key_configured else "missing",
            "desmos_api_key": "configured" if desmos_key_configured else "missing",
            "async_database": "aiosqlite_ready"
        }
    )

@app.get("/api/auth/session", response_model=dict, tags=["Auth"])
@limiter.limit("20/minute")
async def get_anonymous_session(request: Request):
    """
    Issues an anonymous JWT token for session tracking without requiring user registration.
    """
    token = create_anonymous_jwt_token()
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": 24
    }

@app.get("/api/config", response_model=dict, tags=["System"])
@limiter.limit("30/minute")
async def get_frontend_config(request: Request):
    """
    Returns public frontend configuration including the Desmos API Key.
    """
    desmos_key = os.getenv("DESMOS_API_KEY", "").strip() or "dcb31709b452b1cf9dc26972add0fda6"
    return {
        "desmos_api_key": desmos_key
    }

@app.post("/api/visualize", response_model=VisualizeAPIResponse, tags=["Visualization"])
@limiter.limit("15/minute")
async def generate_visualization(
    request: Request,
    payload: VisualizeRequest,
    session: dict = Depends(get_current_session)
):
    """
    Main API endpoint: Validates math relevance, processes prompt via AI pipeline,
    sanitizes LaTeX, evaluates mathematical validity, logs request asynchronously, and returns payload.
    """
    start_time = time.time()
    session_id = session.get("session_id", "anonymous")

    # Step 1: Datapipeline verification and preprocessing
    is_valid_math, sanitized_prompt, metadata = process_and_verify_request(payload.prompt)
    if not is_valid_math:
        error_msg = metadata.get("error", "Invalid or non-mathematical request.")
        await log_visualization_request(
            session_id=session_id,
            prompt=payload.prompt,
            status="REJECTED_NON_MATH",
            error_message=error_msg
        )
        return VisualizeAPIResponse(
            success=False,
            error=error_msg,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    # Step 2: Route prompt to AI Pipeline (ai_pipeline.py)
    try:
        from backend.ai_pipeline import generate_calculus_visualization
        llm_response = await generate_calculus_visualization(sanitized_prompt, metadata)
    except Exception as e:
        error_msg = f"AI Pipeline error: {str(e)}"
        await log_visualization_request(
            session_id=session_id,
            prompt=sanitized_prompt,
            status="AI_PIPELINE_ERROR",
            error_message=error_msg
        )
        return VisualizeAPIResponse(
            success=False,
            error=error_msg,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    # Step 3: Evaluate payload quality
    eval_result = PipelineEvaluator.evaluate_response(llm_response)

    # Step 4: Asynchronously log request to SQLite database
    processing_time_ms = round((time.time() - start_time) * 1000, 2)
    await log_visualization_request(
        session_id=session_id,
        prompt=sanitized_prompt,
        status="SUCCESS" if eval_result["passed"] else "EVAL_FAILED",
        expressions_count=len(llm_response.expressions),
        processing_time_ms=processing_time_ms,
        error_message="; ".join(eval_result["issues"]) if eval_result["issues"] else None
    )

    return VisualizeAPIResponse(
        success=True,
        data=llm_response,
        processing_time_ms=processing_time_ms
    )

@app.get("/api/logs", response_model=dict, tags=["Logs"])
@limiter.limit("10/minute")
async def get_recent_visualization_logs(
    request: Request,
    session: dict = Depends(get_current_session)
):
    """
    Returns recent visualization logs asynchronously from SQLite database.
    """
    logs = await get_visualization_logs(limit=20)
    return {
        "success": True,
        "count": len(logs),
        "logs": logs
    }

# Mount static frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", tags=["System"])
async def root():
    """Root endpoint delivering index.html or API status."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Calculus Visualizer API is active.",
        "health_check": "/health",
        "documentation": "/docs"
    }
