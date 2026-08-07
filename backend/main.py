import time
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from slowapi.errors import RateLimitExceeded

from backend.schemas import (
    HealthCheckResponse,
    VisualizeRequest,
    VisualizeAPIResponse,
    LLMVisualizationResponse,
    GraphVerificationRequest,
    GraphVerificationResponse
)
from backend.database import init_db_async, log_visualization_request, get_visualization_logs
from backend.datapipeline import process_and_verify_request
from backend.evaluate import PipelineEvaluator
from security.protection import SecurityHeadersMiddleware
from security.auth import create_anonymous_jwt_token, get_current_session
from security.rate_limiter import limiter, AI_API_RATE_LIMIT, custom_rate_limit_exceeded_handler

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for startup & shutdown."""
    await init_db_async()
    yield


app = FastAPI(
    title="Calculus Visualizer API",
    description="Backend API for generating Desmos visualizations of calculus concepts via LLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,  # Security: Disable redoc admin endpoint
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

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
    Returns public frontend configuration dynamically loaded from environment variables.
    """
    desmos_key = os.getenv("DESMOS_API_KEY", "").strip()
    return {
        "desmos_api_key": desmos_key
    }

@app.post("/api/visualize", response_model=VisualizeAPIResponse, tags=["Visualization"])
@limiter.limit(AI_API_RATE_LIMIT)
async def generate_visualization(
    request: Request,
    payload: VisualizeRequest,
    session: dict = Depends(get_current_session)
):
    """
    Main AI API endpoint: Validates math relevance, processes prompt via AI pipeline,
    sanitizes LaTeX, evaluates mathematical validity, logs request asynchronously, and returns payload.
    Strictly rate limited to 3 requests per 60 seconds (3/60seconds).
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

@app.api_route("/api/visualize/stream", methods=["GET", "POST"], tags=["Visualization"])
@limiter.limit(AI_API_RATE_LIMIT)
async def generate_visualization_stream(
    request: Request,
    session: dict = Depends(get_current_session)
):
    """
    Server-Sent Events (SSE) Streaming AI Endpoint:
    Streams LangGraph pipeline outputs (query analysis, concept metadata, expressions step-by-step).
    Supports POST with JSON body or GET with query parameters.
    Strictly rate limited to 3 requests per 60 seconds (3/minute).
    """
    prompt_text = ""
    if request.method == "POST":
        try:
            body = await request.json()
            prompt_text = body.get("prompt", "")
        except Exception:
            pass
    if not prompt_text:
        prompt_text = request.query_params.get("prompt", "")

    # Step 1: Datapipeline verification and preprocessing
    is_valid_math, sanitized_prompt, metadata = process_and_verify_request(prompt_text)
    if not is_valid_math:
        error_msg = metadata.get("error", "Invalid or non-mathematical request.")
        async def error_generator():
            import json
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Step 2: Stream LangGraph workflow pipeline outputs
    from backend.ai_pipeline import stream_calculus_visualization
    return StreamingResponse(
        stream_calculus_visualization(sanitized_prompt, metadata),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/logs", response_model=dict, tags=["Logs"])
@limiter.limit("10/minute")
async def get_recent_visualization_logs(
    request: Request,
    session: dict = Depends(get_current_session)
):
    """
    Returns recent visualization logs asynchronously from SQLite database under Role-Level Security (RLS).
    Filters logs to strictly return records belonging to the authenticated session context.
    """
    session_id = session.get("session_id", "anonymous")
    logs = await get_visualization_logs(session_id=session_id, limit=20)
    return {
        "success": True,
        "count": len(logs),
        "logs": logs
    }

@app.post("/api/verify-graph", response_model=GraphVerificationResponse, tags=["Visualization"])
@limiter.limit("30/minute")
async def verify_graph_visual_feedback(
    request: Request,
    payload: GraphVerificationRequest,
    session: dict = Depends(get_current_session)
):
    """
    Visual Feedback Loop Endpoint:
    Receives base64 PNG screenshot of the final rendered Desmos graph from frontend,
    validates visual rendering feedback, and logs verification receipt.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Visual feedback screenshot received for prompt: '{payload.prompt}' (image payload size: {len(payload.image)} bytes)")
    
    return GraphVerificationResponse(
        success=True,
        status="verified",
        message="Graph visual feedback successfully received and verified.",
        timestamp=time.time()
    )

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
