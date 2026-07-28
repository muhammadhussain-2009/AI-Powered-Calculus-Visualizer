import pytest
import asyncio
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import DesmosExpression, LLMVisualizationResponse, VisualizeRequest
from backend.datapipeline import is_math_related, process_and_verify_request
from backend.evaluate import PipelineEvaluator
from backend.ai_pipeline import sanitize_latex, generate_fallback_visualization
from backend.database import init_db_async
from security.auth import verify_jwt_token

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    """Ensure database schema is initialized before running tests."""
    asyncio.run(init_db_async())

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Calculus Visualizer API"
    assert "async_database" in data["dependencies"]

def test_security_headers_present():
    response = client.get("/health")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert "Content-Security-Policy" in response.headers

def test_anonymous_jwt_session_issuance():
    response = client.get("/api/auth/session")
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    token = data["access_token"]
    payload = verify_jwt_token(token)
    assert payload["type"] == "anonymous_session"
    assert "session_id" in payload

def test_config_endpoint():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "desmos_api_key" in data

def test_datapipeline_math_verification():
    assert is_math_related("Riemann sum of x^2 from 0 to 5") is True
    assert is_math_related("Tangent line to sin(x) at x=pi") is True

    assert is_math_related("How to bake a sourdough bread?") is False
    assert is_math_related("Tell me a story about a dragon") is False

    is_valid, sanitized, metadata = process_and_verify_request("Riemann sum of x^2")
    assert is_valid is True
    assert "riemann_sum" in metadata["detected_topics"]

def test_sanitize_latex_function():
    raw_md = "```latex\ny = \\sin(x)\n```"
    assert sanitize_latex(raw_md) == "y = \\sin(x)"

    raw_slash = "y = \\\\frac{x}{2}"
    assert sanitize_latex(raw_slash) == "y = \\frac{x}{2}"

    unclosed = "f(x) = \\frac{1}{x"
    assert sanitize_latex(unclosed) == "f(x) = \\frac{1}{x}"

def test_fallback_visualization_generator():
    riemann = generate_fallback_visualization("Riemann sum of x^2", {"detected_topics": ["riemann_sum"]})
    assert "Riemann" in riemann.title
    assert len(riemann.expressions) >= 5

    tangent = generate_fallback_visualization("Tangent line to sin(x)", {"detected_topics": ["differentiation"]})
    assert "Tangent" in tangent.title
    assert len(tangent.expressions) >= 4

def test_pipeline_evaluator():
    valid_resp = LLMVisualizationResponse(
        title="Parabola Graph",
        concept_explanation="Interactive graph of y=x^2",
        expressions=[DesmosExpression(latex="y=x^2", color="#2d70b3")]
    )
    eval_result = PipelineEvaluator.evaluate_response(valid_resp)
    assert eval_result["passed"] is True
    assert eval_result["score"] >= 0.8

def test_visualize_endpoint_non_math_rejection():
    response = client.post("/api/visualize", json={"prompt": "Write a recipe for cookies"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    assert "calculus or mathematics" in data["error"]

def test_visualize_endpoint_valid_math():
    response = client.post("/api/visualize", json={"prompt": "Riemann sum of x^2 from x=0 to x=3"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"] is not None
    assert len(data["data"]["expressions"]) > 0
    for exp in data["data"]["expressions"]:
        assert "```" not in exp["latex"]

def test_logs_endpoint():
    response = client.get("/api/logs")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "logs" in data
