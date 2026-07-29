import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "venv", "Lib", "site-packages"))
sys.path.insert(0, os.getcwd())

import pytest
import asyncio
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import DesmosExpression, LLMVisualizationResponse, VisualizeRequest
from backend.datapipeline import is_math_related, process_and_verify_request
from backend.evaluate import PipelineEvaluator
from backend.ai_pipeline import sanitize_latex, generate_calculus_visualization, stream_calculus_visualization, analyze_query_fallback, generate_desmos_translation_fallback
from backend.database import init_db_async
from security.auth import verify_jwt_token

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Ensure database schema is initialized before running tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_db_async())
    loop.close()

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
    assert is_math_related("Limit: Secant line approaching tangent line") is True
    assert is_math_related("Mean Value Theorem for f(x) = x^3 - 3x") is True
    assert is_math_related("Area between f(x) = x^2 and g(x) = x + 2") is True
    assert is_math_related("Arc length of f(x) = x^(3/2)") is True
    assert is_math_related("Newton's method iteration") is True
    assert is_math_related("Euler's method step-by-step") is True

    assert is_math_related("How to bake a sourdough bread?") is False
    assert is_math_related("Tell me a story about a dragon") is False

    is_valid, sanitized, metadata = process_and_verify_request("Riemann sum of x^2")
    assert is_valid is True

def test_sanitize_latex_function():
    raw_md = "```latex\ny = \\sin(x)\n```"
    assert sanitize_latex(raw_md) == "y = \\sin(x)"

    raw_slash = "y = \\\\frac{x}{2}"
    assert sanitize_latex(raw_slash) == "y = \\frac{x}{2}"

    unclosed = "f(x) = \\frac{1}{x"
    assert sanitize_latex(unclosed) == "f(x) = \\frac{1}{x}"

    leading_eq = "= -2"
    assert sanitize_latex(leading_eq) == "a = -2"

    empty_coord = "(, f())"
    assert sanitize_latex(empty_coord) == "(b, f(b))"

    empty_frac = "\\frac{}{x}"
    assert sanitize_latex(empty_frac) == "\\frac{1}{x}"

def test_langgraph_query_analysis_advanced_concepts():
    mvt = analyze_query_fallback("Mean Value Theorem for f(x) = x^3 - 3x on [-2, 2]")
    assert mvt["concept_type"] == "mvt"
    assert mvt["primary_function"] == "x^3 - 3x"

    limit_sec = analyze_query_fallback("Limit: Secant line approaching tangent line for f(x) = x^2 at x = 1")
    assert limit_sec["concept_type"] == "limit_secant"
    assert limit_sec["x_val"] == "1"

    area_b = analyze_query_fallback("Area between f(x) = x + 2 and g(x) = x^2")
    assert area_b["concept_type"] == "area_between_curves"

    arc_l = analyze_query_fallback("Arc length of f(x) = x^(3/2) from x = 0 to x = 4")
    assert arc_l["concept_type"] == "arc_length"

    newton = analyze_query_fallback("Newton's method iteration for finding root of f(x) = x^3 - 2x - 5")
    assert newton["concept_type"] == "newtons_method"

    euler = analyze_query_fallback("Euler's method step-by-step for dy/dx = x + y with y(0) = 1")
    assert euler["concept_type"] == "eulers_method"

def test_langgraph_pipeline_execution():
    res = asyncio.run(generate_calculus_visualization("Mean Value Theorem for f(x) = x^3 - 3x on [-2, 2]", {}))
    assert res is not None
    assert "Mean Value Theorem" in res.title
    assert len(res.expressions) >= 5

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
    response = client.post("/api/visualize", json={"prompt": "Limit: Secant line approaching tangent line for f(x) = x^2 at x = 1"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"] is not None
    assert len(data["data"]["expressions"]) > 0

def test_visualize_stream_endpoint():
    response = client.post("/api/visualize/stream", json={"prompt": "Arc length of f(x) = x^(3/2) from x = 0 to x = 4"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "data: {" in content
    assert '"type": "analysis"' in content
    assert '"type": "expression"' in content
    assert '"type": "complete"' in content

def test_logs_endpoint():
    response = client.get("/api/logs")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "logs" in data

def test_desmos_translation_pure_math_synthesis():
    res = asyncio.run(generate_calculus_visualization("cubic function", {}))
    assert res is not None
    assert len(res.expressions) > 0
    for exp in res.expressions:
        assert "graph of" not in exp.latex.lower()
        assert "cubic function" not in exp.latex.lower()
    # Check that a concrete mathematical equation is generated
    assert any("x^3" in exp.latex for exp in res.expressions)

def test_desmos_expression_schema_validation():
    # Valid expression
    exp = DesmosExpression(latex="a = -2")
    assert exp.latex == "a = -2"

    # Truncated expression starting with equals sign
    with pytest.raises(ValueError, match="cannot start with an equals sign"):
        DesmosExpression(latex="= -2")

    # Incomplete coordinate pair
    with pytest.raises(ValueError, match="Incomplete coordinate pair"):
        DesmosExpression(latex="(, 3)")

    # Empty fraction parameter
    with pytest.raises(ValueError, match="Missing numerator or denominator"):
        DesmosExpression(latex="\\frac{}{x}")

    # English prose filler inside LaTeX string
    with pytest.raises(ValueError, match="Output PURE mathematical LaTeX only"):
        DesmosExpression(latex="f(x) = the graph of a cubic function")


def test_verify_graph_visual_feedback_endpoint():
    response = client.get("/api/auth/session")
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "prompt": "Riemann sum of x^2",
        "expressions_count": 5
    }

    resp = client.post("/api/verify-graph", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["status"] == "verified"
    assert "successfully received" in data["message"]


def test_langgraph_rag_template_fitting_and_no_over_generalization():
    # 1. Custom Riemann Sum prompt
    prompt_riemann = "Riemann sum of x^3 + 2x from x = 1 to x = 4 with 10 rectangles"
    analysis_r = analyze_query_fallback(prompt_riemann)
    assert analysis_r["concept_type"] == "riemann_sum"
    assert "x^3" in analysis_r["primary_function"]
    assert analysis_r["lower_limit"] == "1"
    assert analysis_r["upper_limit"] == "4"
    assert analysis_r["n_val"] == "10"
    assert len(analysis_r["matched_templates"]) > 0

    res_r = asyncio.run(generate_calculus_visualization(prompt_riemann, {}))
    assert res_r is not None
    assert any("x^3" in exp.latex for exp in res_r.expressions)
    assert any("a = 1" in exp.latex for exp in res_r.expressions)
    assert any("b = 4" in exp.latex for exp in res_r.expressions)
    assert any("n = 10" in exp.latex for exp in res_r.expressions)

    # 2. Custom Taylor Series prompt
    prompt_taylor = "Taylor series of cos(x) around x = 0"
    analysis_t = analyze_query_fallback(prompt_taylor)
    assert analysis_t["concept_type"] == "taylor_series"
    assert "cos" in analysis_t["primary_function"].lower()

    res_t = asyncio.run(generate_calculus_visualization(prompt_taylor, {}))
    assert res_t is not None
    assert any("cos" in exp.latex.lower() for exp in res_t.expressions)
    assert not any("sin" in exp.latex.lower() for exp in res_t.expressions)

    # 3. Custom Newton's Method prompt
    prompt_newton = "Newton's method for f(x) = x^2 - 5 at x_0 = 2"
    analysis_n = analyze_query_fallback(prompt_newton)
    assert analysis_n["concept_type"] == "newtons_method"
    assert "x^2 - 5" in analysis_n["primary_function"]
    assert analysis_n["x_val"] == "2"

    res_n = asyncio.run(generate_calculus_visualization(prompt_newton, {}))
    assert res_n is not None
    assert any("x^2 - 5" in exp.latex for exp in res_n.expressions)
    assert any("x_0 = 2" in exp.latex for exp in res_n.expressions)

    # 4. Custom Implicit Differentiation prompt
    prompt_implicit = "Implicit differentiation for x^2 + 4y^2 = 16"
    analysis_i = analyze_query_fallback(prompt_implicit)
    assert analysis_i["concept_type"] == "implicit_differentiation"
    assert "4y^2" in analysis_i["primary_function"]

    res_i = asyncio.run(generate_calculus_visualization(prompt_implicit, {}))
    assert res_i is not None
    assert any("4y^2" in exp.latex for exp in res_i.expressions)


def test_call_llm_with_timeout_and_retry_success():
    from backend.ai_pipeline import call_llm_with_timeout_and_retry

    def dummy_fast_func():
        return "success"

    result = asyncio.run(call_llm_with_timeout_and_retry(dummy_fast_func, max_retries=2, timeout_seconds=1.0, retry_delay=0.1))
    assert result == "success"


def test_call_llm_with_timeout_and_retry_timeout_failover():
    from backend.ai_pipeline import call_llm_with_timeout_and_retry

    attempts = 0

    def dummy_slow_func():
        nonlocal attempts
        attempts += 1
        time.sleep(0.5)
        return "slow"

    # Expect timeout with 0.1s timeout threshold
    with pytest.raises(Exception):
        asyncio.run(call_llm_with_timeout_and_retry(dummy_slow_func, max_retries=2, timeout_seconds=0.1, retry_delay=0.05))

    # Initial attempt + 2 retries = 3 attempts total
    assert attempts == 3


def test_ai_endpoint_rate_limiting_enforcement():
    """
    Verifies that sending more than 3 requests in 60 seconds to AI API endpoints
    blocks/throttles further requests and returns an HTTP 429 status code with a clear rate-limit message.
    """
    from security.rate_limiter import limiter
    if hasattr(limiter, "reset"):
        try:
            limiter.reset()
        except Exception:
            pass

    # Issue a session token for session-based rate keying
    session_resp = client.get("/api/auth/session")
    token = session_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"prompt": "Derivative of x^2"}

    # Execute requests up to the 3 request limit threshold
    responses = []
    for i in range(3):
        res = client.post("/api/visualize", json=payload, headers=headers)
        responses.append(res.status_code)

    # All first 3 requests should be accepted (HTTP 200)
    assert all(code == 200 for code in responses)

    # 4th request in the 60 second window MUST be blocked/throttled (HTTP 429)
    blocked_res = client.post("/api/visualize", json=payload, headers=headers)
    assert blocked_res.status_code == 429
    data = blocked_res.json()
    assert data["success"] is False
    assert "Rate limit exceeded" in data["error"]
    assert "3 requests in 60 seconds" in data["error"]


def test_query_analysis_target_function_isolation_and_no_few_shot_overfitting():
    """
    Verifies that the Query Analysis node isolates the user's exact mathematical function
    into target_function and does NOT overfit or override it with few-shot defaults like x^3 - 2x.
    """
    prompt = "Newton's method for f(x) = x^4 - 7x + 2 at x_0 = 3"
    analysis = analyze_query_fallback(prompt)

    assert analysis["concept_type"] == "newtons_method"
    assert analysis["target_function"] == "x^4 - 7x + 2"
    assert analysis["primary_function"] == "x^4 - 7x + 2"
    assert analysis["target_function"] != "x^3 - 2x"

    res = generate_desmos_translation_fallback(analysis)
    assert res is not None
    assert any("x^4 - 7x + 2" in exp.latex for exp in res.expressions)
    assert not any("x^3 - 2x" in exp.latex for exp in res.expressions)


def test_langgraph_self_correction_loop_routing():
    """
    Verifies that the LangGraph validation node detects invalid expressions / dropped target functions
    and triggers cyclic retry routing to desmos_translation node before final completion.
    """
    from backend.ai_pipeline import should_retry, GraphState
    from langgraph.graph import END

    # State with validation_error and retry_count < 2 should trigger retry back to desmos_translation
    state_retry: GraphState = {
        "prompt": "Test",
        "metadata": {},
        "analysis": {"target_function": "x^2"},
        "llm_response": None,
        "validated_response": None,
        "validation_issues": ["English prose detected"],
        "error": None,
        "retry_count": 1,
        "validation_error": "Validation Error: English prose detected"
    }
    assert should_retry(state_retry) == "desmos_translation"

    # State with max retries reached (retry_count >= 2) should proceed to END
    state_end: GraphState = {
        "prompt": "Test",
        "metadata": {},
        "analysis": {"target_function": "x^2"},
        "llm_response": None,
        "validated_response": None,
        "validation_issues": [],
        "error": None,
        "retry_count": 2,
        "validation_error": "Validation Error: Max retries"
    }
    assert should_retry(state_end) == END


