import re
import os
import logging
from typing import Dict, Any, List, Optional

from backend.schemas import LLMVisualizationResponse, DesmosExpression, SliderBounds

logger = logging.getLogger(__name__)

def sanitize_latex(latex: str) -> str:
    """
    Sanitizes LaTeX string for Desmos calculator compatibility.
    Strips raw markdown formatting, normalizes backslashes, removes unsupported LaTeX macros.
    """
    if not latex:
        return ""
    
    cleaned = re.sub(r'```(?:latex|json)?', '', latex, flags=re.IGNORECASE)
    cleaned = cleaned.replace('```', '').strip()
    
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    cleaned = re.sub(r'\\operatorname\{([^}]+)\}', r'\\\1', cleaned)
    cleaned = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\text\{([^}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\displaystyle\b', '', cleaned)
    cleaned = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', cleaned)
    cleaned = re.sub(r'\\le\b', r'\\le ', cleaned)
    cleaned = re.sub(r'\\ge\b', r'\\ge ', cleaned)

    open_braces = cleaned.count('{')
    close_braces = cleaned.count('}')
    if open_braces > close_braces:
        cleaned += '}' * (open_braces - close_braces)

    return cleaned

def parse_user_prompt(prompt: str) -> Dict[str, Any]:
    """
    General Math & Calculus Parser: Analyzes user prompt to determine mathematical intent,
    equations, functions, points, or specialized calculus topics.
    """
    cleaned = prompt.strip()
    lowered = cleaned.lower()

    # Remove action verbs at start e.g. "plot", "graph", "draw", "show", "visualize"
    action_stripped = re.sub(r'^(?:plot|graph|draw|show|visualize|display)\s+', '', cleaned, flags=re.IGNORECASE).strip()

    # 1. Riemann Sum Concept
    if 'riemann' in lowered or 'rectangle' in lowered:
        m = re.search(r'(?:riemann\s+sum\s+of|of)\s+([^\,;\.\n]+?)(?=\s+(?:from|at|with|in|where|between|$))', action_stripped, re.IGNORECASE)
        func = m.group(1).strip() if m else 'x^2'
        func = re.sub(r'^(?:f\(x\)|y)\s*=\s*', '', func, flags=re.IGNORECASE)
        return {'type': 'riemann', 'func': func, 'raw': cleaned}

    # 2. Tangent Line / Derivative Concept
    if 'tangent' in lowered or 'derivative' in lowered or 'slope' in lowered:
        m = re.search(r'(?:tangent\s+line\s+to|derivative\s+of|to|of)\s+([^\,;\.\n]+?)(?=\s+(?:at|from|with|in|where|$))', action_stripped, re.IGNORECASE)
        func = m.group(1).strip() if m else action_stripped
        func = re.sub(r'^(?:f\(x\)|y)\s*=\s*', '', func, flags=re.IGNORECASE)
        m_x = re.search(r'at\s+x\s*=\s*([^\,\s]+)', action_stripped, re.IGNORECASE)
        x_val = m_x.group(1).strip() if m_x else '1'
        return {'type': 'tangent', 'func': func, 'x_val': x_val, 'raw': cleaned}

    # 3. Area Under Curve / Integral Concept
    if 'integral' in lowered or 'area' in lowered:
        m = re.search(r'(?:area\s+under|integral\s+of|under|of)\s+([^\,;\.\n]+?)(?=\s+(?:from|at|with|in|where|$))', action_stripped, re.IGNORECASE)
        func = m.group(1).strip() if m else action_stripped
        func = re.sub(r'^(?:f\(x\)|y)\s*=\s*', '', func, flags=re.IGNORECASE)
        return {'type': 'integral', 'func': func, 'raw': cleaned}

    # 4. Explicit Equation (e.g. y = 3x + 34, x^2 + y^2 = 25, 2x + 3y = 12)
    m_eq = re.search(r'([a-zA-Z0-9\+\-\*\/\^\(\)\.\s]+\s*=\s*[a-zA-Z0-9\+\-\*\/\^\(\)\.\s]+)', action_stripped)
    if m_eq:
        eq_str = m_eq.group(1).strip()
        return {'type': 'equation', 'latex': eq_str, 'raw': cleaned}

    # 5. General Function / Expression (e.g. 3x + 34, sin(x) + cos(x), x^3 - 3x)
    func = action_stripped
    latex_expr = func if (func.startswith('y=') or func.startswith('f(x)=')) else f'y = {func}'
    return {'type': 'general', 'latex': latex_expr, 'raw': cleaned}

def generate_fallback_visualization(prompt: str, metadata: Dict[str, Any]) -> LLMVisualizationResponse:
    """
    Generates high-quality calculus and mathematical visualizations dynamically for ANY user prompt & equation.
    """
    parsed = parse_user_prompt(prompt)
    p_type = parsed['type']

    # 1. Riemann Sum
    if p_type == 'riemann':
        func = sanitize_latex(parsed['func'])
        return LLMVisualizationResponse(
            title=f"Riemann Sum of f(x) = {func}",
            concept_explanation=f"Approximating the area under f(x) = {func} using n rectangular subintervals.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {func}", showLabel=True),
                DesmosExpression(id="exp_a", latex="a = 0", color="#000000", hidden=True),
                DesmosExpression(id="exp_b", latex="b = 3", color="#000000", hidden=True),
                DesmosExpression(id="exp_n", latex="n = 6", color="#6042a6", sliderBounds=SliderBounds(min="1", max="50", step="1")),
                DesmosExpression(id="exp_dx", latex="w = \\frac{b-a}{n}", hidden=True),
                DesmosExpression(id="exp_sum", latex="S = \\sum_{k=1}^{n} f(a + k \\cdot w) \\cdot w", color="#388c46", label="Right Riemann Sum S", showLabel=True),
                DesmosExpression(id="exp_rect", latex="0 \\le y \\le f(a + \\floor(\\frac{x-a}{w}) \\cdot w + w) \\{a \\le x \\le b\\}", color="#c74440")
            ]
        )

    # 2. Tangent Line / Derivative
    if p_type == 'tangent':
        func = sanitize_latex(parsed['func'])
        x_val = parsed.get('x_val', '1')
        return LLMVisualizationResponse(
            title=f"Tangent Line & Derivative of f(x) = {func}",
            concept_explanation=f"Visualizing tangent line and rate of change f'(x) for f(x) = {func} at x = {x_val}.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {func}", showLabel=True),
                DesmosExpression(id="exp_x1", latex=f"x_1 = {x_val}", color="#6042a6", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_pt", latex="(x_1, f(x_1))", color="#c74440", showLabel=True, label=f"P({x_val}, f({x_val}))"),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", color="#388c46", lineStyle="DASHED", label="f'(x)", showLabel=True),
                DesmosExpression(id="exp_tan", latex="y - f(x_1) = g(x_1) \\cdot (x - x_1)", color="#c74440", lineWidth=2.5, label="Tangent Line", showLabel=True)
            ]
        )

    # 3. Integral / Area Under Curve
    if p_type == 'integral':
        func = sanitize_latex(parsed['func'])
        return LLMVisualizationResponse(
            title=f"Definite Integral of f(x) = {func}",
            concept_explanation=f"Shaded accumulation region representing \\int_{{a}}^{{b}} f(x) dx for f(x) = {func}.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {func}", color="#2d70b3", lineWidth=3.0, label=f"f(x)", showLabel=True),
                DesmosExpression(id="exp_a", latex="a = 0", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_b", latex="b = 3", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_area", latex="0 \\le y \\le f(x) \\{a \\le x \\le b\\}", color="#388c46"),
                DesmosExpression(id="exp_val", latex="I = \\int_{a}^{b} f(x) dx", color="#c74440", label="Area I", showLabel=True)
            ]
        )

    # 4. Direct Equation Plotting (e.g. y = 3x + 34, x^2 + y^2 = 25)
    latex_str = sanitize_latex(parsed['latex'])
    return LLMVisualizationResponse(
        title=f"Graph of {latex_str}",
        concept_explanation=f"Interactive Desmos 2D graph rendering mathematical curve for: {latex_str}.",
        expressions=[
            DesmosExpression(id="exp_main", latex=latex_str, color="#2d70b3", lineWidth=3.0, label=latex_str, showLabel=True)
        ]
    )

async def generate_calculus_visualization(prompt: str, metadata: Dict[str, Any]) -> LLMVisualizationResponse:
    """
    Main AI Pipeline entry point. Uses Instructor / Google GenAI client to produce
    pydantic-validated Desmos commands for ANY user prompt, with generalized fallback handling.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not google_api_key:
        logger.info("Using generalized math visualizer engine (No API Key).")
        return generate_fallback_visualization(prompt, metadata)

    try:
        import instructor
        from google import genai

        client = instructor.from_genai(
            client=genai.Client(api_key=google_api_key),
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
        )

        system_prompt = (
            "You are an expert calculus teacher and Desmos API specialist.\n"
            "Your task is to take a natural language calculus or mathematical concept prompt and output a structured JSON "
            "conforming strictly to the LLMVisualizationResponse schema.\n\n"
            "GUIDELINES FOR DESMOS EXPRESSIONS:\n"
            "1. Tailor the expressions strictly to the user's requested function, equation, or concept.\n"
            "2. If the user asks to plot a simple equation like 'plot y=3x+34', return ONLY that equation in expressions.\n"
            "3. Each expression MUST have valid Desmos LaTeX syntax.\n"
            "4. For derivatives, use g(x) = \\frac{d}{dx}f(x) or f'(x) = ...\n"
            "5. For area under curve / inequalities, use bounds like 0 \\le y \\le f(x) \\{a \\le x \\le b\\}.\n"
            "6. Use distinct hex colors: #2d70b3 (blue), #c74440 (red), #388c46 (green), #6042a6 (purple), #000000 (black).\n"
            "7. Keep expressions clear, renderable, dynamic, and educational."
        )

        candidate_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-lite"]
        last_error = None

        for model in candidate_models:
            try:
                response: LLMVisualizationResponse = client.messages.create(
                    model=model,
                    response_model=LLMVisualizationResponse,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Create a dynamic visualization for: '{prompt}'"}
                    ]
                )

                for exp in response.expressions:
                    exp.latex = sanitize_latex(exp.latex)

                return response
            except Exception as model_err:
                last_error = model_err
                logger.warning(f"Model {model} failed: {model_err}")

        raise last_error if last_error else RuntimeError("All Gemini models failed.")

    except Exception as e:
        logger.warning(f"Google GenAI LLM call failed ({e}). Falling back to generalized math parser engine.")
        response = generate_fallback_visualization(prompt, metadata)
        for exp in response.expressions:
            exp.latex = sanitize_latex(exp.latex)
        return response
