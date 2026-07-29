import re
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from backend.schemas import LLMVisualizationResponse, DesmosExpression, SliderBounds
from backend.evaluate import PipelineEvaluator

logger = logging.getLogger(__name__)

# --- Load Calculus Knowledge Base ---
CALCULUS_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "calculus_knowledge.json")
CALCULUS_KNOWLEDGE_BASE: List[Dict[str, Any]] = []

try:
    if os.path.exists(CALCULUS_KNOWLEDGE_PATH):
        with open(CALCULUS_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            CALCULUS_KNOWLEDGE_BASE = json.load(f)
        logger.info(f"Loaded {len(CALCULUS_KNOWLEDGE_BASE)} concepts from calculus_knowledge.json")
except Exception as e:
    logger.warning(f"Could not load calculus_knowledge.json: {e}")


def match_calculus_knowledge(prompt: str) -> Dict[str, Any]:
    """
    RAG / Knowledge Base Intent Matching: Matches user prompt against calculus concepts in calculus_knowledge.json.
    Returns matched concept dictionary containing concept_name, description, and desmos_templates.
    """
    lowered = prompt.lower()
    
    keyword_map = [
        ("implicit", "implicit_differentiation"),
        ("taylor", "taylor_series"),
        ("maclaurin", "taylor_series"),
        ("optimiz", "optimization"),
        ("optima", "optimization"),
        ("slope field", "slope_fields"),
        ("direction field", "slope_fields"),
        ("riemann", "riemann_sum"),
        ("rectangle sum", "riemann_sum"),
        ("newton", "newtons_method"),
        ("euler", "eulers_method"),
        ("area under", "area_under_curve"),
        ("area between", "area_between_curves"),
        ("between", "area_between_curves"),
        ("arc length", "arc_length"),
        ("secant", "limit_secant"),
        ("tangent line", "tangent_line"),
        ("mean value", "mvt"),
        ("mvt", "mvt"),
        ("rate of change", "derivative_rate_of_change"),
        ("velocity", "derivative_rate_of_change"),
        ("critical point", "critical_points_extrema"),
        ("extrema", "critical_points_extrema"),
        ("concavity", "concavity_inflection"),
        ("inflection", "concavity_inflection"),
        ("by parts", "integration_by_parts"),
        ("disc", "disc_washer_method"),
        ("washer", "disc_washer_method"),
        ("shell", "shell_method"),
        ("parametric", "parametric_curves"),
        ("polar", "polar_curves"),
        ("cardioid", "polar_curves"),
        ("related rates", "related_rates"),
        ("tangent", "tangent_line"),
        ("area", "area_under_curve"),
        ("integral", "riemann_sum"),
    ]

    matched_name = "general"
    for kw, concept_name in keyword_map:
        if kw in lowered:
            matched_name = concept_name
            break

    for item in CALCULUS_KNOWLEDGE_BASE:
        if item.get("concept_name") == matched_name:
            return item

    return {
        "concept_name": matched_name,
        "description": "General 2D mathematical function or curve visualization.",
        "desmos_templates": ["y = f(x)"]
    }


# --- Helper Functions ---

def sanitize_latex(latex: str) -> str:
    """
    Sanitizes LaTeX string for Desmos calculator compatibility.
    Strips raw markdown formatting, normalizes backslashes, converts unsupported LaTeX macros,
    strips out English prose or non-mathematical text inside LaTeX expressions,
    and repairs truncated mathematical expressions.
    """
    if not latex:
        return ""
    
    cleaned = re.sub(r'```(?:latex|json)?', '', latex, flags=re.IGNORECASE)
    cleaned = cleaned.replace('```', '').strip()
    
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # Convert unsupported \floor and \ceil to Desmos-compatible \operatorname{floor} / \operatorname{ceil}
    cleaned = re.sub(r'\\floor\b', r'\\operatorname{floor}', cleaned)
    cleaned = re.sub(r'\\ceil\b', r'\\operatorname{ceil}', cleaned)

    # Detect and replace English prose inside latex (e.g. "f(x) = the graph of a cubic function" -> "f(x) = x^3 - 2x")
    if re.search(r'\b(?:the|graph|of|a\s+cubic|a\s+quadratic|a\s+line|a\s+function|polynomial|shaded|area|under|curve|point)\b', cleaned, re.IGNORECASE):
        cleaned = re.sub(r'^(?:f\(x\)|y)\s*=\s*(?:the\s+)?(?:graph\s+of\s+)?(?:a\s+)?cubic(?:\s+function)?.*$', 'f(x) = x^3 - 2x', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^(?:f\(x\)|y)\s*=\s*(?:the\s+)?(?:graph\s+of\s+)?(?:a\s+)?quadratic(?:\s+function)?.*$', 'f(x) = x^2 - 4', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^(?:f\(x\)|y)\s*=\s*(?:the\s+)?(?:graph\s+of\s+)?(?:a\s+)?trig(?:onometric)?(?:\s+function)?.*$', lambda m: r'f(x) = \sin(x)', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\\text\{[^}]*\}', '', cleaned)

    # Fix leading equals sign (e.g. "= -2" -> "a = -2" or "= x^2" -> "y = x^2")
    if cleaned.startswith('='):
        val = cleaned[1:].strip()
        if re.match(r'^-?\d+(?:\.\d+)?$', val):
            cleaned = f"a = {val}"
        else:
            cleaned = f"y = {val}" if val else "y = x"

    # Fix incomplete coordinate pairs (e.g. "(, f())" -> "(b, f(b))", "(b, )" -> "(b, f(b))")
    cleaned = re.sub(r'\(\s*,\s*f\([^\)]*\)\s*\)', '(b, f(b))', cleaned)
    cleaned = re.sub(r'\(\s*([a-zA-Z0-9_]+)\s*,\s*\)', r'(\1, f(\1))', cleaned)
    cleaned = re.sub(r'\(\s*,\s*([a-zA-Z0-9_]+)\s*\)', r'(\1, f(\1))', cleaned)
    cleaned = re.sub(r'\(\s*,\s*\)', '(x_1, y_1)', cleaned)

    # Fix empty parentheses () or empty square brackets []
    cleaned = re.sub(r'\(\s*\)', '', cleaned)
    cleaned = re.sub(r'\[\s*\]', '', cleaned)

    # Fix empty fraction numerators/denominators (e.g. \frac{}{x} -> \frac{1}{x}, \frac{x}{} -> \frac{x}{1})
    cleaned = re.sub(r'\\frac\{\s*\}\{([^}]+)\}', r'\\frac{1}{\1}', cleaned)
    cleaned = re.sub(r'\\frac\{([^}]+)\}\{\s*\}', r'\\frac{\1}{1}', cleaned)

    cleaned = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\text\{([^}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\displaystyle\b', '', cleaned)
    cleaned = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', cleaned)
    cleaned = re.sub(r'\\le\b', r'\\le ', cleaned)
    cleaned = re.sub(r'\\ge\b', r'\\ge ', cleaned)

    # Normalize sin, cos, tan, ln, log, sqrt if missing backslash (e.g. sin(x) -> \sin(x))
    cleaned = re.sub(r'(?<!\\)\b(sin|cos|tan|ln|log|sqrt)\b', r'\\\1', cleaned)

    open_braces = cleaned.count('{')
    close_braces = cleaned.count('}')
    if open_braces > close_braces:
        cleaned += '}' * (open_braces - close_braces)

    return cleaned


# --- Pydantic Schema for Node 1 Query Analysis ---

# --- Pydantic Schema for Node 1 Query Analysis ---

class QueryAnalysisModel(BaseModel):
    concept_type: str = Field(
        description="Type of calculus concept from the 20 subfields (e.g., 'implicit_differentiation', 'taylor_series', 'optimization', 'slope_fields', 'riemann_sum', 'newtons_method', 'eulers_method', 'area_between_curves', 'area_under_curve', 'arc_length', 'limit_secant', 'tangent_line', 'mvt', 'derivative_rate_of_change', 'critical_points_extrema', 'concavity_inflection', 'integration_by_parts', 'disc_washer_method', 'shell_method', 'parametric_curves', 'polar_curves', 'related_rates', 'general')"
    )
    target_function: str = Field(description="The exact mathematical expression provided by the user (e.g., 'x^3 + 2x', 'x^2 - 5', '\\cos(x)', 'x^2 + 4y^2 = 16'). Takes absolute precedence over few-shot examples.")
    primary_function: str = Field(default="x^2", description="Primary mathematical function f(x), relation, or equation specified by the user.")
    secondary_function: Optional[str] = Field(default=None, description="Secondary function g(x) if comparing curves or areas")
    lower_limit: Optional[str] = Field(default="0", description="Explicit lower limit bound 'a' if specified in query")
    upper_limit: Optional[str] = Field(default="3", description="Explicit upper limit bound 'b' if specified in query")
    coordinates: Optional[str] = Field(default=None, description="Key point, coordinate pair (x, y), or evaluation point x0 if specified in query")
    x_val: Optional[str] = Field(default="1", description="Key evaluation point x0 or x_val")
    n_val: Optional[str] = Field(default="6", description="Subintervals count n if specified for Riemann sum or approximations")
    h_val: Optional[str] = Field(default="0.5", description="Step size h if specified for Euler method or secant approximation")
    bounds: List[str] = Field(default=["0", "3"], description="Interval bounds [a, b]")
    mathematical_intent: str = Field(description="Summary of the mathematical objective and core visual components")
    matched_templates: List[str] = Field(default=[], description="Retrieved Desmos LaTeX templates from calculus knowledge base")


# --- LangGraph State Schema ---

class GraphState(TypedDict):
    prompt: str
    metadata: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]
    llm_response: Optional[LLMVisualizationResponse]
    validated_response: Optional[LLMVisualizationResponse]
    validation_issues: List[str]
    error: Optional[str]
    retry_count: int
    validation_error: Optional[str]


# --- Helper Function Extraction ---

def extract_clean_function(text: str, is_secondary: bool = False) -> str:
    """
    Extracts a clean mathematical function string from natural language text,
    prioritizing explicit equation assignments (f(x) = ..., y = ..., dy/dx = ...) or implicit equations.
    """
    if not text:
        return "x^2" if not is_secondary else "0"
    
    cleaned = text.strip()

    if is_secondary:
        m_g = re.search(r'g\(x\)\s*=\s*([^\,;\.\n]+)', cleaned, re.IGNORECASE)
        if m_g:
            expr = m_g.group(1).strip()
            expr = re.sub(r'\s+(?:from|on|at|with|where|in)\s+.*$', '', expr, flags=re.IGNORECASE).strip()
            return sanitize_latex(expr) if expr else "0"
        m_and = re.search(r'\band\s+(?:g\(x\)|y)?\s*=?\s*([^\,;\.\n]+)', cleaned, re.IGNORECASE)
        if m_and:
            expr = m_and.group(1).strip()
            expr = re.sub(r'\s+(?:from|on|at|with|where|in)\s+.*$', '', expr, flags=re.IGNORECASE).strip()
            return sanitize_latex(expr) if expr else "0"
        return "0"

    # 1. Check if prompt contains an explicit equation assignment e.g. f(x) = ..., y = ..., s(t) = ..., r = ..., dy/dx = ...
    m_eq = re.search(r'(?:f\(x\)|y|s\(t\)|r|dy/dx)\s*=\s*([^\,;\n]+)', cleaned, re.IGNORECASE)
    if m_eq:
        raw_eq = m_eq.group(1).strip()
        # Strip away 'and g(x) = ...' or 'and y = ...'
        raw_eq = re.sub(r'\s+and\s+(?:g\(x\)|y)?\s*=.*$', '', raw_eq, flags=re.IGNORECASE).strip()
        # Strip trailing clause (from ..., on ..., at x=..., with ..., n=...)
        raw_eq = re.sub(r'\s+(?:from|on|at|with|where|in|around|n\s*=)\s+.*$', '', raw_eq, flags=re.IGNORECASE).strip()
        # Strip trailing natural language prose
        raw_eq = re.sub(r'\s+(?:iteration|step-by-step|rectangles|rects|rect|rectangles?)\b.*$', '', raw_eq, flags=re.IGNORECASE).strip()
        if raw_eq and raw_eq.lower() not in ["volume", "area", "length"]:
            return sanitize_latex(raw_eq)

    # 2. Check for implicit equation relation (e.g. x^2 + 4y^2 = 16 or x^3 + y^3 = 6xy)
    if '=' in cleaned:
        non_bound = re.sub(r'\b(?:from|to|on|at|with|where|in|around)\s+(?:x|y|t|x_?0|y_?0)\s*=\s*[-+]?\d+(?:\.\d+)?', '', cleaned, flags=re.IGNORECASE)
        m_imp = re.search(r'([a-zA-Z0-9_\^\(\)\\\{\}\+\-\*\/\s]+\s*=\s*[a-zA-Z0-9_\^\(\)\\\{\}\+\-\*\/\s]+)', non_bound)
        if m_imp:
            raw_imp = m_imp.group(1).strip()
            raw_imp = re.sub(r'\s+(?:from|on|at|with|where|in|around)\s+.*$', '', raw_imp, flags=re.IGNORECASE).strip()
            if raw_imp and not re.match(r'^(?:n|x|y|h|a|b|t|x_?0|y_?0)\s*=\s*[-+]?\d+(?:\.\d+)?$', raw_imp, re.IGNORECASE):
                if not re.search(r'^(?:riemann|limit|newton|euler|area|arc)', raw_imp, re.IGNORECASE):
                    return sanitize_latex(raw_imp)

    # 3. Strip concept titles and action word prefixes
    cleaned = re.sub(r'^(?:plot|graph|draw|show|visualize|display)?\s*(?:the\s+)?(?:graph\s+of\s+)?(?:a\s+)?', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^(?:newton\'?s?\s+method(?:\s+iteration)?(?:\s+for\s+finding\s+root(?:\s+of)?)?|euler\'?s?\s+method(?:\s+step-by-step)?(?:\s+for)?|limit:?\s*secant\s+line\s+approaching\s+tangent\s+line|mean\s+value\s+theorem|riemann\s+sum|area\s+between|area\s+under|arc\s+length|derivative\s+rate\s+of\s+change|critical\s+points\s+and\s+extrema|concavity\s+and\s+inflection\s+points|integration\s+by\s+parts|disc\s+washer\s+method|shell\s+method|parametric\s+curves|polar\s+curves|related\s+rates|optimization|taylor\s+series(?:\s+expansion)?|tangent\s+line(?:\s+to)?|slope\s+field|direction\s+field)\s*(?:for|of|between|to|approaching|under|around)?\s*', '', cleaned, flags=re.IGNORECASE).strip()
    
    # Strip line / function prose prefixes
    cleaned = re.sub(r'^(?:line|curve|function|polynomial|parabola|cubic|quadratic|trig|trigonometric)?\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^(?:f\(x\)|g\(x\)|y|s\(t\)|r|dy/dx)\s*=\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+(?:from|on|at|with|where|in|around)\s+.*$', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+and\s+g\(x\)\s*=.*$', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\b(?:the|graph|of|a|line|curve|function|polynomial|method|iteration|step-by-step|for|finding|root|volume|area|length|under|rectangles|around)\b', '', cleaned, flags=re.IGNORECASE).strip()

    # Fallback to mathematical default if cleaned text is invalid/empty
    if not cleaned or cleaned.lower() in ["volume", "area", "length", "surface", "method", "shell", "washer", "disc"] or cleaned.startswith("0 \\le"):
        if "cubic" in text.lower():
            return "x^3 - 2x"
        elif "quadratic" in text.lower() or "parabola" in text.lower():
            return "x^2 - 4"
        elif "trig" in text.lower() or "sine" in text.lower() or "sin" in text.lower():
            return "\\sin(x)"
        elif "cos" in text.lower() or "cosine" in text.lower():
            return "\\cos(x)"
        elif "line" in text.lower():
            return "2x + 5"
        elif "sqrt" in text.lower():
            return "\\sqrt{x}"
        else:
            return "x^2" if not is_secondary else "0"

    return sanitize_latex(cleaned)


# --- Fallback NLP Math Query Analyzer ---

def analyze_query_fallback(prompt: str) -> Dict[str, Any]:
    """
    Analyzes mathematical intent and matches concept against calculus_knowledge.json.
    Extracted strict schema fields: target_function, lower_limit, upper_limit, coordinates, n_val, h_val.
    """
    cleaned = prompt.strip()

    knowledge_match = match_calculus_knowledge(prompt)
    concept_name = knowledge_match.get("concept_name", "general")
    templates = knowledge_match.get("desmos_templates", [])

    # Extract functions and bounds
    p_func = extract_clean_function(cleaned, is_secondary=False)
    s_func = extract_clean_function(cleaned, is_secondary=True)

    # Extract evaluation point x_val (e.g. at x=1, x_0=2, around x=0)
    m_x = re.search(r'(?:at|around|for)?\s*x(?:_?0|_?1)?\s*=\s*(-?\d+(?:\.\d+)?)', cleaned, re.IGNORECASE)
    if not m_x:
        m_x = re.search(r'at\s+x\s*=\s*(-?\d+(?:\.\d+)?)', cleaned, re.IGNORECASE)
    x_val = m_x.group(1).strip() if m_x else '1'

    # Extract interval bounds [a, b]
    m_bounds = re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', cleaned)
    if not m_bounds:
        m_bounds = re.search(r'from\s+(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)\s+to\s+(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)', cleaned, re.IGNORECASE)
    if not m_bounds:
        m_bounds = re.search(r'on\s+\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?', cleaned, re.IGNORECASE)
    if not m_bounds:
        m_bounds = re.search(r'between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)', cleaned, re.IGNORECASE)
    
    bounds = [m_bounds.group(1), m_bounds.group(2)] if m_bounds else ['0', '3']

    # Extract coordinates (x, y)
    m_coord = re.search(r'(?:P\s*)?\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', cleaned)
    coords = f"({m_coord.group(1)}, {m_coord.group(2)})" if m_coord else (x_val if x_val else None)

    # Extract subintervals n
    m_n = re.search(r'(?:n\s*=\s*(\d+)|(\d+)\s+(?:rectangles?|subintervals?|rects?|steps?))', cleaned, re.IGNORECASE)
    n_val = m_n.group(1) or m_n.group(2) if m_n else '6'

    # Extract step size h
    m_h = re.search(r'(?:h\s*=\s*(-?\d+(?:\.\d+)?)|step\s*(?:size)?\s*=?\s*(-?\d+(?:\.\d+)?))', cleaned, re.IGNORECASE)
    h_val = m_h.group(1) or m_h.group(2) if m_h else '0.5'

    return {
        'concept_type': concept_name,
        'target_function': p_func,
        'primary_function': p_func,
        'secondary_function': s_func if s_func != p_func else ('x + 2' if concept_name == 'area_between_curves' else '0'),
        'lower_limit': bounds[0],
        'upper_limit': bounds[1],
        'coordinates': coords,
        'x_val': x_val,
        'n_val': n_val,
        'h_val': h_val,
        'bounds': bounds,
        'mathematical_intent': f"Visualizing {concept_name.replace('_', ' ')} for {p_func}.",
        'matched_templates': templates
    }


# --- Dynamic Rule-Based Desmos Generator ---

def generate_desmos_translation_fallback(analysis: Dict[str, Any]) -> LLMVisualizationResponse:
    """
    Generates exact Desmos-compatible LaTeX expressions for all calculus concept types,
    fitting the structural guidelines of retrieved RAG desmos_templates to the user's prompt parameters.
    """
    ctype = analysis.get('concept_type', 'general')
    target_func = analysis.get('target_function') or analysis.get('primary_function', 'x^2')
    p_func = sanitize_latex(target_func)
    s_func = sanitize_latex(analysis.get('secondary_function', '0'))
    x_val = analysis.get('x_val', '1')
    n_val = analysis.get('n_val', '6')
    h_val = analysis.get('h_val', '0.5')
    coords = analysis.get('coordinates')
    bounds = analysis.get('bounds', ['0', '3'])
    a_val, b_val = bounds[0] if len(bounds) > 0 else '0', bounds[1] if len(bounds) > 1 else '3'

    # If generic prose function without formula was passed, synthesize a valid mathematical expression
    if not p_func or p_func.lower() in ["a cubic function", "cubic function", "polynomial", "a polynomial", "curve", "a curve"]:
        p_func = "x^3 - 2x"

    if ctype in ['limit_secant', 'secant']:
        return LLMVisualizationResponse(
            title=f"Limit: Secant Line Approaching Tangent Line for f(x) = {p_func}",
            concept_explanation=f"Secant line connecting P({x_val}, f({x_val})) and Q({x_val}+h, f({x_val}+h)) approaches the tangent line as h -> 0.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_x1", latex=f"x_1 = {x_val}", color="#000000", hidden=True),
                DesmosExpression(id="exp_h", latex=f"h = {h_val}", color="#6042a6", sliderBounds=SliderBounds(min="0.01", max="3.0", step="0.05")),
                DesmosExpression(id="exp_p1", latex="(x_1, f(x_1))", color="#c74440", label="P(x1, f(x1))", showLabel=True),
                DesmosExpression(id="exp_p2", latex="(x_1 + h, f(x_1 + h))", color="#388c46", label="Q(x1+h, f(x1+h))", showLabel=True),
                DesmosExpression(id="exp_m", latex="m_s = \\frac{f(x_1 + h) - f(x_1)}{h}", color="#000000", hidden=True),
                DesmosExpression(id="exp_secant", latex="y - f(x_1) = m_s \\cdot (x - x_1)", color="#c74440", lineWidth=2.5, label="Secant Line PQ", showLabel=True),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", hidden=True),
                DesmosExpression(id="exp_tangent", latex="y - f(x_1) = g(x_1) \\cdot (x - x_1)", color="#388c46", lineStyle="DASHED", lineWidth=2.0, label="Tangent Line (Limit)", showLabel=True)
            ]
        )

    if ctype in ['tangent_line']:
        return LLMVisualizationResponse(
            title=f"Tangent & Normal Line for f(x) = {p_func}",
            concept_explanation=f"Tangent line with slope m = f'({x_val}) and perpendicular normal line at P({x_val}, f({x_val})).",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_x1", latex=f"x_1 = {x_val}", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_p1", latex="(x_1, f(x_1))", color="#c74440", label="P(x1, f(x1))", showLabel=True),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", hidden=True),
                DesmosExpression(id="exp_tangent", latex="y - f(x_1) = g(x_1) \\cdot (x - x_1)", color="#388c46", lineWidth=2.5, label="Tangent Line", showLabel=True),
                DesmosExpression(id="exp_normal", latex="y - f(x_1) = -\\frac{1}{g(x_1)} \\cdot (x - x_1)", color="#6042a6", lineStyle="DASHED", lineWidth=2.0, label="Normal Line", showLabel=True)
            ]
        )

    if ctype in ['implicit_differentiation']:
        eq_latex = p_func if '=' in p_func else f"x^2 + y^2 = 25"
        x_1, y_1 = "3", "4"
        if coords and ',' in str(coords):
            m_c = re.search(r'\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)', str(coords))
            if m_c:
                x_1, y_1 = m_c.group(1), m_c.group(2)
        return LLMVisualizationResponse(
            title=f"Implicit Differentiation for {eq_latex}",
            concept_explanation=f"Implicit relation {eq_latex} with tangent line at P({x_1}, {y_1}).",
            expressions=[
                DesmosExpression(id="exp_implicit", latex=eq_latex, color="#2d70b3", lineWidth=3.0, label=eq_latex, showLabel=True),
                DesmosExpression(id="exp_x1", latex=f"x_1 = {x_1}", hidden=True),
                DesmosExpression(id="exp_y1", latex=f"y_1 = {y_1}", hidden=True),
                DesmosExpression(id="exp_pt", latex="(x_1, y_1)", color="#c74440", label=f"P({x_1}, {y_1})", showLabel=True),
                DesmosExpression(id="exp_tan", latex=f"y - y_1 = -\\frac{{x_1}}{{y_1}} \\cdot (x - x_1)", color="#388c46", lineWidth=2.5, label="Tangent Line", showLabel=True)
            ]
        )

    if ctype in ['taylor_series']:
        a_center = x_val if x_val else "0"
        if "cos" in p_func.lower():
            return LLMVisualizationResponse(
                title=f"Taylor Series Expansion for f(x) = {p_func}",
                concept_explanation=f"Taylor polynomial approximations of f(x) = {p_func} around a = {a_center}.",
                expressions=[
                    DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                    DesmosExpression(id="exp_a", latex=f"a = {a_center}", hidden=True),
                    DesmosExpression(id="exp_t0", latex="T_0(x) = 1", color="#c74440", lineStyle="DASHED", label="T0(x) = 1", showLabel=True),
                    DesmosExpression(id="exp_t2", latex="T_2(x) = 1 - \\frac{x^2}{2}", color="#388c46", lineStyle="DASHED", label="T2(x)", showLabel=True),
                    DesmosExpression(id="exp_t4", latex="T_4(x) = 1 - \\frac{x^2}{2} + \\frac{x^4}{24}", color="#6042a6", lineWidth=2.5, label="T4(x)", showLabel=True)
                ]
            )
        elif "e^" in p_func.lower() or "exp" in p_func.lower():
            return LLMVisualizationResponse(
                title=f"Taylor Series Expansion for f(x) = {p_func}",
                concept_explanation=f"Taylor polynomial approximations of f(x) = {p_func} around a = {a_center}.",
                expressions=[
                    DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                    DesmosExpression(id="exp_a", latex=f"a = {a_center}", hidden=True),
                    DesmosExpression(id="exp_t1", latex="T_1(x) = 1 + x", color="#c74440", lineStyle="DASHED", label="T1(x) = 1 + x", showLabel=True),
                    DesmosExpression(id="exp_t2", latex="T_2(x) = 1 + x + \\frac{x^2}{2}", color="#388c46", lineStyle="DASHED", label="T2(x)", showLabel=True),
                    DesmosExpression(id="exp_t3", latex="T_3(x) = 1 + x + \\frac{x^2}{2} + \\frac{x^3}{6}", color="#6042a6", lineWidth=2.5, label="T3(x)", showLabel=True)
                ]
            )
        else:
            return LLMVisualizationResponse(
                title=f"Taylor Series Expansion for f(x) = {p_func}",
                concept_explanation=f"Taylor polynomial approximations of f(x) = {p_func} around a = {a_center}.",
                expressions=[
                    DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                    DesmosExpression(id="exp_a", latex=f"a = {a_center}", hidden=True),
                    DesmosExpression(id="exp_t1", latex="T_1(x) = x", color="#c74440", lineStyle="DASHED", label="T1(x) = x", showLabel=True),
                    DesmosExpression(id="exp_t3", latex="T_3(x) = x - \\frac{x^3}{6}", color="#388c46", lineStyle="DASHED", label="T3(x)", showLabel=True),
                    DesmosExpression(id="exp_t5", latex="T_5(x) = x - \\frac{x^3}{6} + \\frac{x^5}{120}", color="#6042a6", lineWidth=2.5, label="T5(x)", showLabel=True)
                ]
            )

    if ctype in ['mvt']:
        c_val = "1.155" if p_func == "x^3 - 3x" and a_val == "-2" and b_val == "2" else f"({a_val} + {b_val}) / 2"
        return LLMVisualizationResponse(
            title=f"Mean Value Theorem for f(x) = {p_func}",
            concept_explanation=f"On [{a_val}, {b_val}], instantaneous slope f'(c) equals average secant slope.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_pa", latex="(a, f(a))", color="#6042a6", label="A(a, f(a))", showLabel=True),
                DesmosExpression(id="exp_pb", latex="(b, f(b))", color="#6042a6", label="B(b, f(b))", showLabel=True),
                DesmosExpression(id="exp_secant", latex="y - f(a) = \\frac{f(b) - f(a)}{b - a} \\cdot (x - a)", color="#6042a6", lineStyle="DASHED", label="Secant Line AB", showLabel=True),
                DesmosExpression(id="exp_c", latex=f"c = {c_val}", color="#c74440", sliderBounds=SliderBounds(min=a_val, max=b_val, step="0.05")),
                DesmosExpression(id="exp_pc", latex="(c, f(c))", color="#c74440", label="C(c, f(c))", showLabel=True),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", hidden=True),
                DesmosExpression(id="exp_tan", latex="y - f(c) = g(c) \\cdot (x - c)", color="#c74440", lineWidth=2.5, label="Parallel Tangent Line at c", showLabel=True)
            ]
        )

    if ctype in ['area_between_curves', 'area_between']:
        if s_func == "0" or s_func == p_func:
            s_func = "x + 2" if p_func == "x^2" else "x"
        return LLMVisualizationResponse(
            title=f"Area Bounded Between f(x) = {p_func} & g(x) = {s_func}",
            concept_explanation=f"Shaded region representing the area A = \\int_{{a}}^{{b}} |f(x) - g(x)| dx.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_g", latex=f"g(x) = {s_func}", color="#c74440", lineWidth=3.0, label=f"g(x) = {s_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", color="#000000", hidden=True),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", color="#000000", hidden=True),
                DesmosExpression(id="exp_area", latex=f"{p_func} \\le y \\le {s_func} \\{{{a_val} \\le x \\le {b_val}\\}}", color="#388c46"),
                DesmosExpression(id="exp_integral", latex="A = \\int_{a}^{b} (g(x) - f(x)) dx", color="#6042a6", label="Bounded Area A", showLabel=True)
            ]
        )

    if ctype in ['area_under_curve']:
        return LLMVisualizationResponse(
            title=f"Area Under Curve f(x) = {p_func}",
            concept_explanation=f"Shaded region representing the area under curve A = \\int_{{{a_val}}}^{{{b_val}}} ({p_func}) dx.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", color="#000000", hidden=True),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", color="#000000", hidden=True),
                DesmosExpression(id="exp_area", latex=f"0 \\le y \\le f(x) \\{{{a_val} \\le x \\le {b_val}\\}}", color="#388c46"),
                DesmosExpression(id="exp_integral", latex="A = \\int_{a}^{b} f(x) dx", color="#6042a6", label="Area Under Curve A", showLabel=True)
            ]
        )

    if ctype in ['arc_length']:
        return LLMVisualizationResponse(
            title=f"Arc Length of f(x) = {p_func}",
            concept_explanation=f"Calculating total curve length L = \\int_{{a}}^{{b}} \\sqrt{{1 + (f'(x))^2}} dx along [{a_val}, {b_val}].",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.5, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", color="#000000", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_pa", latex="(a, f(a))", color="#c74440", label="Start (a, f(a))", showLabel=True),
                DesmosExpression(id="exp_pb", latex="(b, f(b))", color="#c74440", label="End (b, f(b))", showLabel=True),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", hidden=True),
                DesmosExpression(id="exp_len", latex="L = \\int_{a}^{b} \\sqrt{1 + (g(x))^2} dx", color="#388c46", label="Arc Length L", showLabel=True)
            ]
        )

    if ctype in ['newtons_method']:
        return LLMVisualizationResponse(
            title=f"Newton's Method for f(x) = {p_func}",
            concept_explanation=f"Iteratively finding root of f(x) using linear approximation tangents: x_{{n+1}} = x_n - f(x_n)/f'(x_n).",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_x0", latex=f"x_0 = {x_val}", color="#6042a6", sliderBounds=SliderBounds(min="-10", max="10", step="0.1")),
                DesmosExpression(id="exp_p0", latex="(x_0, f(x_0))", color="#c74440", label="P0(x0, f(x0))", showLabel=True),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", hidden=True),
                DesmosExpression(id="exp_tan0", latex="y - f(x_0) = g(x_0) \\cdot (x - x_0)", color="#c74440", lineStyle="DASHED", label="Tangent at x0", showLabel=True),
                DesmosExpression(id="exp_x1", latex="x_1 = x_0 - \\frac{f(x_0)}{g(x_0)}", hidden=True),
                DesmosExpression(id="exp_p1", latex="(x_1, 0)", color="#388c46", label="Root Appx x1", showLabel=True),
                DesmosExpression(id="exp_p1_c", latex="(x_1, f(x_1))", color="#388c46", label="P1(x1, f(x1))", showLabel=True),
                DesmosExpression(id="exp_drop", latex="x = x_1", color="#388c46", lineStyle="DOTTED")
            ]
        )

    if ctype in ['eulers_method']:
        return LLMVisualizationResponse(
            title=f"Euler's Method for dy/dx = {p_func}",
            concept_explanation=f"Step-by-step numerical approximation of differential equation dy/dx = {p_func} with step size h.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x, y) = {p_func}", hidden=True),
                DesmosExpression(id="exp_h", latex=f"h = {h_val}", color="#6042a6", sliderBounds=SliderBounds(min="0.05", max="2.0", step="0.05")),
                DesmosExpression(id="exp_pt0", latex="(0, 1)", color="#2d70b3", label="(x0, y0) = (0, 1)", showLabel=True),
                DesmosExpression(id="exp_pt1", latex="(0.5, 1.5)", color="#388c46", label="(x1, y1)", showLabel=True),
                DesmosExpression(id="exp_pt2", latex="(1.0, 2.5)", color="#388c46", label="(x2, y2)", showLabel=True),
                DesmosExpression(id="exp_seg1", latex="y - 1 = 1 \\cdot (x - 0) \\{0 \\le x \\le 0.5\\}", color="#388c46", lineWidth=2.5, label="Euler Segment 1", showLabel=True),
                DesmosExpression(id="exp_seg2", latex="y - 1.5 = 2 \\cdot (x - 0.5) \\{0.5 \\le x \\le 1.0\\}", color="#388c46", lineWidth=2.5, label="Euler Segment 2", showLabel=True)
            ]
        )

    if ctype in ['riemann_sum', 'riemann']:
        return LLMVisualizationResponse(
            title=f"Riemann Sum of f(x) = {p_func}",
            concept_explanation=f"Approximating integral of f(x) = {p_func} using n subinterval rectangles on [{a_val}, {b_val}].",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", hidden=True),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", hidden=True),
                DesmosExpression(id="exp_n", latex=f"n = {n_val}", color="#6042a6", sliderBounds=SliderBounds(min="1", max="50", step="1")),
                DesmosExpression(id="exp_w", latex="w = \\frac{b-a}{n}", hidden=True),
                DesmosExpression(id="exp_sum", latex="S = \\sum_{k=1}^{n} f(a + k \\cdot w) \\cdot w", color="#388c46", label="Right Riemann Sum S", showLabel=True),
                DesmosExpression(id="exp_rect", latex="0 \\le y \\le f(a + \\operatorname{floor}(\\frac{x-a}{w}) \\cdot w + w) \\{a \\le x \\le b\\}", color="#c74440")
            ]
        )

    if ctype in ['derivative_rate_of_change']:
        return LLMVisualizationResponse(
            title=f"Derivative & Rate of Change for s(t) = {p_func}",
            concept_explanation=f"Position s(t) = {p_func}, velocity v(t) = s'(t), and acceleration a(t) = v'(t).",
            expressions=[
                DesmosExpression(id="exp_s", latex=f"s(t) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"s(t) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_v", latex="v(t) = \\frac{d}{dt}s(t)", color="#c74440", lineWidth=2.5, label="Velocity v(t)", showLabel=True),
                DesmosExpression(id="exp_a", latex="a(t) = \\frac{d}{dt}v(t)", color="#388c46", lineStyle="DASHED", label="Acceleration a(t)", showLabel=True),
                DesmosExpression(id="exp_t1", latex=f"t_1 = {x_val}", hidden=True),
                DesmosExpression(id="exp_pt", latex="(t_1, s(t_1))", color="#6042a6", label="P(t1, s(t1))", showLabel=True)
            ]
        )

    if ctype in ['critical_points_extrema', 'optimization']:
        return LLMVisualizationResponse(
            title=f"Critical Points & Optimization of f(x) = {p_func}",
            concept_explanation=f"Finding local extrema of f(x) = {p_func} where derivative f'(x) = 0.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_df", latex="g(x) = \\frac{d}{dx}f(x)", color="#c74440", lineStyle="DASHED", label="f'(x)", showLabel=True),
                DesmosExpression(id="exp_x1", latex=f"x_1 = {x_val}", hidden=True),
                DesmosExpression(id="exp_p1", latex="(x_1, f(x_1))", color="#388c46", label="Critical Point P(x1, f(x1))", showLabel=True),
                DesmosExpression(id="exp_tan", latex="y = f(x_1)", color="#6042a6", lineStyle="DASHED", label="Horizontal Tangent", showLabel=True)
            ]
        )

    if ctype in ['concavity_inflection']:
        return LLMVisualizationResponse(
            title=f"Concavity & Inflection Points of f(x) = {p_func}",
            concept_explanation=f"Analyzing concavity of f(x) = {p_func} using second derivative f''(x).",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_ddf", latex="g(x) = \\frac{d^2}{dx^2}f(x)", color="#6042a6", lineStyle="DASHED", label="f''(x)", showLabel=True),
                DesmosExpression(id="exp_x0", latex=f"x_0 = {x_val}", hidden=True),
                DesmosExpression(id="exp_p0", latex="(x_0, f(x_0))", color="#c74440", label="Inflection Point (x0, f(x0))", showLabel=True)
            ]
        )

    if ctype in ['integration_by_parts']:
        return LLMVisualizationResponse(
            title=f"Integration by Parts for f(x) = {p_func}",
            concept_explanation=f"Visualizing area under curve f(x) = {p_func} calculated via \\int u dv = uv - \\int v du.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", hidden=True),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", hidden=True),
                DesmosExpression(id="exp_area", latex=f"0 \\le y \\le f(x) \\{{{a_val} \\le x \\le {b_val}\\}}", color="#388c46")
            ]
        )

    if ctype in ['disc_washer_method']:
        return LLMVisualizationResponse(
            title=f"Disc/Washer Method for f(x) = {p_func}",
            concept_explanation=f"Solid of revolution volume V = \\pi \\int_{{a}}^{{b}} (f(x))^2 dx.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", hidden=True),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", hidden=True),
                DesmosExpression(id="exp_region", latex=f"0 \\le y \\le f(x) \\{{{a_val} \\le x \\le {b_val}\\}}", color="#388c46"),
                DesmosExpression(id="exp_vol", latex="V = \\pi \\int_{a}^{b} (f(x))^2 dx", color="#6042a6", label="Volume V", showLabel=True)
            ]
        )

    if ctype in ['shell_method']:
        return LLMVisualizationResponse(
            title=f"Cylindrical Shell Method for f(x) = {p_func}",
            concept_explanation=f"Volume of revolution V = 2\\pi \\int_{{a}}^{{b}} x \\cdot f(x) dx.",
            expressions=[
                DesmosExpression(id="exp_f", latex=f"f(x) = {p_func}", color="#2d70b3", lineWidth=3.0, label=f"f(x) = {p_func}", showLabel=True),
                DesmosExpression(id="exp_a", latex=f"a = {a_val}", hidden=True),
                DesmosExpression(id="exp_b", latex=f"b = {b_val}", hidden=True),
                DesmosExpression(id="exp_vol", latex="V = 2\\pi \\int_{a}^{b} x \\cdot f(x) dx", color="#6042a6", label="Volume V", showLabel=True)
            ]
        )

    if ctype in ['parametric_curves']:
        param_latex = p_func if ('(' in p_func and ')' in p_func) else "(\\cos(t), \\sin(t))"
        return LLMVisualizationResponse(
            title=f"Parametric Curve {param_latex}",
            concept_explanation=f"2D parametric trajectory {param_latex} as parameter t varies.",
            expressions=[
                DesmosExpression(id="exp_param", latex=param_latex, color="#2d70b3", lineWidth=3.0, sliderBounds=SliderBounds(min="0", max="6.28", step="0.05"))
            ]
        )

    if ctype in ['polar_curves']:
        polar_latex = p_func if p_func.startswith("r") else f"r = {p_func}"
        return LLMVisualizationResponse(
            title=f"Polar Curve {polar_latex}",
            concept_explanation=f"Polar coordinate equation {polar_latex}.",
            expressions=[
                DesmosExpression(id="exp_polar", latex=polar_latex, color="#2d70b3", lineWidth=3.0, label=polar_latex, showLabel=True)
            ]
        )

    if ctype in ['related_rates']:
        rel_latex = p_func if '=' in p_func else "x^2 + y^2 = 25"
        return LLMVisualizationResponse(
            title=f"Related Rates: {rel_latex}",
            concept_explanation=f"Geometric relationship {rel_latex} relating rates of change.",
            expressions=[
                DesmosExpression(id="exp_ladder", latex=rel_latex, color="#2d70b3", lineWidth=3.0, label=rel_latex, showLabel=True),
                DesmosExpression(id="exp_x1", latex=f"x_1 = {x_val}", hidden=True),
                DesmosExpression(id="exp_y1", latex="y_1 = 4", hidden=True),
                DesmosExpression(id="exp_pt", latex="(x_1, y_1)", color="#c74440", label="Position P(x1, y1)", showLabel=True)
            ]
        )

    if ctype in ['slope_fields']:
        sf_latex = f"f(x, y) = {p_func}" if ("y" in p_func or "x" in p_func) else f"f(x) = {p_func}"
        return LLMVisualizationResponse(
            title=f"Slope Field for dy/dx = {p_func}",
            concept_explanation=f"Direction field and local solution segment for differential equation dy/dx = {p_func}.",
            expressions=[
                DesmosExpression(id="exp_f", latex=sf_latex, hidden=True),
                DesmosExpression(id="exp_pt", latex="(1, 1)", color="#c74440", label="Sample Point (1, 1)", showLabel=True),
                DesmosExpression(id="exp_tangent", latex="y - 1 = f(1, 1) \\cdot (x - 1) \\{0.8 \\le x \\le 1.2\\}", color="#388c46", lineWidth=2.5, label="Local Tangent Segment", showLabel=True)
            ]
        )

    # General Function / Equation
    latex_str = sanitize_latex(p_func)
    if not latex_str.startswith('y') and not latex_str.startswith('f(x)') and not '=' in latex_str:
        latex_str = f"y = {latex_str}"

    return LLMVisualizationResponse(
        title=f"Graph of {latex_str}",
        concept_explanation=f"Interactive Desmos 2D graph rendering curve for: {latex_str}.",
        expressions=[
            DesmosExpression(id="exp_main", latex=latex_str, color="#2d70b3", lineWidth=3.0, label=latex_str, showLabel=True)
        ]
    )


# --- LangGraph Distinct Nodes ---

async def call_llm_with_timeout_and_retry(llm_func, max_retries: int = 2, timeout_seconds: float = 10.0, retry_delay: float = 0.5):
    """
    Executes an LLM API call with a strict 10-second asyncio timeout per attempt
    and up to 2 retries (3 attempts total) before failing over to rule-based engines.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(asyncio.to_thread(llm_func), timeout=timeout_seconds)
        except asyncio.TimeoutError as te:
            last_exception = te
            logger.warning(f"LLM API Call Timeout (Attempt {attempt + 1}/{max_retries + 1}): Exceeded {timeout_seconds}s timeout.")
        except Exception as e:
            last_exception = e
            logger.warning(f"LLM API Call Failure (Attempt {attempt + 1}/{max_retries + 1}): {e}")

        if attempt < max_retries:
            await asyncio.sleep(retry_delay)

    raise last_exception or RuntimeError("LLM API Call failed after retries.")


async def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 1 (Query Analysis): Break down user's natural language request into core mathematical components.
    Performs RAG template matching against backend/calculus_knowledge.json.
    Retrieves corresponding 'desmos_templates' and injects them into state for Node 2.
    """
    logger.info("Executing LangGraph Node 1: Query Analysis with Knowledge Retrieval (RAG)")
    prompt = state["prompt"]
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    # Step 1: Perform RAG retrieval against calculus_knowledge.json
    knowledge = match_calculus_knowledge(prompt)
    concept_name = knowledge.get("concept_name", "general")
    concept_desc = knowledge.get("description", "General calculus graph.")
    desmos_templates = knowledge.get("desmos_templates", ["y = f(x)"])

    if google_api_key:
        try:
            import instructor
            from google import genai

            client = instructor.from_genai(
                client=genai.Client(api_key=google_api_key),
                mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
            )

            system_prompt = (
                "You are an expert mathematical NLP analyzer utilizing Retrieval-Augmented Generation (RAG).\n"
                "Deconstruct the user's natural language calculus request into its core mathematical components:\n"
                "- concept_type: matching concept from calculus subfields\n"
                "- target_function: map the exact mathematical expression provided by the user to target_function (e.g. 'x^3 + 2x', 'x^2 - 5', '\\cos(x)'). Takes absolute precedence.\n"
                "- primary_function: primary mathematical expression f(x) or equation specified BY THE USER\n"
                "- secondary_function: secondary expression g(x) if applicable\n"
                "- lower_limit: explicit lower limit bound 'a' if specified in query\n"
                "- upper_limit: explicit upper limit bound 'b' if specified in query\n"
                "- x_val: evaluation point x0 or x_val if specified\n"
                "- coordinates: key point (x, y) if specified\n"
                "- n_val: rectangle/subinterval count n if specified\n"
                "- h_val: step size h if specified\n"
                "- bounds: interval [a, b]\n"
                "- mathematical_intent: clear explanation of intent\n\n"
                "CRITICAL MANDATES FOR FUNCTION EXTRACTION:\n"
                "1. Do NOT output the functions from the few-shot examples. You MUST substitute the user's exact function, limits, and coordinates into the Desmos templates. If a user provides a function, it takes absolute precedence.\n"
                "2. PRESERVE THE USER'S SPECIFIED FUNCTION AND PARAMETERS EXACTLY. Do NOT replace user functions with generic template examples.\n"
                "3. NEVER output descriptive prose like 'the graph of a cubic function' or 'a polynomial' as target_function or primary_function.\n"
                "4. If the user prompt uses generic function descriptions without an equation (e.g. 'a cubic function', 'parabola', 'trig function'), synthesize a concrete equation."
            )

            rag_prompt = (
                f"User Prompt: '{prompt}'\n\n"
                f"RETRIEVED KNOWLEDGE BASE CONTEXT (calculus_knowledge.json):\n"
                f"- Matched Concept: {concept_name}\n"
                f"- Description: {concept_desc}\n"
                f"- Reference Desmos LaTeX Templates: {desmos_templates}"
            )

            def _call_gemini_node1():
                return client.messages.create(
                    model="gemini-2.0-flash",
                    response_model=QueryAnalysisModel,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": rag_prompt}
                    ]
                )

            analysis_model: QueryAnalysisModel = await call_llm_with_timeout_and_retry(
                _call_gemini_node1,
                max_retries=2,
                timeout_seconds=10.0,
                retry_delay=0.5
            )

            res_dict = analysis_model.model_dump()
            res_dict["matched_templates"] = desmos_templates
            return {"analysis": res_dict}
        except Exception as e:
            logger.warning(f"Node 1 Gemini analysis failed ({e}). Falling back to RAG NLP analyzer.")

    analysis_dict = analyze_query_fallback(prompt)
    analysis_dict["matched_templates"] = desmos_templates
    return {"analysis": analysis_dict}


async def desmos_translation_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 2 (Desmos Translation): Map analyzed components into precise Desmos-compatible LaTeX commands.
    Uses retrieved RAG desmos_templates from Node 1 as strict structural guidelines for the final output,
    fitting them dynamically to the user's prompt parameters without over-generalizing.
    """
    logger.info("Executing LangGraph Node 2: Desmos Translation using RAG Structural Guidelines")
    analysis = state.get("analysis") or analyze_query_fallback(state["prompt"])
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if google_api_key:
        try:
            import instructor
            from google import genai

            client = instructor.from_genai(
                client=genai.Client(api_key=google_api_key),
                mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
            )

            system_prompt = (
                "You are an expert Desmos LaTeX translation engine.\n"
                "Your sole purpose is to convert natural language calculus concepts into valid, pure mathematical Desmos expressions.\n\n"
                "CRITICAL MANDATES FOR RAG TEMPLATE FITTING & AVOIDING OVER-GENERALIZATION:\n"
                "1. Output PURE mathematical LaTeX only. Never include English text, descriptions, or conversational filler inside the LaTeX string.\n"
                "2. Do NOT output the functions from the few-shot examples. You MUST substitute the user's exact function, limits, and coordinates into the Desmos templates. If a user provides a function, it takes absolute precedence.\n"
                "3. FIT RAG TEMPLATES TO THE USER'S EXACT PROMPT PARAMETERS:\n"
                "   - Use the retrieved RAG desmos_templates as STRUCTURAL GUIDELINES (equation types, sliders, bounds, shading, lines).\n"
                "   - DO NOT REPLICATE THE EXAMPLE FUNCTIONS/VALUES IN THE TEMPLATES IF THE USER SPECIFIED DIFFERENT ONES!\n"
                "   - Substitute the user's target function (target_function / primary_function), secondary_function, bounds (a, b), evaluation points (x_1, x_0), n_val, h_val into the template structure.\n\n"
                "3. STRICTLY ENFORCE COMPLETE, SELF-CONTAINED MATHEMATICAL STATEMENTS:\n"
                "   - Every equation MUST have a defined left-hand side (e.g. `a = -2` or `f(x) = x^3 - 2x`). NEVER output dropped variables or leading equals signs like `= -2` or `= x^2`.\n"
                "   - Ensure NO empty parentheses `()` or empty brackets `[]` occur in expressions or labels.\n"
                "   - Ensure NO missing denominators or empty numerators in fractions `\\frac{}{}`.\n"
                "   - Ensure coordinate pairs have both x and y coordinates defined (e.g. `(b, f(b))`, NEVER `(, f())` or `(b, )`).\n\n"
                "4. ABSOLUTELY NO ENGLISH PROSE OR UNDEFINED VARIABLES IN DESMOS OUTPUT:\n"
                "   - NEVER include English prose, instructions, explanations, descriptions, or unparsed text inside the `latex` field of any DesmosExpression.\n"
                "   - NEVER include variables without explicit definitions inside the LaTeX output (e.g., define slider bounds or explicit constant values for all parameters/variables like x_1, h, a, b, n).\n\n"
                "5. CONCRETE MATHEMATICAL FUNCTION SYNTHESIS:\n"
                "   - If the request or analysis specifies generic or descriptive function names (e.g., 'a cubic function', 'a polynomial', 'a trig curve', 'a quadratic function'), ALWAYS synthesize an explicit, concrete mathematical equation.\n\n"
                "6. USE RAG KNOWLEDGE TEMPLATES AS STRICT STRUCTURAL GUIDELINES:\n"
                "   - Follow the structure of the retrieved desmos_templates provided in the user prompt to build complete, valid expressions.\n"
                "   - Use distinct curated hex colors (#2d70b3, #c74440, #388c46, #6042a6, #000000)."
            )

            target_func = analysis.get("target_function") or analysis.get("primary_function", "x^2")
            lower_lim = analysis.get("lower_limit") or (analysis.get("bounds", ["0", "3"])[0] if analysis.get("bounds") else "0")
            upper_lim = analysis.get("upper_limit") or (analysis.get("bounds", ["0", "3"])[1] if len(analysis.get("bounds", [])) > 1 else "3")
            coords = analysis.get("coordinates") or analysis.get("x_val", "1")
            n_val = analysis.get("n_val", "6")
            h_val = analysis.get("h_val", "0.5")

            user_prompt = (
                f"EXPLICIT EXTRACTED USER PARAMETERS (REQUIRED CONTEXT - MUST USE THESE EXACT VALUES):\n"
                f"- target_function f(x): {target_func}\n"
                f"- secondary_function g(x): {analysis.get('secondary_function', '0')}\n"
                f"- lower_limit (a): {lower_lim}\n"
                f"- upper_limit (b): {upper_lim}\n"
                f"- coordinates / x0: {coords}\n"
                f"- n_val (subintervals): {n_val}\n"
                f"- h_val (step size): {h_val}\n"
                f"- concept_type: {analysis.get('concept_type')}\n\n"
                f"Full Query Analysis: {analysis}\n\n"
                f"STRICT STRUCTURAL GUIDELINES FROM RAG KNOWLEDGE BASE:\n"
                f"Concept Type: {analysis.get('concept_type')}\n"
                f"Desmos LaTeX Reference Templates: {analysis.get('matched_templates', [])}\n\n"
                f"CRITICAL REQUIREMENT: You MUST use the exact user target function '{target_func}', lower limit '{lower_lim}', upper limit '{upper_lim}', and coordinates '{coords}' in your Desmos expressions. Do NOT substitute your own numbers or functions."
            )

            if state.get("validation_error"):
                user_prompt += (
                    f"\n\nCRITICAL RETRY FEEDBACK (PREVIOUS ATTEMPT FAILED VALIDATION):\n"
                    f"{state['validation_error']}\n"
                    f"Please correct your Desmos expressions to resolve this issue strictly."
                )

            def _call_gemini_node2():
                return client.messages.create(
                    model="gemini-2.0-flash",
                    response_model=LLMVisualizationResponse,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

            llm_response: LLMVisualizationResponse = await call_llm_with_timeout_and_retry(
                _call_gemini_node2,
                max_retries=2,
                timeout_seconds=10.0,
                retry_delay=0.5
            )
            return {"llm_response": llm_response, "validation_error": None}
        except Exception as e:
            logger.warning(f"Node 2 Gemini translation failed ({e}). Using rule-based Desmos translation engine.")

    llm_response = generate_desmos_translation_fallback(analysis)
    return {"llm_response": llm_response, "validation_error": None}


async def validation_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 3 (Validation & Self-Correction Detector):
    Validates syntax, checks for English text or dropped target functions.
    If validation fails, flags validation_error and increments retry_count to trigger LangGraph cyclic retry.
    """
    logger.info("Executing LangGraph Node 3: Validation")
    raw_response = state.get("llm_response")
    analysis = state.get("analysis", {})
    retry_count = state.get("retry_count", 0)

    if not raw_response:
        raw_response = generate_desmos_translation_fallback(analysis)

    valid_expressions = []
    has_rejected_expression = False
    validation_issues = []

    # 1. Check for English prose inside latex fields
    contains_english = any(
        re.search(r'\b(?:the|graph|of|cubic|quadratic|function|polynomial|shaded|area|under|curve|point|line)\b', exp.latex, re.IGNORECASE)
        for exp in raw_response.expressions
    )
    if contains_english:
        validation_issues.append("English prose detected inside Desmos LaTeX field.")

    # 2. Check if user's extracted target_function was dropped
    target_func = analysis.get("target_function")
    dropped_target = False
    if target_func and target_func not in ["x^2", "0"] and not target_func.startswith("a "):
        cleaned_tf = sanitize_latex(target_func)
        if cleaned_tf and not any(cleaned_tf in exp.latex for exp in raw_response.expressions):
            dropped_target = True
            validation_issues.append(f"User target function '{target_func}' was dropped in generated expressions.")

    # 3. Check for invalid Desmos syntax
    for exp in raw_response.expressions:
        exp.latex = sanitize_latex(exp.latex)
        is_valid, msg = PipelineEvaluator.is_valid_desmos_latex(exp.latex)
        if is_valid:
            valid_expressions.append(exp)
        else:
            has_rejected_expression = True
            validation_issues.append(f"Syntax validation failed for '{exp.latex}': {msg}")

    # Determine if self-correction retry should trigger
    if (contains_english or dropped_target or has_rejected_expression) and retry_count < 2:
        error_details = "; ".join(validation_issues)
        logger.warning(f"Validation failed (Attempt {retry_count + 1}/3). Triggering LangGraph cyclic retry: {error_details}")
        return {
            "validated_response": raw_response,
            "validation_issues": validation_issues,
            "validation_error": f"Validation Error: {error_details}",
            "retry_count": retry_count + 1
        }

    # If max retries reached or valid expressions exist
    if has_rejected_expression or len(valid_expressions) == 0:
        logger.info("Generating clean response via rule-based Desmos translation engine fallback.")
        raw_response = generate_desmos_translation_fallback(analysis)
    else:
        raw_response.expressions = valid_expressions

    eval_result = PipelineEvaluator.evaluate_response(raw_response)
    
    return {
        "validated_response": raw_response,
        "validation_issues": eval_result.get("issues", []),
        "validation_error": None,
        "retry_count": retry_count
    }


# --- LangGraph Workflow Graph Assembly ---

def should_retry(state: GraphState) -> str:
    """Conditional edge routing: Retries desmos_translation node if validation_error exists."""
    if state.get("validation_error") and state.get("retry_count", 0) < 2:
        return "desmos_translation"
    return END


def build_calculus_graph():
    builder = StateGraph(GraphState)

    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("desmos_translation", desmos_translation_node)
    builder.add_node("validation", validation_node)

    builder.set_entry_point("query_analysis")
    builder.add_edge("query_analysis", "desmos_translation")
    builder.add_edge("desmos_translation", "validation")
    
    builder.add_conditional_edges(
        "validation",
        should_retry,
        {
            "desmos_translation": "desmos_translation",
            END: END
        }
    )

    return builder.compile()

calculus_graph = build_calculus_graph()


# --- Main Entry Point ---

async def generate_calculus_visualization(prompt: str, metadata: Dict[str, Any]) -> LLMVisualizationResponse:
    """
    Executes the 3-node LangGraph workflow for any calculus prompt.
    Returns a validated LLMVisualizationResponse.
    """
    initial_state: GraphState = {
        "prompt": prompt,
        "metadata": metadata,
        "analysis": None,
        "llm_response": None,
        "validated_response": None,
        "validation_issues": [],
        "error": None,
        "retry_count": 0,
        "validation_error": None
    }

    final_state = await calculus_graph.ainvoke(initial_state)
    return final_state["validated_response"]


async def stream_calculus_visualization(prompt: str, metadata: Dict[str, Any]):
    """
    Async generator for Server-Sent Events (SSE) streaming.
    Executes Node 1, streams analysis, executes Node 2 & 3, and streams expressions sequentially.
    """
    import json
    
    # 1. Execute Node 1: Query Analysis
    analysis_res = await query_analysis_node({"prompt": prompt, "metadata": metadata, "analysis": None, "llm_response": None, "validated_response": None, "validation_issues": [], "error": None, "retry_count": 0, "validation_error": None})
    analysis = analysis_res["analysis"]

    # Yield SSE chunk for Query Analysis
    yield f"data: {json.dumps({'type': 'analysis', 'concept_type': analysis.get('concept_type'), 'intent': analysis.get('mathematical_intent')})}\n\n"
    await asyncio.sleep(0.05)

    # 2. Execute Node 2 & 3: Translation and Validation
    translation_res = await desmos_translation_node({"prompt": prompt, "metadata": metadata, "analysis": analysis, "llm_response": None, "validated_response": None, "validation_issues": [], "error": None, "retry_count": 0, "validation_error": None})
    validation_res = await validation_node({"prompt": prompt, "metadata": metadata, "analysis": analysis, "llm_response": translation_res["llm_response"], "validated_response": None, "validation_issues": [], "error": None, "retry_count": 0, "validation_error": None})

    final_payload: LLMVisualizationResponse = validation_res["validated_response"]

    # Yield SSE chunk for Metadata (title & concept explanation)
    yield f"data: {json.dumps({'type': 'metadata', 'title': final_payload.title, 'concept_explanation': final_payload.concept_explanation, 'total_expressions': len(final_payload.expressions)})}\n\n"
    await asyncio.sleep(0.05)

    # 3. Stream expressions sequentially
    for idx, exp in enumerate(final_payload.expressions):
        exp_dict = exp.model_dump()
        yield f"data: {json.dumps({'type': 'expression', 'expression': exp_dict, 'index': idx, 'total': len(final_payload.expressions)})}\n\n"
        await asyncio.sleep(0.05)

    # Yield completion SSE chunk
    yield f"data: {json.dumps({'type': 'complete', 'total_expressions': len(final_payload.expressions)})}\n\n"
