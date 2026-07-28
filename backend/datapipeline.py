import re
from typing import Tuple, Dict, Any
from security.protection import sanitize_user_input

# Key calculus and mathematical keywords/patterns
MATH_KEYWORDS = [
    r"riemann", r"integral", r"integrate", r"derivative", r"differentiate",
    r"limit", r"tangent", r"secant", r"slope", r"area under", r"taylor",
    r"maclaurin", r"series", r"f\(x\)", r"g\(x\)", r"y\s*=", r"x\^",
    r"sin", r"cos", r"tan", r"sec", r"csc", r"cot", r"ln", r"log", r"exp",
    r"polynomial", r"quadratic", r"cubic", r"rational", r"asymptote",
    r"inflection", r"concave", r"convex", r"extrema", r"maximum", r"minimum",
    r"optimization", r"related rates", r"arc length", r"surface area",
    r"volume of revolution", r"washer method", r"shell method", r"vector",
    r"gradient", r"curl", r"divergence", r"partial derivative", r"differential",
    r"f\'", r"f\'\'", r"dy/dx"
]

COMMON_MATH_SYMBOLS = [
    "+", "-", "*", "/", "^", "=", "<", ">", "∫", "∑", "lim", "d/dx", "dx", "dy", "π", "e"
]

def is_math_related(prompt: str) -> bool:
    """
    Verifies whether the prompt contains math/calculus related terminology, equations, or symbols.
    """
    if not prompt or len(prompt.strip()) < 2:
        return False
        
    lowered = prompt.lower().strip()
    
    # Check for keyword matches
    for pattern in MATH_KEYWORDS:
        if re.search(pattern, lowered):
            return True
            
    # Check for equations and math operators (e.g. y = 3x + 34, 2x + 3y = 12, x^2 = 9, 3x+5)
    if re.search(r'[0-9xXyYtT]\s*[\+\-\*\/\^\=]\s*[0-9xXyYtT]', prompt):
        return True

    # Check for explicit equals sign with variable/numbers (e.g. y=..., x=..., f(x)=...)
    if re.search(r'[a-zA-Z0-9\)]\s*=\s*[a-zA-Z0-9\(\-]', prompt):
        return True

    # Check for explicit function/variable notation e.g. x^2, sin(x), y=...
    if re.search(r'([a-zA-Z]\([a-zA-Z]\)|[xXyY]\s*[\^=]|\b(x|y|t|theta)\b)', lowered):
        return True
        
    return False

def extract_prompt_metadata(prompt: str) -> Dict[str, Any]:
    """
    Extracts structural components, target topic, domain limits, and hint context for AI.
    """
    lowered = prompt.lower()
    detected_topics = []
    
    if "riemann" in lowered or "rectangle" in lowered:
        detected_topics.append("riemann_sum")
    if "derivative" in lowered or "tangent" in lowered or "slope" in lowered:
        detected_topics.append("differentiation")
    if "integral" in lowered or "area under" in lowered:
        detected_topics.append("integration")
    if "taylor" in lowered or "series" in lowered:
        detected_topics.append("taylor_series")
    if "limit" in lowered:
        detected_topics.append("limits")
    if "vector" in lowered or "gradient" in lowered:
        detected_topics.append("multivariable")

    # Extract function if explicitly defined like f(x) = x^2 or y = sin(x)
    func_match = re.search(r'(f\(x\)|y)\s*=\s*([^\,;\.\n]+)', prompt, re.IGNORECASE)
    extracted_func = func_match.group(2).strip() if func_match else None

    return {
        "raw_prompt": prompt,
        "sanitized_prompt": sanitize_user_input(prompt),
        "detected_topics": detected_topics if detected_topics else ["general_calculus"],
        "extracted_function": extracted_func,
        "is_math": True
    }

def process_and_verify_request(raw_prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Main pipeline entrypoint to sanitize, verify math relevance, and extract prompt metadata.
    Returns: (is_valid: bool, sanitized_prompt: str, metadata: dict)
    """
    sanitized = sanitize_user_input(raw_prompt)
    if not sanitized:
        return False, "", {"error": "Prompt cannot be empty after sanitization."}
        
    if not is_math_related(sanitized):
        return False, sanitized, {
            "error": "Prompt does not appear to be related to calculus or mathematics. Please enter a calculus concept or mathematical function."
        }
        
    metadata = extract_prompt_metadata(sanitized)
    return True, sanitized, metadata
