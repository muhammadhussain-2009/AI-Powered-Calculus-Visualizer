import re
from typing import List, Dict, Any, Tuple
from backend.schemas import LLMVisualizationResponse, DesmosExpression

class PipelineEvaluator:
    """
    Evaluator for checking mathematical validity, LaTeX syntax, and Desmos compatibility
    of AI-generated visualization payloads.
    """

    @staticmethod
    def check_balanced_delimiters(latex_str: str) -> bool:
        """Verifies that braces {}, parentheses (), and brackets [] are balanced."""
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for char in latex_str:
            if char in "({[":
                stack.append(char)
            elif char in ")}]" :
                if not stack or stack[-1] != mapping[char]:
                    return False
                stack.pop()
        return len(stack) == 0

    @staticmethod
    def is_valid_desmos_latex(latex_str: str) -> Tuple[bool, str]:
        """
        Validates individual LaTeX string for common rendering breakages.
        """
        if not latex_str or not latex_str.strip():
            return False, "Empty LaTeX string."

        # Check balanced braces
        if not PipelineEvaluator.check_balanced_delimiters(latex_str):
            return False, "Unbalanced braces or parentheses in LaTeX."

        # Check for unescaped illegal backslashes or broken LaTeX commands
        if re.search(r'\\(?:[a-zA-Z]+)?$', latex_str):
            return False, "Trailing unclosed backslash sequence."

        # Verify no markdown or non-LaTeX artifacts
        if "```" in latex_str or "JSON" in latex_str or "html" in latex_str:
            return False, "Contains raw code fence or text artifacts."

        return True, "Valid"

    @classmethod
    def evaluate_response(cls, response: LLMVisualizationResponse) -> Dict[str, Any]:
        """
        Evaluates an entire LLMVisualizationResponse payload.
        Returns score (0.0 - 1.0), passed boolean, and detailed issue reports.
        """
        issues = []
        valid_expressions_count = 0
        total_expressions = len(response.expressions)

        if total_expressions == 0:
            return {
                "score": 0.0,
                "passed": False,
                "issues": ["Response contains no Desmos expressions."]
            }

        for idx, exp in enumerate(response.expressions):
            is_valid, msg = cls.is_valid_desmos_latex(exp.latex)
            if is_valid:
                valid_expressions_count += 1
            else:
                issues.append(f"Expression #{idx+1} (id={exp.id}): {msg}")

        # Basic score based on percentage of valid expressions
        score = valid_expressions_count / total_expressions

        # Ensure title and explanation are non-trivial
        if len(response.title.strip()) < 3:
            score -= 0.1
            issues.append("Title is too short or empty.")
        if len(response.concept_explanation.strip()) < 10:
            score -= 0.1
            issues.append("Explanation is too short.")

        score = max(0.0, min(1.0, score))
        passed = score >= 0.8 and len(issues) == 0

        return {
            "score": round(score, 2),
            "passed": passed,
            "valid_expressions": valid_expressions_count,
            "total_expressions": total_expressions,
            "issues": issues
        }
