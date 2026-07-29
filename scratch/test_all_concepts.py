import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from backend.ai_pipeline import generate_calculus_visualization, analyze_query_fallback, generate_desmos_translation_fallback

prompts = [
    "Implicit differentiation for x^2 + y^2 = 25",
    "Taylor series expansion for sin(x)",
    "Optimization and critical points of x^3 - 3x",
    "Slope field for dy/dx = x + y",
    "Riemann sum of x^2 from 0 to 3",
    "Newton's method for x^3 - 2x - 5",
    "Euler's method for dy/dx = x + y",
    "Area between f(x) = x + 2 and g(x) = x^2",
    "Arc length of x^(3/2) from 0 to 4",
    "Limit secant line approaching tangent line for x^2 at x = 1",
    "Mean Value Theorem for x^3 - 3x on [-2, 2]",
    "Derivative rate of change for s(t) = -16t^2 + 64t",
    "Critical points and extrema of 2x^3 - 3x^2 - 12x + 5",
    "Concavity and inflection points of x^3 - 3x",
    "Integration by parts for x * sin(x)",
    "Disc washer method for sqrt(x)",
    "Shell method for volume",
    "Parametric curves x = cos(t), y = sin(t)",
    "Polar curves r = 1 + cos(theta)",
    "Related rates ladder sliding down wall",
    "cubic function",
    "plot a line y = 2x + 5",
    "graph of a polynomial"
]

async def run_all():
    print("=== TESTING ALL CONCEPTS AND PROMPTS ===")
    for p in prompts:
        print(f"\n--- PROMPT: '{p}' ---")
        try:
            res = await generate_calculus_visualization(p, {})
            print(f"TITLE: {res.title}")
            for idx, exp in enumerate(res.expressions):
                print(f"  [{idx+1}] id={exp.id} | latex='{exp.latex}' | label='{exp.label}' | slider={exp.sliderBounds}")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(run_all())
