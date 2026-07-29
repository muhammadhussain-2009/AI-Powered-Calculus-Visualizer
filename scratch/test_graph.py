import sys
import os
sys.path.insert(0, os.getcwd())

import asyncio
from backend.ai_pipeline import build_calculus_graph

async def run():
    graph = build_calculus_graph()
    state = {
        'prompt': 'Mean Value Theorem for f(x) = x^3 - 3x on [-2, 2]',
        'metadata': {},
        'analysis': None,
        'llm_response': None,
        'validated_response': None,
        "validation_issues": [],
        'error': None,
        'retry_count': 0,
        'validation_error': None
    }
    async for step in graph.astream(state):
        print("STEP:", step)

if __name__ == "__main__":
    asyncio.run(run())
