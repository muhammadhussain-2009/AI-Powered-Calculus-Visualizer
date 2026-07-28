# Agent Instructions

## Product Manager Agent
* Role: Ensure all implemented features align with the PRD.
* Action: Before concluding a phase, verify that the acceptance criteria are met and no scope creep has occurred.

## Backend Python Agent
* Role: Build the FastAPI server, AI pipeline, and security sanitization layer.
* Action: Prioritize Pydantic schemas. Ensure `ai_pipeline.py` strictly returns validated LaTeX commands.

## Frontend UI Agent
* Role: Develop the HTML/JS interface and Desmos integration.
* Action: Ensure the Desmos calculator instance updates cleanly without page reloads using `calculator.setBlank()` and `calculator.setExpression()`.

## QA & Testing Agent
* Role: Validate system integrity.
* Action: Write pytest functions to verify API payload structures and regex sanitization effectiveness.
