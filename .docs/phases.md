# Project Phases

## Phase 1: Foundation & Schemas
* Define Pydantic schemas in `schemas.py` for API requests, Desmos commands, and LLM outputs.
* Set up basic FastAPI server in `main.py` with a health-check endpoint.

## Phase 2: AI Pipeline & Backend Logic
* Implement `ai_pipeline.py` using Instructor to query the LLM.
* Create the `/api/visualize` POST endpoint.
* Implement the regex sanitization function for LaTeX strings.
* Containerise with Docker and create a Docker Compose File 

## Phase 3: Frontend Interface
* Build `index.html` with the input form and layout structure.
* Initialize the Desmos API calculator instance.
* Write JavaScript to fetch data from the backend and populate the calculator.

## Phase 4: Testing & Evaluation
* Write unit tests for API endpoints.
* Implement DeepEval checks to verify the mathematical accuracy of generated commands.
* Perform end-to-end testing with sample calculus concepts.

## Phase 5: Handoff & Memory Generation
* Review the completed implementation against the PRD.
* Generate a `memory.md` file in the root directory. This file must summarize:
  * The current state of the project and working features.
  * Any deviations made from the original architecture.
  * Commands needed to run the backend and frontend locally.
  * Next steps or known edge cases (e.g., specific LaTeX strings that still need better sanitization).
