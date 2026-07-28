# Product Requirements Document: Calculus Visualizer MVP

## Vision
To address learning gaps in STEM education by converting abstract calculus concepts into tangible, interactive visualizations using LLM-generated Desmos commands. 

## Core Functionality
* Users input a natural language calculus concept (e.g., "Riemann sum of x^2").
* The backend AI translates this prompt into a structured JSON array of Desmos-compatible LaTeX expressions.
* The frontend injects these commands into an embedded Desmos graphing calculator.
* The calculator renders the visual output concurrently.

## Target Audience
STEM students, educators, and researchers looking for visual feedback tools for complex mathematical properties.

## Success Metrics
* Under 2-second response time from prompt submission to visual rendering.
* 100% syntactically valid JSON output from the AI pipeline.
* Zero occurrences of unescaped, malformed LaTeX breaking the frontend script.
