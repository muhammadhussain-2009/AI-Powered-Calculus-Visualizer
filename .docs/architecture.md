# Architecture and Project Structure

## System Flow
1. User submits text via Frontend UI.
2. Frontend sends POST request to FastAPI Backend (`/api/visualize`).
3. Backend routes prompt to the AI Pipeline (LangChain + Instructor).
4. AI generates Pydantic-validated JSON containing LaTeX strings.
5. Backend sanitizes LaTeX using Regex and returns payload.
6. Frontend parses JSON and pushes updates to Desmos API via `setExpression`.

## Folder and Project Structure 

```text
AI-Powered-Calculus-Visualizer/
├── backend/
│   ├── main.py
│   ├── ai_pipeline.py
│   ├── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/
│   ├── test_api.py
│   └── test_eval.py
├── security/
│   ├── __init__.py 
│   ├── auth.py
│   ├── database_sec.py
│   └── protection.py
├── docs/
│   ├── agents.md 
│   ├── skill.md
│   ├── architecture.md
│   ├── phases.md 
│   ├── rules.md 
│   └── prd.md
├── Dockerfile 
├── docker-compose.yml
├── gitignore
├── .env.example
├── .env
├── run.py
└── requirements.txt
```

## Tech Stack
* Backend: Python 3.11+, FastAPI, Uvicorn, Pydantic, SQL Alchemy
* AI Pipeline: LangChain, Instructor, Google API Key
* Frontend: HTML5, Vanilla JavaScript, Desmos API.
* Database: SQLite Database
* Testing: Pytest, DeepEval.
* Containerization: Docker 
