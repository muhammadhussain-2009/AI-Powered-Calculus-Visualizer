# Development Rules

## What to Use
* Strictly enforce types using Pydantic models for all API requests and AI outputs.
* Use regex to sanitize all AI-generated LaTeX before sending it to the frontend.
* Write modular, test-driven code; keep endpoint logic separate from AI logic.
* Always conclude the final implementation phase by generating a `memory.md` file. This acts as the save-state for the CLI's context so future sessions can resume seamlessly without rereading the entire codebase.
1
## What to Avoid
* Do not allow the LLM to generate unstructured text; always use Instructor to force JSON schema adherence.
* Do not use complex frontend frameworks (React, Vue) for this MVP; stick to lightweight HTML/JS to prioritize the Desmos integration.
* Do not blindly trust AI outputs; the backend must validate mathematical symbols before responding.
* Avoid exposing API keys in the frontend code; handle all LLM routing on the backend.
* Make sure to sanitise all user requests, do not leave any debug or admin endpoints in the frontend 
* This project will have no sign ups required from users, but use relevant JWTs to track user requests  
* you do not have basic security headers on please turn them on 
* you do not implement rate limiting for the APIs (please implement proper rate limiting using SLOWAPI or any other dependency of your choice)
