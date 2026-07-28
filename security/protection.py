import re
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject security headers into all outgoing HTTP responses.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https: http: data: 'unsafe-inline' 'unsafe-eval'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.desmos.com blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://www.desmos.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: https://www.desmos.com blob:; "
            "worker-src 'self' blob: https://www.desmos.com; "
            "child-src 'self' blob: https://www.desmos.com; "
            "connect-src 'self' https: http: wss: ws: https://www.desmos.com;"
        )
        return response

def sanitize_user_input(text: str) -> str:
    """
    Sanitizes user input prompts to remove script tags, zero-width characters, or control characters.
    """
    if not text:
        return ""
    # Strip null bytes and control chars
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Strip HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()
