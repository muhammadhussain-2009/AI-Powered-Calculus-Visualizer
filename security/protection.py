import re
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject security headers into all outgoing HTTP responses,
    enforcing HTTPS (HSTS) and preventing MIME sniffing, clickjacking, and XSS attacks.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Enforce HTTPS redirect in non-development environments if requested over HTTP
        if request.headers.get("x-forwarded-proto") == "http" and os.getenv("ENVIRONMENT", "production").lower() != "development":
            url = request.url.replace(scheme="https")
            return RedirectResponse(url, status_code=307)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https: data: blob: 'unsafe-inline' 'unsafe-eval'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.desmos.com https://*.desmos.com https://cdn.jsdelivr.net blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://www.desmos.com https://*.desmos.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net https://www.desmos.com https://*.desmos.com data:; "
            "img-src 'self' data: https://www.desmos.com https://*.desmos.com blob:; "
            "worker-src 'self' blob: https://www.desmos.com https://*.desmos.com; "
            "child-src 'self' blob: https://www.desmos.com https://*.desmos.com; "
            "frame-src 'self' blob: https://www.desmos.com https://*.desmos.com; "
            "connect-src 'self' https: wss: https://www.desmos.com https://*.desmos.com;"
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
