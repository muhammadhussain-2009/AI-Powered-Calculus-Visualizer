import os
import time
import uuid
import jwt
from typing import Optional
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

JWT_SECRET = os.getenv("JWT_SECRET", "calculus-visualizer-secret-key-change-in-prod-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

security_bearer = HTTPBearer(auto_error=False)

def create_anonymous_jwt_token() -> str:
    """
    Generates a JWT token for tracking user sessions without requiring sign-up.
    """
    now = time.time()
    payload = {
        "session_id": str(uuid.uuid4()),
        "iat": now,
        "exp": now + (JWT_EXPIRATION_HOURS * 3600),
        "type": "anonymous_session"
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(token: str) -> dict:
    """
    Verifies JWT token validity and returns decoded payload.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.")

async def get_current_session(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer)) -> dict:
    """
    FastAPI dependency to extract or generate session context.
    """
    if not credentials or not credentials.credentials:
        # Generate new anonymous session token if missing
        token = create_anonymous_jwt_token()
        payload = verify_jwt_token(token)
        payload["new_token"] = token
        return payload
    
    return verify_jwt_token(credentials.credentials)
