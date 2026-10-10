import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()

def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    expected_key = os.environ.get("ADMIN_API_KEY")
    if not expected_key or not x_admin_key or x_admin_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing admin API key"
        )