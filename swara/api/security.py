"""Security utilities, API key authentication, and request validation."""

from typing import Optional
from fastapi import Header, HTTPException, Request, status

from swara.core.config import settings


async def verify_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> None:
    """Validates API Key if configured in settings.

    Supports either 'Authorization: Bearer <token>' or 'X-API-Key: <token>'.
    If settings.api_key is None, authentication is skipped (open for local dev).
    """
    configured_key = settings.api_key
    if not configured_key:
        return

    provided_key = None
    if x_api_key:
        provided_key = x_api_key
    elif authorization and authorization.lower().startswith("bearer "):
        provided_key = authorization.split(" ", 1)[1].strip()

    if not provided_key or provided_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_audio_content_type(content_type: Optional[str]) -> None:
    """Validates that uploaded file has an acceptable audio media type."""
    if not content_type:
        return

    valid_types = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/flac",
        "audio/ogg",
        "audio/mpeg",
        "application/octet-stream",
    ]
    if not any(content_type.lower().startswith(vt) for vt in valid_types):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio media type '{content_type}'. Supported: wav, flac, ogg, mpeg.",
        )
