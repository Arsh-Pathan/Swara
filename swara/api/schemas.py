"""Pydantic schemas for the Swara HTTP REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    service: str = "swara-speaker-intelligence"


class ReadyResponse(BaseModel):
    ready: bool
    models_loaded: bool
    device: str
    active_model: str
    enrolled_speakers_count: int


class ModelItem(BaseModel):
    model_id: str
    filename: str
    architecture: str
    size_mb: float
    license: str
    is_downloaded: bool
    is_verified: bool
    is_default: bool
    path: Optional[str] = None


class SpeakerResponse(BaseModel):
    speaker_id: str
    name: str
    created_at: str
    updated_at: str
    sample_count: int
    total_audio_duration_s: float
    embedding_dim: int
    model_version: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateMatchResponse(BaseModel):
    speaker_id: str
    similarity: float
    confidence: float
    is_match: bool


class IdentificationResponse(BaseModel):
    speaker_id: str
    enrolled_id: Optional[str]
    similarity: Optional[float]
    confidence: Optional[float]
    is_match: bool
    audio_duration: float
    candidates: List[CandidateMatchResponse] = Field(default_factory=list)


class VerificationResponse(BaseModel):
    speaker_id: str
    verified: bool
    similarity: float
    confidence: float
    threshold: float


class DiarizationSegmentResponse(BaseModel):
    start: float
    end: float
    duration: float
    speaker_id: str
    identity: Optional[str] = None
    similarity: Optional[float] = None


class DiarizationResponse(BaseModel):
    num_speakers: int
    duration: float
    audio_path: Optional[str] = None
    segments: List[DiarizationSegmentResponse] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
