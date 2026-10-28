"""FastAPI route handlers for Swara."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from swara import __version__
from swara.api.schemas import (
    CandidateMatchResponse,
    DiarizationResponse,
    DiarizationSegmentResponse,
    HealthResponse,
    IdentificationResponse,
    ModelItem,
    ReadyResponse,
    SpeakerResponse,
    VerificationResponse,
)
from swara.api.security import validate_audio_content_type, verify_api_key
from swara.core.audio import AudioValidationError, load_audio
from swara.core.config import settings
from swara.engine import Swara

router = APIRouter()


def get_engine(request: Request) -> Swara:
    """Dependency retrieving the shared Swara engine from app state."""
    engine: Optional[Swara] = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Swara speech engine is not initialized or still starting.",
        )
    return engine


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness probe returning service status and version."""
    return HealthResponse(version=__version__)


@router.get("/ready", response_model=ReadyResponse, tags=["System"])
async def ready_check(engine: Swara = Depends(get_engine)) -> ReadyResponse:
    """Readiness probe verifying that models and storage are loaded."""
    speakers_count = len(engine.list_speakers())
    return ReadyResponse(
        ready=True,
        models_loaded=True,
        device=settings.execution_provider,
        active_model=engine.encoder.model_version,
        enrolled_speakers_count=speakers_count,
    )


@router.get("/models", response_model=List[ModelItem], tags=["Models"])
async def list_models(engine: Swara = Depends(get_engine)) -> List[ModelItem]:
    """Lists registered models, download status, and licensing info."""
    models_data = engine.model_manager.list_models()
    return [ModelItem(**m) for m in models_data]


@router.get("/speakers", response_model=List[SpeakerResponse], tags=["Speakers"])
async def list_speakers(
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> List[SpeakerResponse]:
    """Retrieves all currently enrolled speaker profiles."""
    speakers = engine.list_speakers()
    return [SpeakerResponse(**s.to_dict()) for s in speakers]


@router.get("/speakers/{speaker_id}", response_model=SpeakerResponse, tags=["Speakers"])
async def get_speaker(
    speaker_id: str,
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> SpeakerResponse:
    """Retrieves profile of a specific enrolled speaker."""
    spk = engine.get_speaker(speaker_id)
    if not spk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker '{speaker_id}' not found.",
        )
    return SpeakerResponse(**spk.to_dict())


@router.delete("/speakers/{speaker_id}", tags=["Speakers"])
async def delete_speaker(
    speaker_id: str,
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Deletes an enrolled speaker identity and their embedding profile."""
    success = engine.delete_speaker(speaker_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker '{speaker_id}' not found.",
        )
    return {"status": "deleted", "speaker_id": speaker_id}


@router.post("/enroll", response_model=SpeakerResponse, tags=["Biometrics"])
async def enroll_speaker(
    speaker_id: str = Form(...),
    name: Optional[str] = Form(None),
    audio: UploadFile = File(...),
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> SpeakerResponse:
    """Enrolls a voice sample for a speaker identity."""
    validate_audio_content_type(audio.content_type)
    content = await audio.read()

    max_bytes = int(settings.max_audio_upload_size_mb * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size exceeds limit ({settings.max_audio_upload_size_mb} MB).",
        )

    try:
        audio_chunk = load_audio(content)
        profile = engine.enroll(speaker_id=speaker_id, audio=audio_chunk, name=name)
        return SpeakerResponse(**profile.to_dict())
    except AudioValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/identify", response_model=IdentificationResponse, tags=["Biometrics"])
async def identify_speaker(
    audio: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> IdentificationResponse:
    """Matches an input voice sample against all enrolled speakers."""
    validate_audio_content_type(audio.content_type)
    content = await audio.read()

    try:
        audio_chunk = load_audio(content)
        res = engine.identify(audio=audio_chunk, threshold=threshold)
        return IdentificationResponse(
            speaker_id=res.speaker_id,
            enrolled_id=res.speaker_id if res.is_match else None,
            similarity=res.similarity,
            confidence=res.confidence,
            is_match=res.is_match,
            audio_duration=res.audio_duration,
            candidates=[CandidateMatchResponse(**c.__dict__) for c in res.candidates],
        )
    except AudioValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/verify", response_model=VerificationResponse, tags=["Biometrics"])
async def verify_speaker(
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> VerificationResponse:
    """Performs 1:1 voice verification against a target enrolled speaker."""
    validate_audio_content_type(audio.content_type)
    content = await audio.read()

    try:
        audio_chunk = load_audio(content)
        is_ver, sim, conf = engine.verify(
            speaker_id=speaker_id, audio=audio_chunk, threshold=threshold
        )
        t = threshold or settings.identification_threshold
        return VerificationResponse(
            speaker_id=speaker_id,
            verified=is_ver,
            similarity=round(sim, 4),
            confidence=round(conf, 4),
            threshold=t,
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AudioValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/diarize", response_model=DiarizationResponse, tags=["Diarization"])
async def diarize_audio(
    audio: UploadFile = File(...),
    enhance_with_identity: bool = Form(True),
    threshold: Optional[float] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    engine: Swara = Depends(get_engine),
    _: None = Depends(verify_api_key),
) -> DiarizationResponse:
    """Performs speaker diarization on a multi-speaker audio recording."""
    validate_audio_content_type(audio.content_type)
    content = await audio.read()

    try:
        audio_chunk = load_audio(content)
        res = engine.diarize(
            audio=audio_chunk,
            enhance_with_identity=enhance_with_identity,
            threshold=threshold,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return DiarizationResponse(
            num_speakers=res.num_speakers,
            duration=res.duration,
            audio_path=audio.filename,
            segments=[DiarizationSegmentResponse(**s.to_dict()) for s in res.segments],
        )
    except AudioValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/metrics", tags=["System"])
async def get_metrics(engine: Swara = Depends(get_engine)) -> dict[str, Any]:
    """Exposes runtime and system metrics."""
    import psutil

    proc = psutil.Process()
    mem = proc.memory_info()

    return {
        "memory_rss_mb": round(mem.rss / (1024 * 1024), 2),
        "memory_vms_mb": round(mem.vms / (1024 * 1024), 2),
        "cpu_percent": proc.cpu_percent(),
        "threads_count": proc.num_threads(),
        "enrolled_speakers": len(engine.list_speakers()),
        "model_version": engine.encoder.model_version,
        "embedding_dim": engine.encoder.embedding_dim,
    }
