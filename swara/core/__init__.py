"""Core modules for audio processing, types, and configuration."""

from swara.core.types import (
    AudioChunk,
    SpeechSegment,
    EnrolledSpeaker,
    CandidateMatch,
    MatchResult,
    DiarizationSegment,
    DiarizationResult,
)
from swara.core.config import Settings, settings
from swara.core.dsp import KaldiFbank, default_fbank
from swara.core.audio import load_audio, save_audio, resample_audio, record_microphone, AudioError, AudioValidationError

__all__ = [
    "AudioChunk",
    "SpeechSegment",
    "EnrolledSpeaker",
    "CandidateMatch",
    "MatchResult",
    "DiarizationSegment",
    "DiarizationResult",
    "Settings",
    "settings",
    "KaldiFbank",
    "default_fbank",
    "load_audio",
    "save_audio",
    "resample_audio",
    "record_microphone",
    "AudioError",
    "AudioValidationError",
]
