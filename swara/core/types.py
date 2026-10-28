"""Data contracts and domain models for Swara."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class AudioChunk:
    """Represents a segment of audio waveform in memory."""

    samples: np.ndarray  # 1D float32 array in range [-1.0, 1.0]
    sample_rate: int = 16000

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / float(self.sample_rate)

    @property
    def num_samples(self) -> int:
        return len(self.samples)


@dataclass
class SpeechSegment:
    """Represents a temporal speech interval detected by VAD."""

    start: float  # seconds
    end: float  # seconds
    confidence: float = 1.0  # mean speech probability

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class EnrolledSpeaker:
    """Metadata describing an enrolled voice identity."""

    speaker_id: str
    name: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sample_count: int = 1
    total_audio_duration_s: float = 0.0
    embedding_dim: int = 256
    model_version: str = "wespeaker-resnet34-lm"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "name": self.name or self.speaker_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_count": self.sample_count,
            "total_audio_duration_s": round(self.total_audio_duration_s, 2),
            "embedding_dim": self.embedding_dim,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


@dataclass
class CandidateMatch:
    """Candidate match with similarity and distance metrics."""

    speaker_id: str
    similarity: float  # Cosine similarity in [-1.0, 1.0]
    confidence: float  # Calibrated probability metric in [0.0, 1.0]
    is_match: bool


@dataclass
class MatchResult:
    """Result of speaker identification."""

    speaker_id: str  # Enrolled speaker_id or 'UNKNOWN'
    similarity: Optional[float]  # Cosine similarity to best match
    confidence: Optional[float]  # Calibrated confidence
    is_match: bool  # True if similarity >= threshold
    candidates: List[CandidateMatch] = field(default_factory=list)
    audio_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "enrolled_id": self.speaker_id if self.is_match else None,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "is_match": self.is_match,
            "audio_duration": round(self.audio_duration, 2),
            "candidates": [
                {
                    "speaker_id": c.speaker_id,
                    "similarity": round(c.similarity, 4),
                    "confidence": round(c.confidence, 4),
                    "is_match": c.is_match,
                }
                for c in self.candidates
            ],
        }


@dataclass
class DiarizationSegment:
    """A time-aligned diarization interval."""

    start: float  # seconds
    end: float  # seconds
    speaker_id: str  # e.g., 'speaker_1', 'speaker_2'
    identity: Optional[str] = None  # Enrolled identity if matched
    similarity: Optional[float] = None  # Similarity to enrolled identity if matched

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "speaker_id": self.speaker_id,
            "identity": self.identity,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
        }
        return d


@dataclass
class DiarizationResult:
    """Complete speaker diarization result."""

    segments: List[DiarizationSegment]
    num_speakers: int
    duration: float
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_speakers": self.num_speakers,
            "duration": round(self.duration, 3),
            "audio_path": self.audio_path,
            "segments": [s.to_dict() for s in self.segments],
        }
