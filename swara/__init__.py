"""Swara: Ultra-low-latency, lightweight speaker intelligence AI system.

CSKY - "Every Voice Has an Identity."
"""

from swara.engine import Swara
from swara.core.types import (
    AudioChunk,
    SpeechSegment,
    EnrolledSpeaker,
    MatchResult,
    DiarizationSegment,
    DiarizationResult,
)

__version__ = "1.0.0"
__all__ = [
    "Swara",
    "AudioChunk",
    "SpeechSegment",
    "EnrolledSpeaker",
    "MatchResult",
    "DiarizationSegment",
    "DiarizationResult",
    "__version__",
]
