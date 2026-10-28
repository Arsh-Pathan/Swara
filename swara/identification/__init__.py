"""Speaker identification and enrollment package."""

from swara.identification.store import SpeakerStore
from swara.identification.engine import SpeakerIdentificationEngine

__all__ = ["SpeakerStore", "SpeakerIdentificationEngine"]
