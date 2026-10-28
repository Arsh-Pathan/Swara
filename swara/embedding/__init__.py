"""Speaker embedding extraction package."""

from swara.embedding.base import BaseSpeakerEncoder
from swara.embedding.wespeaker import WeSpeakerEncoder

__all__ = ["BaseSpeakerEncoder", "WeSpeakerEncoder"]
