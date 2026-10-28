"""Abstract base interface for Voice Activity Detection."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from swara.core.types import AudioChunk, SpeechSegment


class BaseVAD(ABC):
    """Abstract interface for voice activity detectors."""

    @abstractmethod
    def detect_segments(self, audio: AudioChunk) -> List[SpeechSegment]:
        """Detects speech intervals in complete audio recording.

        Args:
            audio: AudioChunk sampled at 16kHz.

        Returns:
            List of SpeechSegment objects containing start, end timestamps and confidence.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> float:
        """Processes a single audio frame (e.g., 512 samples) in streaming mode.

        Args:
            frame: 1D numpy array of audio samples.

        Returns:
            Speech probability between 0.0 and 1.0.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets internal streaming state / recurrent memory."""
        pass
