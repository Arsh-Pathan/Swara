"""Energy-based voice activity detector for lightweight fallback and testing."""

from typing import List
import numpy as np

from swara.core.types import AudioChunk, SpeechSegment
from swara.vad.base import BaseVAD


class EnergyVAD(BaseVAD):
    """Simple RMS energy-based VAD.

    Useful when running tests in environments without external neural network weights.
    """

    FRAME_SIZE = 512

    def __init__(
        self,
        energy_threshold: float = 0.015,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 200,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.reset()

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray) -> float:
        """Computes normalized RMS energy as a pseudo-speech probability."""
        if len(frame) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        prob = min(1.0, rms / max(1e-6, self.energy_threshold * 2.0))
        return prob

    def detect_segments(self, audio: AudioChunk) -> List[SpeechSegment]:
        """Detects speech using sliding window RMS energy."""
        samples = audio.samples
        sr = audio.sample_rate

        min_speech_samples = int(self.min_speech_duration_ms * sr / 1000)
        min_silence_samples = int(self.min_silence_duration_ms * sr / 1000)

        num_frames = len(samples) // self.FRAME_SIZE
        if num_frames == 0:
            return []

        triggered = False
        speech_start = 0
        temp_end = 0
        segments: List[SpeechSegment] = []

        for i in range(num_frames):
            frame = samples[i * self.FRAME_SIZE : (i + 1) * self.FRAME_SIZE]
            rms = float(np.sqrt(np.mean(frame**2)))
            current_sample = i * self.FRAME_SIZE

            if rms >= self.energy_threshold:
                if not triggered:
                    triggered = True
                    speech_start = current_sample
                temp_end = 0
            else:
                if triggered:
                    if temp_end == 0:
                        temp_end = current_sample
                    if (current_sample - temp_end) >= min_silence_samples:
                        speech_end = temp_end
                        if (speech_end - speech_start) >= min_speech_samples:
                            segments.append(
                                SpeechSegment(
                                    start=speech_start / sr,
                                    end=speech_end / sr,
                                    confidence=0.85,
                                )
                            )
                        triggered = False
                        temp_end = 0

        if triggered:
            speech_end = len(samples)
            if (speech_end - speech_start) >= min_speech_samples:
                segments.append(
                    SpeechSegment(
                        start=speech_start / sr,
                        end=speech_end / sr,
                        confidence=0.85,
                    )
                )

        return segments
