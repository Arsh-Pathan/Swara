"""Silero VAD v5 ONNX implementation for speech boundary detection."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import onnxruntime as ort

from swara.core.config import settings
from swara.core.types import AudioChunk, SpeechSegment
from swara.models.manager import ModelManager
from swara.vad.base import BaseVAD
from swara.utils.logger import get_logger

logger = get_logger("swara.vad.silero")


class SileroVAD(BaseVAD):
    """Voice Activity Detector powered by Silero VAD v5 ONNX.

    Silero VAD v5 requires 512-sample audio chunks at 16kHz with a 64-sample
    leading context window (total input 576 samples) and recurrent state passing.
    """

    FRAME_SIZE = 512  # 32 ms at 16000 Hz
    CONTEXT_SIZE = 64  # 4 ms leading context

    def __init__(
        self,
        model_path: Optional[Path] = None,
        onset_threshold: float = 0.50,
        offset_threshold: float = 0.35,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 30,
        execution_provider: Optional[str] = None,
    ) -> None:
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        # Resolve model path
        if model_path is None:
            mgr = ModelManager()
            if not mgr.is_downloaded("silero_vad"):
                mgr.download_model("silero_vad")
            model_path = mgr.get_model_path("silero_vad")

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found at {self.model_path}")

        # Initialize ONNX Runtime session
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [execution_provider or settings.execution_provider]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_opts, providers=providers
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.sample_rate_tensor = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Resets the recurrent hidden state and leading context window."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SIZE), dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> float:
        """Processes a 512-sample frame and returns speech probability in [0, 1]."""
        if len(frame) != self.FRAME_SIZE:
            # Pad or truncate to 512 samples
            if len(frame) < self.FRAME_SIZE:
                frame = np.pad(frame, (0, self.FRAME_SIZE - len(frame)), mode="constant")
            else:
                frame = frame[: self.FRAME_SIZE]

        frame_2d = frame.astype(np.float32).reshape(1, self.FRAME_SIZE)

        # Prepend 64-sample context: shape (1, 576)
        x = np.concatenate([self._context, frame_2d], axis=1)
        self._context = frame_2d[:, -self.CONTEXT_SIZE :].copy()

        # Run inference: inputs are ['input', 'state', 'sr']
        inputs = {
            self.input_names[0]: x,
            self.input_names[1]: self._state,
            self.input_names[2]: self.sample_rate_tensor,
        }
        outputs = self.session.run(None, inputs)

        prob = float(outputs[0][0][0])
        self._state = outputs[1]
        return prob

    def detect_segments(self, audio: AudioChunk) -> List[SpeechSegment]:
        """Detects speech regions across complete audio recording with hysteresis."""
        self.reset()
        samples = audio.samples
        sr = audio.sample_rate

        min_speech_samples = int(self.min_speech_duration_ms * sr / 1000)
        min_silence_samples = int(self.min_silence_duration_ms * sr / 1000)
        pad_samples = int(self.speech_pad_ms * sr / 1000)

        triggered = False
        speech_start = 0
        speech_end = 0
        temp_end = 0
        current_probs: List[float] = []

        segments: List[SpeechSegment] = []

        num_frames = len(samples) // self.FRAME_SIZE
        if num_frames == 0:
            return []

        for i in range(num_frames):
            frame = samples[i * self.FRAME_SIZE : (i + 1) * self.FRAME_SIZE]
            prob = self.process_frame(frame)
            current_probs.append(prob)
            current_sample = i * self.FRAME_SIZE

            if prob >= self.onset_threshold:
                if not triggered:
                    triggered = True
                    speech_start = max(0, current_sample - pad_samples)
                temp_end = 0

            elif prob < self.offset_threshold:
                if triggered:
                    if temp_end == 0:
                        temp_end = current_sample
                    if (current_sample - temp_end) >= min_silence_samples:
                        speech_end = min(len(samples), temp_end + pad_samples)
                        duration_samples = speech_end - speech_start
                        if duration_samples >= min_speech_samples:
                            # Mean confidence
                            start_idx = speech_start // self.FRAME_SIZE
                            end_idx = speech_end // self.FRAME_SIZE + 1
                            segment_probs = current_probs[start_idx:end_idx]
                            conf = float(np.mean(segment_probs)) if segment_probs else prob
                            segments.append(
                                SpeechSegment(
                                    start=speech_start / sr,
                                    end=speech_end / sr,
                                    confidence=conf,
                                )
                            )
                        triggered = False
                        temp_end = 0

        # Handle utterance ending while speech was ongoing
        if triggered:
            speech_end = len(samples)
            duration_samples = speech_end - speech_start
            if duration_samples >= min_speech_samples:
                start_idx = speech_start // self.FRAME_SIZE
                segment_probs = current_probs[start_idx:]
                conf = float(np.mean(segment_probs)) if segment_probs else 0.8
                segments.append(
                    SpeechSegment(
                        start=speech_start / sr,
                        end=speech_end / sr,
                        confidence=conf,
                    )
                )

        self.reset()
        return segments
