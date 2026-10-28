"""WeSpeaker ResNet-34 ONNX deep speaker encoder."""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import onnxruntime as ort

from swara.core.config import settings
from swara.core.dsp import KaldiFbank
from swara.core.types import AudioChunk
from swara.embedding.base import BaseSpeakerEncoder
from swara.models.manager import ModelManager
from swara.utils.logger import get_logger

logger = get_logger("swara.embedding.wespeaker")


class WeSpeakerEncoder(BaseSpeakerEncoder):
    """Production speaker embedding extractor based on WeSpeaker ResNet-34 LM."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_int8: Optional[bool] = None,
        execution_provider: Optional[str] = None,
        intra_op_threads: Optional[int] = None,
    ) -> None:
        self.use_int8 = settings.use_int8_model if use_int8 is None else use_int8
        model_id = "wespeaker_resnet34_int8" if self.use_int8 else "wespeaker_resnet34"

        # Resolve model path
        if model_path is None:
            mgr = ModelManager()
            if not mgr.is_downloaded(model_id):
                mgr.download_model(model_id)
            model_path = mgr.get_model_path(model_id)

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"WeSpeaker model not found at {self.model_path}")

        # Initialize ONNX Runtime session
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = settings.inter_op_num_threads
        sess_opts.intra_op_num_threads = (
            intra_op_threads or settings.intra_op_num_threads
        )
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [execution_provider or settings.execution_provider]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Acoustic front-end feature extractor
        self.fbank = KaldiFbank(
            sample_rate=settings.target_sample_rate,
            num_mel_bins=80,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            preemphasis=0.97,
            n_fft=512,
            cmn=True,
        )

        self._embedding_dim = 256
        self._model_version = f"wespeaker-resnet34-lm{'-int8' if self.use_int8 else ''}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_version(self) -> str:
        return self._model_version

    def extract_embedding(self, audio: Union[AudioChunk, np.ndarray]) -> np.ndarray:
        """Extracts a 256-dimensional L2-normalized speaker embedding."""
        samples = audio.samples if isinstance(audio, AudioChunk) else audio
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)

        # Handle very short utterance by padding to at least 400 samples (25 ms)
        if len(samples) < 400:
            samples = np.pad(samples, (0, 400 - len(samples)), mode="constant")

        # Extract 80-bin Mel filterbank features: shape (T, 80)
        feats = self.fbank.compute(samples)
        if len(feats) < 9:
            # Model requires minimum 9 frames
            repeats = (9 // len(feats)) + 1
            feats = np.tile(feats, (repeats, 1))[:10]

        # Add batch dimension: shape (1, T, 80)
        feats_batch = np.expand_dims(feats, axis=0).astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: feats_batch})
        embedding = outputs[0][0]  # Shape (256,)

        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)

        return embedding.astype(np.float32)

    def extract_batch(self, audios: List[Union[AudioChunk, np.ndarray]]) -> np.ndarray:
        """Extracts embeddings for a list of audio segments."""
        if not audios:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Because frame counts T can vary per utterance, we process individual utterances
        # or pad. Processing sequentially with warm session is ~10ms each.
        embeddings = [self.extract_embedding(a) for a in audios]
        return np.vstack(embeddings).astype(np.float32)

    @staticmethod
    def calibrate_confidence(similarity: float, threshold: float = 0.68, k: float = 12.0) -> float:
        """Converts raw cosine similarity into a well-calibrated confidence score in [0.0, 1.0].

        Using logistic mapping around decision threshold:
        Confidence = 1 / (1 + exp(-k * (similarity - threshold)))
        """
        x = k * (similarity - threshold)
        # Numerical stability against overflow
        if x >= 40.0:
            return 1.0
        if x <= -40.0:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-x)))
