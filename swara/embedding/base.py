"""Abstract base interface for Speaker Embedding encoders."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

from swara.core.types import AudioChunk


class BaseSpeakerEncoder(ABC):
    """Abstract interface for extracting deep speaker representations."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version or name string."""
        pass

    @abstractmethod
    def extract_embedding(self, audio: Union[AudioChunk, np.ndarray]) -> np.ndarray:
        """Extracts a single L2-normalized 1D embedding vector.

        Args:
            audio: AudioChunk or 1D numpy array of audio samples at 16kHz.

        Returns:
            np.ndarray of shape (embedding_dim,) normalized to unit norm.
        """
        pass

    @abstractmethod
    def extract_batch(self, audios: List[Union[AudioChunk, np.ndarray]]) -> np.ndarray:
        """Extracts batched embeddings.

        Args:
            audios: List of AudioChunks or numpy arrays.

        Returns:
            np.ndarray of shape (N, embedding_dim) with L2-normalized rows.
        """
        pass

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two unit-normalized vectors.

        Args:
            emb1: 1D numpy array of shape (D,).
            emb2: 1D numpy array of shape (D,).

        Returns:
            Cosine similarity scalar in range [-1.0, 1.0].
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0
        similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
        return float(np.clip(similarity, -1.0, 1.0))
