"""Unified top-level Swara engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from swara.core.audio import load_audio
from swara.core.config import Settings, settings
from swara.core.types import (
    AudioChunk,
    DiarizationResult,
    EnrolledSpeaker,
    MatchResult,
    SpeechSegment,
)
from swara.diarization.engine import SpeakerDiarizationEngine
from swara.diarization.identity import IdentityEnhancedDiarization
from swara.embedding.base import BaseSpeakerEncoder
from swara.embedding.wespeaker import WeSpeakerEncoder
from swara.identification.engine import SpeakerIdentificationEngine
from swara.identification.store import SpeakerStore
from swara.models.manager import ModelManager
from swara.vad.base import BaseVAD
from swara.vad.silero import SileroVAD
from swara.utils.logger import get_logger

logger = get_logger("swara")


class Swara:
    """Production speaker intelligence engine.

    Provides Voice Activity Detection (VAD), speaker embedding generation,
    speaker enrollment, identification, unknown rejection, and diarization.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        vad: Optional[BaseVAD] = None,
        encoder: Optional[BaseSpeakerEncoder] = None,
        store: Optional[SpeakerStore] = None,
        auto_download_models: bool = True,
    ) -> None:
        self.config = config or settings
        self.model_manager = ModelManager(cache_dir=self.config.models_dir)

        if auto_download_models:
            # Ensure default models are present
            if not self.model_manager.is_downloaded("silero_vad"):
                self.model_manager.download_model("silero_vad")
            target_model = (
                "wespeaker_resnet34_int8"
                if self.config.use_int8_model
                else "wespeaker_resnet34"
            )
            if not self.model_manager.is_downloaded(target_model):
                self.model_manager.download_model(target_model)

        # Initialize core components
        self.vad = vad or SileroVAD(
            model_path=self.model_manager.get_model_path("silero_vad"),
            onset_threshold=self.config.vad_onset_threshold,
            offset_threshold=self.config.vad_offset_threshold,
            min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
            min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
        )

        model_id = (
            "wespeaker_resnet34_int8"
            if self.config.use_int8_model
            else "wespeaker_resnet34"
        )
        self.encoder = encoder or WeSpeakerEncoder(
            model_path=self.model_manager.get_model_path(model_id),
            use_int8=self.config.use_int8_model,
            execution_provider=self.config.execution_provider,
        )

        self.store = store or SpeakerStore(
            storage_dir=self.config.speaker_store_path
        )

        # Identification Engine
        self.identification = SpeakerIdentificationEngine(
            encoder=self.encoder,
            store=self.store,
            default_threshold=self.config.identification_threshold,
        )

        # Diarization Engine
        self.diarization = SpeakerDiarizationEngine(
            vad=self.vad,
            encoder=self.encoder,
            clustering_threshold=self.config.diarization_clustering_threshold,
            window_size_s=self.config.diarization_window_size_s,
            step_size_s=self.config.diarization_step_size_s,
            collar_s=self.config.diarization_collar_s,
        )

        # Identity-Enhanced Diarization
        self.identity_diarization = IdentityEnhancedDiarization(
            encoder=self.encoder,
            store=self.store,
            match_threshold=self.config.diarization_identity_match_threshold,
        )

        logger.info(
            f"Initialized Swara engine (Model: {self.encoder.model_version}, Dim: {self.encoder.embedding_dim})"
        )

    def detect_speech(
        self, audio: Union[str, Path, AudioChunk, np.ndarray]
    ) -> List[SpeechSegment]:
        """Detects speech regions in audio.

        Args:
            audio: File path, AudioChunk, or numpy array.

        Returns:
            List of SpeechSegment objects.
        """
        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)
        return self.vad.detect_segments(chunk)

    def extract_embedding(
        self, audio: Union[str, Path, AudioChunk, np.ndarray]
    ) -> np.ndarray:
        """Extracts a 256-dimensional speaker embedding vector."""
        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)
        return self.encoder.extract_embedding(chunk)

    def enroll(
        self,
        speaker_id: str,
        audio_path: Optional[Union[str, Path]] = None,
        audio: Optional[Union[AudioChunk, np.ndarray]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrolledSpeaker:
        """Enrolls a speaker voice sample into the persistent store.

        Args:
            speaker_id: Unique string identifier (e.g., 'arsh').
            audio_path: Path to audio file (WAV, FLAC, etc.).
            audio: Direct AudioChunk or numpy array.
            name: Display name for speaker.
            metadata: Arbitrary metadata dictionary.

        Returns:
            EnrolledSpeaker metadata object.
        """
        src = audio if audio is not None else audio_path
        if src is None:
            raise ValueError("Either audio_path or audio must be provided.")
        return self.identification.enroll(
            speaker_id=speaker_id, audio=src, name=name, metadata=metadata
        )

    def identify(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        audio: Optional[Union[AudioChunk, np.ndarray]] = None,
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Identifies speaker from voice sample against all enrolled profiles.

        Args:
            audio_path: Path to probe audio file.
            audio: Direct AudioChunk or numpy array.
            threshold: Cosine similarity cutoff.

        Returns:
            MatchResult containing speaker_id, similarity, confidence, and candidate matches.
        """
        src = audio if audio is not None else audio_path
        if src is None:
            raise ValueError("Either audio_path or audio must be provided.")
        return self.identification.identify(audio=src, threshold=threshold)

    def verify(
        self,
        speaker_id: str,
        audio_path: Optional[Union[str, Path]] = None,
        audio: Optional[Union[AudioChunk, np.ndarray]] = None,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float, float]:
        """Performs 1:1 voice verification against an enrolled speaker.

        Returns:
            Tuple of (is_verified, similarity, confidence).
        """
        src = audio if audio is not None else audio_path
        if src is None:
            raise ValueError("Either audio_path or audio must be provided.")
        return self.identification.verify(speaker_id=speaker_id, audio=src, threshold=threshold)

    def diarize(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        audio: Optional[Union[AudioChunk, np.ndarray]] = None,
        enhance_with_identity: bool = True,
        threshold: Optional[float] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """Performs speaker diarization (who spoke when).

        If enhance_with_identity is True and enrolled speakers exist, matches clusters
        to enrolled identities.

        Args:
            audio_path: Path to conversation audio file.
            audio: Direct AudioChunk or numpy array.
            enhance_with_identity: Whether to label clusters with enrolled names.
            threshold: Clustering distance threshold.
            min_speakers: Minimum speaker count hint.
            max_speakers: Maximum speaker count hint.

        Returns:
            DiarizationResult with time-aligned speaker segments.
        """
        src = audio if audio is not None else audio_path
        if src is None:
            raise ValueError("Either audio_path or audio must be provided.")

        result = self.diarization.diarize(
            audio=src,
            threshold=threshold,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        if enhance_with_identity:
            result = self.identity_diarization.enhance(diarization=result, audio=src)

        return result

    def list_speakers(self) -> List[EnrolledSpeaker]:
        """Lists all enrolled speakers."""
        return self.store.list_speakers()

    def get_speaker(self, speaker_id: str) -> Optional[EnrolledSpeaker]:
        """Retrieves profile of a specific enrolled speaker."""
        return self.store.get_speaker(speaker_id)

    def delete_speaker(self, speaker_id: str) -> bool:
        """Deletes an enrolled speaker profile and embeddings."""
        return self.store.delete_speaker(speaker_id)
