"""Speaker identification and enrollment engine."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from swara.core.audio import AudioValidationError, load_audio
from swara.core.config import settings
from swara.core.types import AudioChunk, CandidateMatch, EnrolledSpeaker, MatchResult
from swara.embedding.base import BaseSpeakerEncoder
from swara.identification.store import SpeakerStore
from swara.utils.logger import get_logger

logger = get_logger("swara.identification")


class SpeakerIdentificationEngine:
    """Production speaker enrollment, verification, and identification engine."""

    def __init__(
        self,
        encoder: BaseSpeakerEncoder,
        store: Optional[SpeakerStore] = None,
        default_threshold: Optional[float] = None,
    ) -> None:
        self.encoder = encoder
        self.store = store or SpeakerStore()
        self.default_threshold = (
            default_threshold or settings.identification_threshold
        )

    def enroll(
        self,
        speaker_id: str,
        audio: Union[str, AudioChunk, np.ndarray],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrolledSpeaker:
        """Enrolls a voice sample for a given speaker identity.

        Args:
            speaker_id: Unique string identifier (e.g. 'arsh', 'user_123').
            audio: AudioChunk, file path, or numpy array.
            name: Optional human-readable display name.
            metadata: Optional dictionary of arbitrary metadata.

        Returns:
            EnrolledSpeaker profile.

        Raises:
            AudioValidationError: If audio is too short or silence.
        """
        if not speaker_id or not speaker_id.strip():
            raise ValueError("speaker_id cannot be empty.")

        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)

        if chunk.duration < settings.min_enrollment_duration_s:
            raise AudioValidationError(
                f"Audio duration ({chunk.duration:.2f}s) is too short for enrollment (minimum {settings.min_enrollment_duration_s:.2f}s required)."
            )

        # Check energy to avoid enrolling pure silence
        rms = float(np.sqrt(np.mean(chunk.samples**2)))
        if rms < 0.005:
            raise AudioValidationError("Audio sample appears to be complete silence or unvoiced noise.")

        # Extract deep embedding vector
        embedding = self.encoder.extract_embedding(chunk)

        # Build metadata profile
        profile = EnrolledSpeaker(
            speaker_id=speaker_id.strip(),
            name=name.strip() if name else speaker_id.strip(),
            embedding_dim=self.encoder.embedding_dim,
            model_version=self.encoder.model_version,
            total_audio_duration_s=chunk.duration,
            metadata=metadata or {},
        )

        self.store.save_speaker(profile, embedding, duration_s=chunk.duration)
        return self.store.get_speaker(speaker_id.strip()) or profile

    def identify(
        self,
        audio: Union[str, AudioChunk, np.ndarray],
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> MatchResult:
        """Matches input audio against all enrolled speakers.

        Args:
            audio: AudioChunk, file path, or numpy array.
            threshold: Cosine similarity cutoff (defaults to config threshold).
            top_k: Number of candidate matches to return.

        Returns:
            MatchResult containing best match or UNKNOWN, similarity score, and candidate rankings.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)

        if chunk.duration < settings.min_identification_duration_s:
            raise AudioValidationError(
                f"Audio duration ({chunk.duration:.2f}s) is too short for identification (minimum {settings.min_identification_duration_s:.2f}s required)."
            )

        # Check for enrolled speakers
        speaker_ids, enrolled_embeddings = self.store.get_all_embeddings()
        if len(speaker_ids) == 0:
            return MatchResult(
                speaker_id="UNKNOWN",
                similarity=None,
                confidence=None,
                is_match=False,
                candidates=[],
                audio_duration=chunk.duration,
            )

        # Extract probe embedding
        probe_emb = self.encoder.extract_embedding(chunk)

        # Compute cosine similarities in parallel: matrix product since vectors are unit-norm
        # shape: (N,)
        similarities = np.dot(enrolled_embeddings, probe_emb)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Rank descending
        ranked_indices = np.argsort(-similarities)

        candidates: List[CandidateMatch] = []
        for idx in ranked_indices[:top_k]:
            sid = speaker_ids[idx]
            sim = float(similarities[idx])
            from swara.embedding.wespeaker import WeSpeakerEncoder

            conf = WeSpeakerEncoder.calibrate_confidence(sim, threshold=thresh)
            is_cand_match = sim >= thresh
            candidates.append(
                CandidateMatch(
                    speaker_id=sid,
                    similarity=sim,
                    confidence=conf,
                    is_match=is_cand_match,
                )
            )

        best_sim = float(similarities[ranked_indices[0]])
        best_sid = speaker_ids[ranked_indices[0]]
        from swara.embedding.wespeaker import WeSpeakerEncoder

        best_conf = WeSpeakerEncoder.calibrate_confidence(best_sim, threshold=thresh)
        is_match = best_sim >= thresh

        return MatchResult(
            speaker_id=best_sid if is_match else "UNKNOWN",
            similarity=best_sim,
            confidence=best_conf,
            is_match=is_match,
            candidates=candidates,
            audio_duration=chunk.duration,
        )

    def verify(
        self,
        speaker_id: str,
        audio: Union[str, AudioChunk, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """Performs 1:1 voice verification against an enrolled speaker.

        Returns:
            Tuple of (is_verified, cosine_similarity, calibrated_confidence).
        """
        thresh = threshold if threshold is not None else self.default_threshold
        enrolled_emb = self.store.get_embedding(speaker_id)
        if enrolled_emb is None:
            raise KeyError(f"Enrolled speaker '{speaker_id}' not found.")

        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)
        probe_emb = self.encoder.extract_embedding(chunk)

        sim = BaseSpeakerEncoder.compute_similarity(probe_emb, enrolled_emb)
        from swara.embedding.wespeaker import WeSpeakerEncoder

        conf = WeSpeakerEncoder.calibrate_confidence(sim, threshold=thresh)
        is_match = sim >= thresh
        return is_match, sim, conf
