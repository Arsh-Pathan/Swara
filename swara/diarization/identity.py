"""Identity-Enhanced Diarization.

Links unsupervised diarized speaker clusters to registered voice profiles in the
SpeakerStore, outputting time-aligned segments with verified personal identity labels.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from swara.core.audio import load_audio
from swara.core.config import settings
from swara.core.types import AudioChunk, DiarizationResult, DiarizationSegment
from swara.embedding.base import BaseSpeakerEncoder
from swara.identification.store import SpeakerStore
from swara.utils.logger import get_logger

logger = get_logger("swara.diarization.identity")


class IdentityEnhancedDiarization:
    """Enhances diarization segments with enrolled speaker identities."""

    def __init__(
        self,
        encoder: BaseSpeakerEncoder,
        store: SpeakerStore,
        match_threshold: Optional[float] = None,
    ) -> None:
        self.encoder = encoder
        self.store = store
        self.match_threshold = (
            match_threshold or settings.diarization_identity_match_threshold
        )

    def enhance(
        self,
        diarization: DiarizationResult,
        audio: Union[str, AudioChunk, np.ndarray],
        threshold: Optional[float] = None,
    ) -> DiarizationResult:
        """Matches each diarized speaker cluster against enrolled identities.

        Args:
            diarization: DiarizationResult from SpeakerDiarizationEngine.
            audio: The source audio waveform or path.
            threshold: Cosine similarity cutoff for identity matching.

        Returns:
            A new DiarizationResult with identity and similarity populated where matched.
        """
        thresh = threshold or self.match_threshold
        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)
        sr = chunk.sample_rate

        enrolled_ids, enrolled_embs = self.store.get_all_embeddings()
        if len(enrolled_ids) == 0 or len(diarization.segments) == 0:
            return diarization

        # 1. Group segments by speaker_id and extract composite embeddings
        speaker_samples: Dict[str, List[np.ndarray]] = {}
        for seg in diarization.segments:
            st = int(seg.start * sr)
            en = min(len(chunk.samples), int(seg.end * sr))
            if (en - st) > 160:  # At least 10ms
                speaker_samples.setdefault(seg.speaker_id, []).append(chunk.samples[st:en])

        # 2. Compute mean centroid embedding for each diarized speaker
        cluster_centroids: Dict[str, np.ndarray] = {}
        for spk_id, samples_list in speaker_samples.items():
            embs = []
            for s in samples_list:
                if len(s) >= 400:
                    embs.append(self.encoder.extract_embedding(s))
            if embs:
                mean_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(mean_emb)
                cluster_centroids[spk_id] = mean_emb / norm if norm > 1e-12 else mean_emb

        # 3. Match centroids against enrolled identities
        # Use greedy bipartite matching: best score gets assigned first to prevent duplicates
        match_candidates: List[Tuple[float, str, str]] = []  # (sim, spk_id, enrolled_id)
        for spk_id, centroid in cluster_centroids.items():
            sims = np.dot(enrolled_embs, centroid)
            for i, sim in enumerate(sims):
                if sim >= thresh:
                    match_candidates.append((float(sim), spk_id, enrolled_ids[i]))

        # Sort descending by similarity
        match_candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_spks = set()
        assigned_enrolled = set()
        identity_map: Dict[str, Tuple[str, float]] = {}

        for sim, spk_id, enrolled_id in match_candidates:
            if spk_id not in assigned_spks and enrolled_id not in assigned_enrolled:
                identity_map[spk_id] = (enrolled_id, sim)
                assigned_spks.add(spk_id)
                assigned_enrolled.add(enrolled_id)

        # 4. Create updated segments with identity metadata
        enhanced_segments: List[DiarizationSegment] = []
        for seg in diarization.segments:
            ident, sim = identity_map.get(seg.speaker_id, (None, None))
            enhanced_segments.append(
                DiarizationSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker_id=seg.speaker_id,
                    identity=ident,
                    similarity=sim,
                )
            )

        return DiarizationResult(
            segments=enhanced_segments,
            num_speakers=diarization.num_speakers,
            duration=diarization.duration,
            audio_path=diarization.audio_path,
        )
