"""Offline and streaming speaker diarization engine."""

from typing import List, Optional, Union
import numpy as np

from swara.core.audio import load_audio
from swara.core.config import settings
from swara.core.types import AudioChunk, DiarizationResult, DiarizationSegment, SpeechSegment
from swara.diarization.clustering import AgglomerativeClustering, OnlineCentroidClustering
from swara.embedding.base import BaseSpeakerEncoder
from swara.vad.base import BaseVAD
from swara.utils.logger import get_logger

logger = get_logger("swara.diarization")


class SpeakerDiarizationEngine:
    """Production speaker diarization pipeline combining VAD, embedding extraction, and clustering."""

    def __init__(
        self,
        vad: BaseVAD,
        encoder: BaseSpeakerEncoder,
        clustering_threshold: Optional[float] = None,
        window_size_s: Optional[float] = None,
        step_size_s: Optional[float] = None,
        collar_s: Optional[float] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> None:
        self.vad = vad
        self.encoder = encoder
        self.clustering_threshold = (
            clustering_threshold or settings.diarization_clustering_threshold
        )
        self.window_size_s = window_size_s or settings.diarization_window_size_s
        self.step_size_s = step_size_s or settings.diarization_step_size_s
        self.collar_s = collar_s or settings.diarization_collar_s
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def diarize(
        self,
        audio: Union[str, AudioChunk, np.ndarray],
        threshold: Optional[float] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """Performs speaker diarization on complete audio recording.

        Args:
            audio: AudioChunk, file path, or numpy array.
            threshold: Distance threshold for clustering.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.

        Returns:
            DiarizationResult with time-aligned speaker segments.
        """
        chunk = audio if isinstance(audio, AudioChunk) else load_audio(audio)
        total_duration = chunk.duration
        audio_path_str = str(audio) if isinstance(audio, str) else None

        if total_duration < 0.2:
            return DiarizationResult(segments=[], num_speakers=0, duration=total_duration, audio_path=audio_path_str)

        # 1. Voice Activity Detection
        speech_segments = self.vad.detect_segments(chunk)
        if not speech_segments:
            return DiarizationResult(segments=[], num_speakers=0, duration=total_duration, audio_path=audio_path_str)

        # 2. Windowed sub-segmentation of speech regions
        sr = chunk.sample_rate
        sub_segments: List[tuple[float, float, np.ndarray]] = []
        win_samples = int(self.window_size_s * sr)
        step_samples = int(self.step_size_s * sr)

        for sp_seg in speech_segments:
            start_s = sp_seg.start
            end_s = sp_seg.end
            seg_len_s = end_s - start_s

            start_idx = int(start_s * sr)
            end_idx = min(len(chunk.samples), int(end_s * sr))
            seg_samples = chunk.samples[start_idx:end_idx]

            if len(seg_samples) <= win_samples:
                sub_segments.append((start_s, end_s, seg_samples))
            else:
                # Sliding window
                cur_start = 0
                while cur_start + win_samples <= len(seg_samples):
                    w_samples = seg_samples[cur_start : cur_start + win_samples]
                    w_start_s = start_s + (cur_start / sr)
                    w_end_s = w_start_s + self.window_size_s
                    sub_segments.append((w_start_s, w_end_s, w_samples))
                    cur_start += step_samples

                # Trailing chunk if any
                if cur_start < len(seg_samples) and (len(seg_samples) - cur_start) > (0.4 * sr):
                    w_samples = seg_samples[cur_start:]
                    w_start_s = start_s + (cur_start / sr)
                    w_end_s = end_s
                    sub_segments.append((w_start_s, w_end_s, w_samples))

        if not sub_segments:
            return DiarizationResult(segments=[], num_speakers=0, duration=total_duration, audio_path=audio_path_str)

        # 3. Extract speaker embeddings
        embeddings_list = []
        for _, _, w_samples in sub_segments:
            emb = self.encoder.extract_embedding(w_samples)
            embeddings_list.append(emb)

        embeddings_matrix = np.vstack(embeddings_list)

        # 4. Cluster embeddings into discrete speaker labels
        t = threshold or self.clustering_threshold
        clusterer = AgglomerativeClustering(
            threshold=t,
            min_speakers=min_speakers or self.min_speakers,
            max_speakers=max_speakers or self.max_speakers,
        )
        labels = clusterer.cluster(embeddings_matrix)

        # 5. Build raw segments
        raw_segments: List[DiarizationSegment] = []
        for i, (start_s, end_s, _) in enumerate(sub_segments):
            spk_label = f"speaker_{labels[i] + 1}"
            raw_segments.append(
                DiarizationSegment(start=start_s, end=end_s, speaker_id=spk_label)
            )

        # 6. Smooth and merge adjacent/overlapping segments of the same speaker
        merged_segments = self._merge_segments(raw_segments, self.collar_s)
        unique_spks = len(set(s.speaker_id for s in merged_segments))

        return DiarizationResult(
            segments=merged_segments,
            num_speakers=unique_spks,
            duration=total_duration,
            audio_path=audio_path_str,
        )

    def _merge_segments(
        self, segments: List[DiarizationSegment], collar_s: float
    ) -> List[DiarizationSegment]:
        """Merges adjacent segments belonging to the same speaker separated by small collar."""
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start)
        merged: List[DiarizationSegment] = []
        current = segments[0]

        for nxt in segments[1:]:
            if current.speaker_id == nxt.speaker_id:
                # Same speaker: if overlapping or gap is less than collar, merge
                if nxt.start <= (current.end + collar_s):
                    current.end = max(current.end, nxt.end)
                    continue

            merged.append(current)
            current = nxt

        merged.append(current)
        return merged
