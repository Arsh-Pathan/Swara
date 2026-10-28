"""Unit and integration tests for speaker diarization and clustering."""

import numpy as np
import pytest

from swara.core.audio import load_audio
from swara.core.types import AudioChunk
from swara.diarization.clustering import AgglomerativeClustering, OnlineCentroidClustering
from swara.diarization.engine import SpeakerDiarizationEngine
from swara.diarization.identity import IdentityEnhancedDiarization
from swara.embedding.wespeaker import WeSpeakerEncoder
from swara.identification.store import SpeakerStore
from swara.vad.silero import SileroVAD


def test_ahc_clustering():
    # 3 clusters of 256-dim embeddings
    np.random.seed(42)
    c1 = np.random.randn(256)
    c1 /= np.linalg.norm(c1)

    c2 = np.random.randn(256)
    c2 /= np.linalg.norm(c2)

    # 4 embeddings around c1, 4 around c2
    embs1 = [c1 + np.random.randn(256) * 0.05 for _ in range(4)]
    embs2 = [c2 + np.random.randn(256) * 0.05 for _ in range(4)]
    all_embs = np.vstack([e / np.linalg.norm(e) for e in (embs1 + embs2)])

    ahc = AgglomerativeClustering(threshold=0.50)
    labels = ahc.cluster(all_embs)

    assert len(labels) == 8
    # First 4 should have same label, last 4 should have another
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_online_centroid_clustering():
    online = OnlineCentroidClustering(match_threshold=0.70)
    c1 = np.random.randn(256)
    c1 /= np.linalg.norm(c1)

    # Step 1: first sample creates speaker 0
    s0 = online.step(c1)
    assert s0 == 0

    # Step 2: sample near c1 assigns to speaker 0
    near_c1 = c1 + np.random.randn(256) * 0.05
    near_c1 /= np.linalg.norm(near_c1)
    assert online.step(near_c1) == 0

    # Step 3: distant sample creates speaker 1
    c2 = -c1  # opposite
    assert online.step(c2) == 1


def test_conversation_diarization(temp_dir):
    vad = SileroVAD()
    encoder = WeSpeakerEncoder()
    engine = SpeakerDiarizationEngine(vad=vad, encoder=encoder, clustering_threshold=0.65)

    audio = load_audio("samples/conversation_test.wav")
    res = engine.diarize(audio)

    assert res.num_speakers == 2
    assert len(res.segments) >= 2
    for seg in res.segments:
        assert seg.start < seg.end
        assert seg.speaker_id in ("speaker_1", "speaker_2")
