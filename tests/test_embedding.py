"""Unit tests for WeSpeaker ResNet-34 speaker embedding extractor."""

import numpy as np
import pytest

from swara.core.types import AudioChunk
from swara.embedding.wespeaker import WeSpeakerEncoder


def test_wespeaker_embedding_shape_and_norm():
    encoder = WeSpeakerEncoder()
    assert encoder.embedding_dim == 256

    # 1.5 seconds of audio
    audio = np.random.randn(24000).astype(np.float32) * 0.1
    emb = encoder.extract_embedding(audio)

    assert emb.shape == (256,)
    assert emb.dtype == np.float32
    # Verify unit norm L2 = 1.0
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4


def test_wespeaker_int8_embedding():
    encoder_int8 = WeSpeakerEncoder(use_int8=True)
    assert encoder_int8.embedding_dim == 256
    assert "int8" in encoder_int8.model_version

    audio = np.random.randn(24000).astype(np.float32) * 0.1
    emb = encoder_int8.extract_embedding(audio)
    assert emb.shape == (256,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_cosine_similarity_properties():
    encoder = WeSpeakerEncoder()
    emb = np.random.randn(256).astype(np.float32)
    emb = emb / np.linalg.norm(emb)

    # Self similarity should be 1.0
    sim_self = encoder.compute_similarity(emb, emb)
    assert abs(sim_self - 1.0) < 1e-5

    # Orthogonal or opposite
    sim_opp = encoder.compute_similarity(emb, -emb)
    assert abs(sim_opp - (-1.0)) < 1e-5


def test_calibrated_confidence_metric():
    # Cosine similarity equal to threshold gives 0.50 confidence
    thresh = 0.68
    conf_mid = WeSpeakerEncoder.calibrate_confidence(thresh, threshold=thresh)
    assert abs(conf_mid - 0.50) < 1e-3

    # High similarity (>0.90) gives near 1.0 confidence
    conf_high = WeSpeakerEncoder.calibrate_confidence(0.95, threshold=thresh)
    assert conf_high > 0.95

    # Low similarity (<0.30) gives near 0.0 confidence
    conf_low = WeSpeakerEncoder.calibrate_confidence(0.20, threshold=thresh)
    assert conf_low < 0.01
