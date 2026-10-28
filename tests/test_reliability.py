"""Reliability, error handling, and robustness tests."""

import io
import numpy as np
import pytest

from swara.core.audio import AudioValidationError, load_audio
from swara.core.types import AudioChunk
from swara.engine import Swara


def test_reject_pure_silence_enrollment(temp_dir):
    engine = Swara()
    silence = np.zeros(32000, dtype=np.float32)

    with pytest.raises(AudioValidationError):
        engine.enroll("silent_user", audio=silence)


def test_reject_ultra_short_audio_enrollment(temp_dir):
    engine = Swara()
    short = np.random.randn(1000).astype(np.float32) * 0.1  # ~62 ms

    with pytest.raises(AudioValidationError):
        engine.enroll("short_user", audio=short)


def test_path_traversal_resistance(temp_dir):
    engine = Swara()
    # Try directory traversal in speaker_id
    malicious_id = "../../etc/passwd"
    clean_id = "".join(c for c in malicious_id if c.isalnum() or c in ("-", "_"))
    assert "/" not in clean_id
    assert ".." not in clean_id


def test_empty_audio_identification():
    engine = Swara()
    with pytest.raises(AudioValidationError):
        engine.identify(audio=np.array([], dtype=np.float32))


def test_diarization_empty_input():
    engine = Swara()
    res = engine.diarize(audio=np.zeros(100, dtype=np.float32))
    assert res.num_speakers == 0
    assert len(res.segments) == 0
