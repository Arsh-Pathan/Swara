"""Unit tests for audio loading, normalization, resampling, and validation."""

import io
import numpy as np
import pytest
import soundfile as sf

from swara.core.audio import (
    AudioValidationError,
    load_audio,
    resample_audio,
    save_audio,
)
from swara.core.types import AudioChunk


def test_load_audio_from_numpy():
    samples = np.random.randn(16000).astype(np.float32) * 0.1
    chunk = load_audio(samples, sample_rate=16000)
    assert isinstance(chunk, AudioChunk)
    assert chunk.sample_rate == 16000
    assert len(chunk.samples) == 16000
    assert chunk.duration == 1.0


def test_load_audio_multichannel_downmix():
    # Stereo 2-channel input
    stereo = np.random.randn(16000, 2).astype(np.float32) * 0.1
    chunk = load_audio(stereo, sample_rate=16000)
    assert chunk.samples.ndim == 1
    assert len(chunk.samples) == 16000


def test_resample_audio():
    # 8 kHz to 16 kHz
    orig_sr = 8000
    target_sr = 16000
    t = np.linspace(0, 1.0, 8000, dtype=np.float32)
    tone = np.sin(2 * np.pi * 300 * t)

    resampled = resample_audio(tone, orig_sr, target_sr)
    assert len(resampled) == 16000
    assert resampled.dtype == np.float32


def test_audio_normalization():
    loud = np.ones(8000, dtype=np.float32) * 5.0
    chunk = load_audio(loud, sample_rate=16000, normalize=True)
    assert np.max(np.abs(chunk.samples)) <= 0.96


def test_save_and_reload(temp_dir):
    wav_path = temp_dir / "test_io.wav"
    samples = np.random.randn(16000).astype(np.float32) * 0.2
    chunk = AudioChunk(samples=samples, sample_rate=16000)

    save_audio(wav_path, chunk)
    assert wav_path.is_file()

    reloaded = load_audio(wav_path)
    assert reloaded.sample_rate == 16000
    assert abs(reloaded.duration - 1.0) < 0.01


def test_reject_empty_audio():
    empty = np.array([], dtype=np.float32)
    with pytest.raises(AudioValidationError):
        load_audio(empty, sample_rate=16000)


def test_reject_non_finite_audio():
    bad = np.array([0.1, np.nan, 0.5], dtype=np.float32)
    with pytest.raises(AudioValidationError):
        load_audio(bad, sample_rate=16000)


def test_reject_exceeded_duration():
    long_audio = np.zeros(16000 * 20, dtype=np.float32)
    with pytest.raises(AudioValidationError):
        load_audio(long_audio, sample_rate=16000, max_duration_seconds=5.0)
