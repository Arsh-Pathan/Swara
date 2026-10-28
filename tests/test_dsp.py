"""Unit tests for Kaldi-compatible 80-bin Mel filterbank feature extraction."""

import numpy as np
import pytest

from swara.core.dsp import KaldiFbank


def test_kaldi_mel_conversion():
    # Frequency to mel and inverse
    f = 1000.0
    mel = KaldiFbank.hz_to_mel(f)
    recovered_f = KaldiFbank.mel_to_hz(mel)
    assert abs(f - recovered_f) < 1e-3


def test_fbank_output_shape():
    fbank = KaldiFbank(sample_rate=16000, num_mel_bins=80)
    # 1.0 second of audio at 16kHz
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    feats = fbank.compute(audio)

    # 16000 samples: (16000 - 400) // 160 + 1 = 98 frames
    assert feats.ndim == 2
    assert feats.shape[1] == 80
    assert abs(feats.shape[0] - 98) <= 2
    assert feats.dtype == np.float32


def test_fbank_cmn_mean_subtraction():
    fbank = KaldiFbank(sample_rate=16000, num_mel_bins=80, cmn=True)
    audio = np.random.randn(32000).astype(np.float32) * 0.1
    feats = fbank.compute(audio)

    # Mean across time frames should be approximately 0.0
    means = np.mean(feats, axis=0)
    assert np.all(np.abs(means) < 1e-5)


def test_fbank_short_utterance():
    fbank = KaldiFbank(sample_rate=16000, num_mel_bins=80)
    # 50 ms audio (800 samples)
    short_audio = np.random.randn(800).astype(np.float32) * 0.1
    feats = fbank.compute(short_audio)
    assert feats.shape[1] == 80
    assert feats.shape[0] >= 1
