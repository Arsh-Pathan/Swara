"""Unit tests for Voice Activity Detection (Silero VAD and Energy VAD)."""

import numpy as np
import pytest

from swara.core.types import AudioChunk
from swara.vad.energy import EnergyVAD
from swara.vad.silero import SileroVAD


def test_energy_vad_speech_detection():
    vad = EnergyVAD(energy_threshold=0.01)
    sr = 16000

    # 1s silence, 1s loud tone, 1s silence
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = np.sin(2 * np.pi * 300 * t) * 0.5
    silence = np.zeros(sr, dtype=np.float32)

    audio = AudioChunk(samples=np.concatenate([silence, tone, silence]), sample_rate=sr)
    segments = vad.detect_segments(audio)

    assert len(segments) >= 1
    # Speech should be roughly between 1.0s and 2.0s
    assert abs(segments[0].start - 1.0) < 0.2
    assert abs(segments[0].end - 2.0) < 0.2


def test_silero_vad_streaming_frame():
    vad = SileroVAD()
    # 512 samples frame
    frame = np.zeros(512, dtype=np.float32)
    prob = vad.process_frame(frame)
    assert 0.0 <= prob <= 1.0
    assert prob < 0.1  # Silence should have low probability


def test_silero_vad_real_speech():
    from swara.core.audio import load_audio
    import os

    sample_path = "samples/en_speech.wav"
    if not os.path.exists(sample_path):
        pytest.skip("en_speech.wav sample not found.")

    audio = load_audio(sample_path)
    vad = SileroVAD()
    segments = vad.detect_segments(audio)

    assert len(segments) > 0
    for s in segments:
        assert s.start < s.end
        assert s.confidence > 0.5
