"""Pytest test configuration and fixtures."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest

from swara.core.config import Settings
from swara.core.types import AudioChunk
from swara.engine import Swara
from swara.identification.store import SpeakerStore
from swara.models.manager import ModelManager


@pytest.fixture
def temp_dir():
    """Provides a temporary directory that is automatically cleaned up."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def test_config(temp_dir):
    """Provides test settings pointing to temporary storage."""
    cfg = Settings(
        base_dir=temp_dir,
        models_dir=Path("models/cache").resolve(),
        speaker_store_path=temp_dir / "speakers",
        api_key=None,
    )
    return cfg


@pytest.fixture
def clean_store(test_config):
    """Provides a fresh, empty SpeakerStore."""
    store = SpeakerStore(storage_dir=test_config.speaker_store_path)
    return store


@pytest.fixture
def swara_engine(test_config):
    """Provides an initialized Swara engine for testing."""
    engine = Swara(config=test_config, auto_download_models=True)
    return engine


@pytest.fixture
def sample_sine_audio():
    """Generates 2.0s of synthetic 200 Hz tone."""
    sr = 16000
    t = np.linspace(0, 2.0, 32000, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 200 * t)
    return AudioChunk(samples=samples, sample_rate=sr)


@pytest.fixture
def real_speech_audio():
    """Returns sample real speech AudioChunk if file exists."""
    from swara.core.audio import load_audio

    sample_path = Path("samples/speechbrain_example1.wav")
    if sample_path.is_file():
        return load_audio(sample_path)
    # Fallback to harmonic audio if file is missing
    sr = 16000
    t = np.linspace(0, 2.0, 32000, dtype=np.float32)
    samples = (
        0.4 * np.sin(2 * np.pi * 160 * t)
        + 0.2 * np.sin(2 * np.pi * 320 * t)
        + 0.1 * np.sin(2 * np.pi * 480 * t)
    )
    return AudioChunk(samples=samples.astype(np.float32), sample_rate=sr)
