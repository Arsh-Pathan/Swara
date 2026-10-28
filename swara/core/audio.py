"""Audio I/O, validation, resampling, and normalization."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf

from swara.core.types import AudioChunk


class AudioError(Exception):
    """Base exception for audio processing errors."""


class AudioValidationError(AudioError):
    """Raised when audio fails sanity or duration checks."""


def load_audio(
    source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    sample_rate: Optional[int] = None,
    target_sample_rate: int = 16000,
    normalize: bool = True,
    max_duration_seconds: Optional[float] = 1800.0,
) -> AudioChunk:
    """Loads, validates, downmixes to mono, and resamples audio to target sample rate.

    Args:
        source: File path, raw bytes, BytesIO stream, or raw numpy array.
        sample_rate: Original sample rate if source is raw numpy array.
        target_sample_rate: Target sample rate for model inference (default 16000).
        normalize: Whether to peak-normalize audio to [-0.95, 0.95].
        max_duration_seconds: Maximum allowed audio duration.

    Returns:
        AudioChunk with normalized 1D float32 samples at target_sample_rate.

    Raises:
        AudioValidationError: If audio is corrupt, empty, exceeds duration limits, or contains NaNs.
    """
    raw_samples: np.ndarray
    orig_sr: int

    if isinstance(source, np.ndarray):
        raw_samples = source
        orig_sr = sample_rate if sample_rate is not None else target_sample_rate
    elif isinstance(source, (bytes, io.BytesIO)):
        bio = io.BytesIO(source) if isinstance(source, bytes) else source
        try:
            raw_samples, orig_sr = sf.read(bio, dtype="float32")
        except Exception as e:
            raise AudioValidationError(f"Failed to decode audio bytes: {e}") from e
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise AudioValidationError(f"Audio file not found: {p}")
        if p.stat().st_size == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): {p}")
        try:
            raw_samples, orig_sr = sf.read(str(p), dtype="float32")
        except Exception as e:
            raise AudioValidationError(f"Failed to read audio file '{p}': {e}") from e
    else:
        raise AudioValidationError(f"Unsupported audio source type: {type(source)}")

    # Sanity checks
    if raw_samples is None or len(raw_samples) == 0:
        raise AudioValidationError("Audio data contains zero samples.")

    # Check for NaN / Inf
    if not np.isfinite(raw_samples).all():
        raise AudioValidationError("Audio contains non-finite values (NaN or Inf).")

    # Downmix multi-channel to mono
    if raw_samples.ndim > 1:
        # Shape (samples, channels)
        raw_samples = np.mean(raw_samples, axis=-1)

    raw_samples = raw_samples.astype(np.float32)

    # Check duration limit
    duration_s = len(raw_samples) / float(orig_sr)
    if max_duration_seconds is not None and duration_s > max_duration_seconds:
        raise AudioValidationError(
            f"Audio duration ({duration_s:.1f}s) exceeds maximum allowed limit ({max_duration_seconds:.1f}s)."
        )

    # Resample if needed
    if orig_sr != target_sample_rate:
        raw_samples = resample_audio(raw_samples, orig_sr, target_sample_rate)

    # Peak normalization
    if normalize and len(raw_samples) > 0:
        max_val = np.max(np.abs(raw_samples))
        if max_val > 1e-6:
            raw_samples = raw_samples / max_val * 0.95

    return AudioChunk(samples=raw_samples, sample_rate=target_sample_rate)


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples audio using high-quality polyphase filtering."""
    if orig_sr == target_sr:
        return samples

    # Find greatest common divisor to reduce up/down factors
    import math

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = scipy.signal.resample_poly(samples, up, down).astype(np.float32)
    return resampled


def save_audio(path: Union[str, Path], audio: AudioChunk) -> None:
    """Saves AudioChunk to a 16-bit PCM WAV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), audio.samples, audio.sample_rate, subtype="PCM_16")


def record_microphone(
    duration_seconds: float,
    sample_rate: int = 16000,
    device: Optional[int] = None,
) -> AudioChunk:
    """Records audio from the default or specified local microphone.

    Raises:
        AudioError: If sounddevice is unavailable or no recording device is found.
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise AudioError("sounddevice package is required for microphone recording.") from e

    num_samples = int(duration_seconds * sample_rate)
    try:
        recording = sd.rec(
            num_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as e:
        raise AudioError(f"Microphone recording failed: {e}") from e

    samples = recording.flatten()
    # Normalize
    max_val = np.max(np.abs(samples))
    if max_val > 1e-6:
        samples = samples / max_val * 0.95

    return AudioChunk(samples=samples, sample_rate=sample_rate)
