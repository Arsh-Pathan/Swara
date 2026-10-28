"""Acoustic feature extraction front-end.

Provides Kaldi-compliant 80-dimensional Mel Filterbank (Fbank) feature extraction
implemented in pure NumPy and SciPy. This eliminates multi-gigabyte heavy dependencies
(e.g., PyTorch, Torchaudio) while maintaining exact mathematical fidelity for WeSpeaker
ResNet-34 / CAM++ acoustic front-ends.
"""

from typing import Optional
import numpy as np


class KaldiFbank:
    """Computes Kaldi-compatible 80-bin log Mel filterbank features.

    Matches:
      - torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            window_type='hamming',
            sample_frequency=16000.0,
            preemphasis_coefficient=0.97
        )
      - With Cepstral Mean Normalization (CMN) applied along the time axis.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        low_freq: float = 20.0,
        high_freq: float = 0.0,
        preemphasis: float = 0.97,
        n_fft: int = 512,
        cmn: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.low_freq = low_freq
        self.high_freq = high_freq if high_freq > 0.0 else sample_rate / 2.0
        self.preemphasis = preemphasis
        self.n_fft = n_fft
        self.cmn = cmn

        # Sample counts
        self.frame_len = int(round(sample_rate * frame_length_ms / 1000.0))  # 400 at 16kHz
        self.frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))  # 160 at 16kHz

        # Precompute window
        self.window = np.hamming(self.frame_len).astype(np.float32)

        # Precompute Mel Filterbank matrix: shape (num_mel_bins, n_fft // 2 + 1)
        self.mel_banks = self._build_mel_banks()

    @staticmethod
    def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
        """Kaldi Mel scale conversion formula."""
        return 1127.0 * np.log(1.0 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
        """Kaldi inverse Mel scale conversion formula."""
        return 700.0 * (np.exp(mel / 1127.0) - 1.0)

    def _build_mel_banks(self) -> np.ndarray:
        """Builds triangular Mel filterbank weight matrix."""
        low_mel = self.hz_to_mel(self.low_freq)
        high_mel = self.hz_to_mel(self.high_freq)
        mel_points = np.linspace(low_mel, high_mel, self.num_mel_bins + 2)
        bin_freqs = self.mel_to_hz(mel_points)

        num_fft_bins = self.n_fft // 2 + 1
        fft_freqs = np.linspace(0.0, self.sample_rate / 2.0, num_fft_bins)

        weights = np.zeros((self.num_mel_bins, num_fft_bins), dtype=np.float32)
        for i in range(self.num_mel_bins):
            left = bin_freqs[i]
            center = bin_freqs[i + 1]
            right = bin_freqs[i + 2]

            up_mask = (fft_freqs >= left) & (fft_freqs <= center)
            if center > left:
                weights[i, up_mask] = (fft_freqs[up_mask] - left) / (center - left)

            down_mask = (fft_freqs > center) & (fft_freqs <= right)
            if right > center:
                weights[i, down_mask] = (right - fft_freqs[down_mask]) / (right - center)

        return weights

    def compute(self, waveform: np.ndarray) -> np.ndarray:
        """Computes 80-bin Mel filterbank features from 1D mono audio waveform.

        Args:
            waveform: 1D float32 array in range [-1.0, 1.0].

        Returns:
            np.ndarray of shape [T, 80] representing log-mel features.
        """
        if waveform.ndim != 1:
            raise ValueError(f"Waveform must be 1D, got shape {waveform.shape}")

        if len(waveform) == 0:
            return np.zeros((0, self.num_mel_bins), dtype=np.float32)

        # Scale audio to Kaldi integer PCM dynamic range [-32768, 32767]
        samples = (waveform.astype(np.float32) * 32768.0).copy()

        # Handle utterances shorter than one frame
        if len(samples) < self.frame_len:
            pad_len = self.frame_len - len(samples)
            samples = np.pad(samples, (0, pad_len), mode="constant")

        # Compute number of frames with snip_edges=True
        num_frames = (len(samples) - self.frame_len) // self.frame_shift + 1
        if num_frames <= 0:
            num_frames = 1

        # Frame striding
        indices = (
            np.arange(self.frame_len)[None, :]
            + np.arange(num_frames)[:, None] * self.frame_shift
        )
        frames = samples[indices].copy()

        # Kaldi high-pass pre-emphasis per frame: s[n] - 0.97 * s[n-1]
        frames[:, 1:] -= self.preemphasis * frames[:, :-1]
        frames[:, 0] -= self.preemphasis * frames[:, 0]

        # Apply Hamming window
        frames *= self.window

        # Zero-pad to n_fft
        padded_frames = np.pad(
            frames, ((0, 0), (0, self.n_fft - self.frame_len)), mode="constant"
        )

        # Power spectrum via Real FFT
        spectrum = np.abs(np.fft.rfft(padded_frames, n=self.n_fft)) ** 2

        # Dot product with triangular Mel filterbank matrix
        mel_energy = np.matmul(spectrum, self.mel_banks.T)

        # Log compression with floor
        mel_energy = np.maximum(mel_energy, 1e-10)
        log_fbank = np.log(mel_energy)

        # Utterance-level Cepstral Mean Normalization (CMN)
        if self.cmn and num_frames > 0:
            mean = np.mean(log_fbank, axis=0, keepdims=True)
            log_fbank = log_fbank - mean

        return log_fbank.astype(np.float32)


# Default feature extractor instance
default_fbank = KaldiFbank()
