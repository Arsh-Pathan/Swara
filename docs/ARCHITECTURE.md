# Swara System Architecture Specification

**Product:** Swara  
**Company:** CSKY  
**Category:** Speaker Intelligence AI  
**Tagline:** *"Every Voice Has an Identity."*  
**Version:** 1.0.0  

---

## 1. System Overview & Design Philosophy

Swara is an ultra-low-latency, lightweight, production-grade speaker intelligence engine engineered from the ground up for **local-first** and edge deployments. 

Unlike typical speech systems that require heavy multi-gigabyte frameworks (e.g. PyTorch, Torchaudio, NeMo) or remote cloud AI APIs, Swara operates with an ultra-compact footprint:
- **Total Model Footprint:** 27.5 MB (FP32) or 8.6 MB (INT8 Quantized).
- **Cold Initialization Time:** < 100 ms.
- **Inference Latency:** ~20 ms for a 2-second utterance on commodity CPU.
- **Real-Time Factor (RTF):** ~0.01 (processing 100 seconds of audio in 1 second).
- **Zero Cloud Leakage:** 100% of feature extraction, VAD, embedding generation, enrollment, and diarization execute strictly locally.

---

## 2. Core Architectural Components

```
+-------------------------------------------------------------------------+
|                              USER INTERFACES                            |
|    Python API (Swara)   |   CLI (swara)   |   FastAPI HTTP REST Server  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                             SWARA ENGINE                                |
+-------------------------------------------------------------------------+
       |                             |                             |
       v                             v                             v
+---------------+             +---------------+             +---------------+
|   VOICE ACT.  |             |  ACOUSTIC DSP |             |    SPEAKER    |
|   DETECTION   |             |   FRONT-END   |             |   EMBEDDING   |
| (Silero VAD)  |             | (Kaldi Fbank) |             | (WeSpeaker)   |
+---------------+             +---------------+             +---------------+
       |                             |                             |
       |                             v                             v
       |                     +---------------+             +---------------+
       |                     | 80-bin Fbanks |             |  256-dim Unit |
       |                     |  with CMN     |             |   Embedding   |
       |                     +---------------+             +---------------+
       |                                                           |
       +-----------------------------+-----------------------------+
                                     |
       +-----------------------------+-----------------------------+
       |                                                           |
       v                                                           v
+-----------------------------+             +-----------------------------+
|    SPEAKER IDENTIFICATION   |             |     SPEAKER DIARIZATION     |
| - Enrollment Management     |             | - Sliding-Window Slicing    |
| - Cosine Similarity Metric  |             | - AHC Cosine Clustering     |
| - Calibrated Confidence     |             | - Online Centroid Tracking  |
| - Unknown Voice Rejection   |             | - Segment Smoothing         |
| - Persistent Profile Store  |             | - Identity-Enhanced Mapping |
+-----------------------------+             +-----------------------------+
```

### 2.1 Audio Ingestion & Normalization (`swara.core.audio`)
- Ingests standard audio files (WAV, FLAC, OGG, MP3) and raw numpy buffers.
- Enforces downmixing of multi-channel stereo to single-channel mono.
- Polyphase filtering resampling (`scipy.signal.resample_poly`) converts arbitrary input sampling frequencies to standard 16,000 Hz.
- Peak normalization to range $[-0.95, 0.95]$ with clipping prevention.
- Validates inputs against corruption, zero-duration, non-finite values (`NaN`, `Inf`), and maximum duration thresholds.

### 2.2 Kaldi Mel Filterbank DSP (`swara.core.dsp`)
- Pure NumPy/SciPy implementation of Kaldi-compliant 80-bin Mel Filterbanks.
- Framing: 25 ms frame length (400 samples at 16kHz) with 10 ms frame shift (160 samples).
- Pre-emphasis high-pass filter: $s[n] - 0.97 \cdot s[n-1]$.
- Hamming window smoothing.
- Real FFT power spectrum with Kaldi triangular Mel filterbank weighting.
- Per-utterance Cepstral Mean Normalization (CMN) to eliminate acoustic channel bias.

### 2.3 Voice Activity Detection (`swara.vad`)
- Silero VAD v5 ONNX neural network (2.22 MB).
- Consumes 512-sample audio chunks (32 ms) with 64-sample leading context window.
- Stateful LSTM recurrent memory tracking speech state across streaming chunks.
- Hysteresis onset (0.50) and offset (0.35) thresholds with minimum speech and silence duration filters.

### 2.4 Deep Speaker Encoder (`swara.embedding`)
- WeSpeaker ResNet-34 Large-Margin (LM) fine-tuned on VoxCeleb (25.31 MB FP32 / 6.38 MB INT8).
- Output representation: 256-dimensional $L_2$-normalized dense vector.
- Verified on VoxCeleb1-O test protocol with 1.10% Equal Error Rate (EER).
- Runs on ONNX Runtime CPU (or optional CUDA execution provider).

### 2.5 Speaker Identification & Persistent Store (`swara.identification`)
- Vectorized cosine similarity matching:
  $$\cos(u, v) = u \cdot v \in [-1.0, 1.0]$$
- Calibrated confidence metric using logistic mapping:
  $$\text{Confidence}(s) = \frac{1}{1 + e^{-k(s - \theta)}}$$
- Strict unknown speaker rejection when $\max(\text{sim}) < \theta_{\text{id}}$.
- Thread-safe persistent store with atomic file swapping preventing corruption.
- Multi-sample enrollment with running centroid updating.

### 2.6 Speaker Diarization Pipeline (`swara.diarization`)
- **Stage 1 (VAD Segmentation):** Speech boundaries detected, non-speech silence isolated.
- **Stage 2 (Sub-segmentation):** Uniform sliding windows (1.5s window with 0.75s step).
- **Stage 3 (Embedding Extraction):** 256-dim embedding generated per sub-segment.
- **Stage 4 (Clustering):** Agglomerative Hierarchical Clustering (AHC) using average linkage on cosine distance.
- **Stage 5 (Smoothing & Merging):** Adjacent chunks of the same speaker separated by $< 0.35$s collar are merged.
- **Stage 6 (Identity Enhancement):** Diarized cluster centroids cross-matched against enrolled profiles to attribute names.
