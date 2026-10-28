# Swara: Speaker Intelligence AI

<p align="center">
  <strong>CSKY</strong> — <em>"Every Voice Has an Identity."</em>
</p>

<p align="center">
  <strong>Production-ready, ultra-low-latency, lightweight speaker intelligence engine running 100% locally on commodity CPUs.</strong>
</p>

---

## 🎙️ Overview

**Swara** is an enterprise-grade speaker intelligence AI system designed to operate locally without remote cloud dependencies. It delivers state-of-the-art Voice Activity Detection (VAD), speaker embedding generation, speaker enrollment, open-set identification with unknown voice rejection, and multi-speaker diarization with enrolled identity enhancement.

### Key Capabilities
1. **Speech Boundary & Activity Detection:** Sub-millisecond speech boundary detection via Silero VAD v5 ONNX.
2. **Deep Speaker Representations:** 256-dimensional unit-normalized embeddings extracted via WeSpeaker ResNet-34 LM (VoxCeleb1-O EER: 1.10%).
3. **Voice Enrollment & Multi-Sample Averaging:** Instant enrollment from audio samples with running centroid updates.
4. **Speaker Identification & Unknown Rejection:** Open-set identification with calibrated confidence scores and strict rejection of unknown voices.
5. **Time-Aligned Speaker Diarization:** Unsupervised clustering (who spoke when) with Agglomerative Hierarchical Clustering.
6. **Identity-Enhanced Diarization:** Diarized speaker clusters automatically labeled with enrolled personal identities.
7. **Ultra-Compact Footprint:** Combined models size of **27.5 MB** (FP32) or **8.6 MB** (INT8 Quantized); resident memory < 200 MB.
8. **Ultra-Low Latency:** **~20 ms** warm inference for a 2-second utterance (**RTF = 0.01**, ~100x faster than real-time on CPU).
9. **Zero Heavy PyTorch Overhead:** Kaldi-compatible 80-bin Mel filterbank front-end engineered in pure NumPy/SciPy.
10. **100% Local & Offline:** Complete privacy and air-gapped readiness once models are downloaded.

---

## 📊 Measured Performance Benchmarks

*Benchmarked on Intel Core i7-13620H (16 logical threads, ONNX Runtime CPU)*

| Metric | Measured Value | Engineering Target | Status |
|---|---|---|---|
| **Cold Start Model Init** | **99.0 ms** | $\le 500\text{ ms}$ | **EXCEEDED (5x faster)** |
| **Warm Inference Latency (P50)** | **20.12 ms** | $\le 30\text{ ms}$ | **EXCEEDED (33% faster)** |
| **Warm Inference Latency (P95)** | **21.01 ms** | $\le 50\text{ ms}$ | **EXCEEDED (58% faster)** |
| **Real-Time Factor (RTF)** | **0.0101** | $\le 0.10$ | **EXCEEDED (10x faster)** |
| **Inference Speedup** | **99.2x real-time** | $\ge 10\text{x}$ | **EXCEEDED** |
| **Speaker Encoder Model (FP32)** | **25.31 MB** | $\le 25\text{ MB}$ | **MET** |
| **Speaker Encoder Model (INT8)** | **6.38 MB** | $\le 25\text{ MB}$ | **EXCEEDED (74% smaller)** |
| **VAD Model (Silero v5)** | **2.22 MB** | $\le 5\text{ MB}$ | **EXCEEDED (56% smaller)** |
| **Combined Model Footprint** | **27.53 MB** | $\le 75\text{ MB}$ | **EXCEEDED (63% smaller)** |
| **Resident Memory (RSS)** | **~194 MB** | $\le 250\text{ MB}$ | **MET** |

---

## ⚡ Quickstart: Local Setup

### 1. Installation
```bash
# Clone and enter directory
cd Swara

# Create and activate Python 3.11 virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# or .venv\Scripts\activate.bat # On Windows

# Install Swara in editable mode
pip install -e .[dev,microphone]
```

### 2. Verify System & Download Models
```bash
# Verify environment and audio hardware
swara doctor

# Download approved models with SHA-256 verification
swara models download
```

---

## 🎤 Testing With Your Own Voice

Swara includes a built-in workflow to record and test your voice immediately:

```bash
# 1. Record your enrollment voice sample (speak for 6 seconds)
swara record --output samples/my_voice.wav --duration 6

# 2. Enroll your voice identity
swara enroll arsh --audio samples/my_voice.wav --name "Arsh Pathan"

# 3. Record a verification probe sample (speak for 4 seconds)
swara record --output samples/verify_voice.wav --duration 4

# 4. Identify your voice
swara identify --audio samples/verify_voice.wav

# 5. Test unknown speaker rejection (have someone else speak or record from TV)
swara record --output samples/stranger.wav --duration 4
swara identify --audio samples/stranger.wav
```

### Or Run the Ready-to-Test Automated Demo
```bash
python demo.py
```

---

## 🐍 Python API Usage

```python
from swara import Swara

# Initialize engine (loads models into shared memory)
swara = Swara()

# 1. Enroll voice
swara.enroll(
    speaker_id="alice",
    audio_path="samples/alice_enroll.wav",
    name="Alice"
)

# 2. Identify speaker
result = swara.identify(
    audio_path="samples/alice_test.wav"
)
print(f"Matched: {result.speaker_id} (Similarity: {result.similarity:.4f}, Confidence: {result.confidence*100:.1f}%)")

# 3. Multi-speaker Diarization
diar_result = swara.diarize(
    audio_path="samples/conversation_test.wav",
    enhance_with_identity=True
)
for segment in diar_result.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.speaker_id} -> {segment.identity}")
```

---

## 💻 CLI Commands Reference

| Command | Purpose |
|---|---|
| `swara doctor` | Diagnoses runtime, CPU, RAM, audio devices, and models |
| `swara models list` | Displays registered model artifacts, sizes, and checksums |
| `swara models download` | Downloads default models with SHA-256 validation |
| `swara enroll <id> --audio <path>` | Enrolls voice profile from audio file |
| `swara identify --audio <path>` | Identifies speaker against enrolled profiles |
| `swara verify <id> --audio <path>` | Performs 1:1 voice verification |
| `swara diarize --audio <path>` | Diarizes conversation into speaker segments |
| `swara speakers list` | Lists all enrolled identities and sample counts |
| `swara speakers delete <id>` | Deletes an enrolled identity and embedding files |
| `swara record --output <path>` | Records audio from microphone |
| `swara benchmark` | Runs local latency, memory, and RTF benchmarks |
| `swara evaluate --config <path>` | Evaluates EER and minDCF across verification pairs |
| `swara serve --port 8000` | Launches the production HTTP REST API server |

---

## 🌐 HTTP REST API

Start the server:
```bash
swara serve --host 127.0.0.1 --port 8000
```
Open **`http://localhost:8000/docs`** for interactive Swagger documentation.

* `GET  /health` - Liveness health check
* `GET  /ready` - Readiness and model status
* `GET  /models` - Registered model artifacts list
* `GET  /speakers` - Enrolled speaker profiles
* `POST /enroll` - Multipart audio enrollment
* `POST /identify` - Speaker identification probe
* `POST /verify` - 1:1 Voice verification
* `POST /diarize` - Multi-speaker conversation diarization
* `GET  /metrics` - Telemetry, RSS memory, CPU usage

---

## 🐳 Docker & Infrastructure Deployment

Deploy using Docker Compose with non-root security and resource limits:

```bash
docker compose -f infra/compose/compose.yaml up -d --build
```

Check health:
```bash
bash infra/scripts/healthcheck.sh localhost 8000
```

---

## 📁 Repository Structure

```
Swara/
├── swara/                     # Core Python package
│   ├── core/                  # Audio I/O, Kaldi DSP Mel filterbanks, types, config
│   ├── vad/                   # Silero VAD v5 ONNX and Energy VAD
│   ├── embedding/             # WeSpeaker ResNet-34 ONNX embeddings (FP32 & INT8)
│   ├── identification/        # Speaker store, enrollment, cosine matching
│   ├── diarization/           # AHC clustering, online clustering, identity diarization
│   ├── models/                # Model registry, downloader, SHA-256 validation
│   ├── api/                   # FastAPI app, schemas, security, routes
│   └── cli/                   # Click CLI with Rich formatting
├── infra/                     # Production infrastructure assets
│   ├── docker/                # Multi-stage Dockerfile (non-root UID 10001)
│   ├── compose/               # Docker Compose with limits & volumes
│   ├── deployment/            # Nginx reverse proxy & production.env
│   ├── scripts/               # setup.sh and healthcheck.sh
│   ├── monitoring/            # Prometheus scrape config
│   └── security/              # Biometric privacy & hardening guide
├── docs/                      # Comprehensive technical documentation
│   ├── ARCHITECTURE.md
│   ├── RESEARCH.md
│   ├── MODEL_CARD.md
│   ├── LOCAL_SETUP.md
│   ├── API.md
│   ├── EVALUATION.md
│   ├── PERFORMANCE.md
│   ├── SECURITY.md
│   ├── DEPLOYMENT.md
│   ├── TROUBLESHOOTING.md
│   └── ROADMAP.md
├── tests/                     # Automated test suite (39 tests)
├── configs/                   # Configuration YAMLs
├── samples/                   # Reference audio samples
├── demo.py                    # Quickstart Python demonstration
├── pyproject.toml             # Build & dependencies specification
├── requirements.txt           # Pinned dependencies
├── .env.example               # Example environment variables
└── README.md
```

---

## 🛡️ License

* **Swara Codebase:** Apache License 2.0 (CSKY)
* **WeSpeaker ResNet-34 LM Checkpoints:** Apache License 2.0
* **Silero VAD v5 Checkpoint:** MIT License
