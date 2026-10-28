# Swara Product & Technical Roadmap

**Company:** CSKY  
**Product:** Swara  
**Tagline:** *"Every Voice Has an Identity."*  

---

## 1. Version 1.0.0 (Current Production Release)

- [x] Pure NumPy/SciPy Kaldi-compatible 80-bin Mel filterbank front-end.
- [x] Silero VAD v5 ONNX implementation with stateful streaming context.
- [x] WeSpeaker ResNet-34 Large-Margin speaker embedding model (256-dim).
- [x] INT8 quantized model variant (6.4 MB model size).
- [x] Speaker Enrollment, 1:1 Verification, and Open-Set Identification.
- [x] Unknown speaker rejection with calibrated confidence scores.
- [x] Multi-speaker diarization with Agglomerative Hierarchical Clustering.
- [x] Identity-Enhanced Diarization attributing segments to registered users.
- [x] Thread-safe persistent speaker store with atomic swapping.
- [x] Full CLI (`doctor`, `models`, `enroll`, `identify`, `diarize`, `record`, `benchmark`, `evaluate`).
- [x] Production FastAPI HTTP REST API with OpenAPI documentation.
- [x] Production container assets (multi-stage Dockerfile, Compose, Nginx, Prometheus).
- [x] 100% offline-first local execution.

---

## 2. Version 1.1.0 (Upcoming)

- [ ] **Streaming WebSocket API:** Real-time bi-directional audio streaming for live transcription and online speaker diarization.
- [ ] **Vector Database Connectors:** Optional Qdrant / Milvus / pgvector storage backend for scaling to 1,000,000+ enrolled enterprise speakers.
- [ ] **AS-Norm Score Normalization:** Adaptive Symmetric Normalization using a cohort database to further reduce False Acceptance Rate on short utterances.

---

## 3. Version 2.0.0 (Future Vision)

- [ ] **Overlapping Speech Separation:** Dual-speaker simultaneous speech decomposition via Conv-TasNet / SepFormer front-end.
- [ ] **Continuous Unsupervised Voice Adaptation:** Incremental online profile updating as enrolled users interact naturally over time.
- [ ] **Multi-Microphone Beamforming:** Spatial acoustic source separation for conference room hardware.
