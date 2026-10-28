# Swara Security & Privacy Guidelines

**Product:** Swara  
**Company:** CSKY  
**Category:** Speaker Intelligence AI  

---

## 1. Biometric Data Classification

Human voice recordings and extracted speaker embeddings are classified as **Sensitive Biometric Data** under global privacy frameworks:
- **European Union:** General Data Protection Regulation (GDPR, Article 9 - Special Category Data).
- **United States (Illinois):** Biometric Information Privacy Act (BIPA).
- **California:** California Consumer Privacy Act (CCPA / CPRA).

---

## 2. Privacy by Design in Swara

### 2.1 Local-First Architecture
Swara is engineered to process 100% of biometric voice data locally on host or container infrastructure.
- Zero audio samples are transmitted to external servers, cloud AI providers, or third-party telemetry aggregators.
- Core inference runs fully in air-gapped offline environments once model artifacts are present.

### 2.2 Numerical Embedding Irreversibility
- Swara stores only 256-dimensional unit-normalized embedding vectors.
- These representations are lossy deep convolutional bottleneck activations that capture speaker vocal tract characteristics rather than phonetic words or linguistic speech content.
- Synthesizing intelligible human speech directly from these raw 256 floating-point values without additional generative acoustic models is mathematically infeasible.

### 2.3 User Consent & Data Lifecycle
- **Explicit Consent:** Organizations deploying Swara must obtain explicit informed biometric consent before calling `/enroll`.
- **Right to Erasure:** Calling `DELETE /speakers/{speaker_id}` immediately and atomically purges the profile record and its `.npy` embedding vector from storage.

---

## 3. Application Security Controls

### 3.1 Path Traversal Immunity
All user-supplied `speaker_id` arguments undergo strict whitelist sanitation:
```python
clean_id = "".join(c for c in speaker_id if c.isalnum() or c in ("-", "_"))
```
Characters such as `..`, `/`, and `\` are stripped, completely preventing directory traversal attacks.

### 3.2 Request Bombing & Resource Exhaustion Defense
- Maximum upload size constrained (`SWARA_MAX_AUDIO_UPLOAD_SIZE_MB=50.0`).
- Maximum duration constrained (`SWARA_MAX_AUDIO_DURATION_SECONDS=1800.0`).
- Payloads exceeding these bounds are rejected immediately with HTTP 413 / 400.

### 3.3 Zero Biometric Data in Logs
Swara logger modules strictly forbid writing raw acoustic waveforms, Mel filterbank matrices, or raw 256-dim embedding vector numbers to stdout, stderr, or logfiles.
