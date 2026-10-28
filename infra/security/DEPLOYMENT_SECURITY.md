# Swara Deployment Security & Voice Privacy Architecture

**Product:** Swara  
**Company:** CSKY  
**Standard:** Enterprise Voice Biometric Security Guidelines  

---

## 1. Threat Model & Security Principles

Swara processes human biometric voice data (acoustic waveforms and mathematical embedding vectors). Because human voice identity is immutable personal biometric data, it requires stringent defense-in-depth protections.

### Threat Vectors Addressed
1. **Biometric Theft / Interception:** Unauthorized extraction or snooping of voice embeddings or raw waveforms.
2. **Denial of Service (DoS):** Malicious submission of excessively large or uncompressed audio files to exhaust CPU and RAM.
3. **Impersonation & Replay Attacks:** High-similarity spoofing or unauthorized identity collision.
4. **Filesystem Traversal:** Malicious `speaker_id` values (e.g., `../../etc/passwd`) designed to compromise host files.
5. **Memory Leakage / Unbounded Growth:** Uncontrolled streaming buffers or request queuing.

---

## 2. Hardening Measures Implemented

### 2.1 Least Privilege Container Isolation
- **Non-Root Execution:** The Docker container enforces a non-root system user (`swara`, UID 10001, GID 10001).
- **Minimal Base Image:** Uses `python:3.11-slim` with zero compiler toolchains in the final runtime stage.
- **Resource Constraints:** Hard CPU and memory quotas defined in `compose.yaml` (e.g., 4 CPU cores, 1024 MB RAM limit) to prevent host exhaustion.

### 2.2 Input Validation & Sanitization
- **Strict Size Limits:** Hard ceiling on audio payload uploads (`SWARA_MAX_AUDIO_UPLOAD_SIZE_MB`, default 50 MB) and max audio duration (`SWARA_MAX_AUDIO_DURATION_SECONDS`, default 1800s).
- **Path Traversal Protection:** All `speaker_id` inputs are strictly sanitized to alphanumeric characters, hyphens, and underscores before disk storage resolution:
  ```python
  clean_id = "".join(c for c in speaker_id if c.isalnum() or c in ("-", "_"))
  ```
- **Integrity Validation:** Corrupted waveforms, empty files, and non-finite values (`NaN`, `Inf`) are immediately rejected before neural network inference.

### 2.3 Biometric Data Protection & Privacy
- **No Raw Audio Stored by Default:** Swara persists only 256-dimensional numerical embedding vectors. Raw audio samples are processed in volatile memory and immediately discarded.
- **Biometric Irreversibility:** Deep speaker embeddings extracted from bottleneck convolutional layers cannot be reverse-engineered to reconstruct raw intelligible acoustic speech.
- **Complete Deletion (Right to be Forgotten):** Invoking `DELETE /speakers/{speaker_id}` or `swara speakers delete <id>` atomically erases both the metadata record and the persisted `.npy` embedding vector from disk.
- **Log Sanitization:** Swara logging configurations explicitly forbid writing raw audio samples, feature matrices, or raw 256-dim embedding vectors to stdout or log collectors.

### 2.4 Network & Authentication Boundary
- **Default Localhost Binding:** The server binds to `127.0.0.1` by default for local development.
- **Token Authentication:** When deployed publicly, setting `SWARA_API_KEY` enforces HTTP Bearer token or `X-API-Key` verification across all biometric and speaker endpoints.
- **Reverse Proxy Recommended:** Public deployments must terminate TLS (HTTPS) upstream via Nginx, Traefik, or an AWS/GCP Application Load Balancer.

---

## 3. Biometric Compliance Checklist (GDPR / CCPA / BIPA)

| Requirement | Implementation in Swara |
|---|---|
| **Explicit Consent** | Applications integrating Swara must capture explicit user consent prior to calling `/enroll`. |
| **Data Minimization** | Only 256 floating-point values are retained per voice profile. |
| **Right to Erasure** | Atomic profile and embedding file deletion supported via API and CLI. |
| **Auditability** | Enrolled profiles record ISO 8601 creation, sample count, and last updated timestamps. |
| **Encryption at Rest** | Deployments should mount `swara-data` on encrypted volumes (LUKS, AWS EBS encrypted, or Cloud Key Management). |
