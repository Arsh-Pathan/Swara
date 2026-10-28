# Swara HTTP REST API Specification

The Swara HTTP REST API provides production endpoints for speaker enrollment, identification, verification, diarization, and system telemetry.

* **Base URL:** `http://localhost:8000`
* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **OpenAPI Schema (JSON):** `http://localhost:8000/openapi.json`

---

## 1. Authentication

When `SWARA_API_KEY` is configured in the environment, all endpoints (except `/health` and `/ready`) require either:
* Header `Authorization: Bearer <token>`
* Header `X-API-Key: <token>`

---

## 2. Endpoints

### 2.1 Liveness Probe
`GET /health`

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "service": "swara-speaker-intelligence"
}
```

---

### 2.2 Readiness Probe
`GET /ready`

**Response (`200 OK`):**
```json
{
  "ready": true,
  "models_loaded": true,
  "device": "CPUExecutionProvider",
  "active_model": "wespeaker-resnet34-lm",
  "enrolled_speakers_count": 2
}
```

---

### 2.3 Enroll Speaker
`POST /enroll`

**Content-Type:** `multipart/form-data`

**Parameters:**
* `speaker_id` (string, required): Unique identifier (e.g., `"arsh"`).
* `name` (string, optional): Display name (e.g., `"Arsh Pathan"`).
* `audio` (file, required): Audio file (WAV, FLAC, OGG).

**Example `curl` Request:**
```bash
curl -X POST "http://localhost:8000/enroll" \
     -F "speaker_id=arsh" \
     -F "name=Arsh Pathan" \
     -F "audio=@samples/my_voice.wav"
```

**Response (`200 OK`):**
```json
{
  "speaker_id": "arsh",
  "name": "Arsh Pathan",
  "created_at": "2026-10-28T16:01:15.280275",
  "updated_at": "2026-10-28T16:01:15.280275",
  "sample_count": 1,
  "total_audio_duration_s": 5.2,
  "embedding_dim": 256,
  "model_version": "wespeaker-resnet34-lm",
  "metadata": {}
}
```

---

### 2.4 Identify Speaker
`POST /identify`

**Content-Type:** `multipart/form-data`

**Parameters:**
* `audio` (file, required): Audio probe file.
* `threshold` (float, optional): Custom cosine similarity threshold (default: `0.68`).

**Example `curl` Request:**
```bash
curl -X POST "http://localhost:8000/identify" \
     -F "audio=@samples/verify_voice.wav"
```

**Response (`200 OK` - Match Found):**
```json
{
  "speaker_id": "arsh",
  "enrolled_id": "arsh",
  "similarity": 0.8842,
  "confidence": 0.9361,
  "is_match": true,
  "audio_duration": 4.12,
  "candidates": [
    {
      "speaker_id": "arsh",
      "similarity": 0.8842,
      "confidence": 0.9361,
      "is_match": true
    }
  ]
}
```

**Response (`200 OK` - Unknown Voice):**
```json
{
  "speaker_id": "UNKNOWN",
  "enrolled_id": null,
  "similarity": 0.2104,
  "confidence": 0.0035,
  "is_match": false,
  "audio_duration": 3.85,
  "candidates": [
    {
      "speaker_id": "arsh",
      "similarity": 0.2104,
      "confidence": 0.0035,
      "is_match": false
    }
  ]
}
```

---

### 2.5 1:1 Voice Verification
`POST /verify`

**Content-Type:** `multipart/form-data`

**Parameters:**
* `speaker_id` (string, required): Target enrolled speaker identity.
* `audio` (file, required): Audio probe file.
* `threshold` (float, optional): Custom threshold (default: `0.68`).

**Response (`200 OK`):**
```json
{
  "speaker_id": "arsh",
  "verified": true,
  "similarity": 0.8842,
  "confidence": 0.9361,
  "threshold": 0.68
}
```

---

### 2.6 Speaker Diarization
`POST /diarize`

**Content-Type:** `multipart/form-data`

**Parameters:**
* `audio` (file, required): Multi-speaker audio conversation file.
* `enhance_with_identity` (boolean, default: `true`): Cross-match diarized speaker clusters with enrolled identities.
* `threshold` (float, optional): Clustering distance threshold.
* `min_speakers` (int, optional): Minimum speaker hint.
* `max_speakers` (int, optional): Maximum speaker hint.

**Example `curl` Request:**
```bash
curl -X POST "http://localhost:8000/diarize" \
     -F "audio=@samples/conversation_test.wav"
```

**Response (`200 OK`):**
```json
{
  "num_speakers": 2,
  "duration": 12.06,
  "audio_path": "conversation_test.wav",
  "segments": [
    {
      "start": 0.07,
      "end": 1.53,
      "duration": 1.47,
      "speaker_id": "speaker_1",
      "identity": "bob",
      "similarity": 0.9305
    },
    {
      "start": 2.18,
      "end": 4.03,
      "duration": 1.85,
      "speaker_id": "speaker_1",
      "identity": "bob",
      "similarity": 0.9305
    },
    {
      "start": 4.39,
      "end": 7.71,
      "duration": 3.32,
      "speaker_id": "speaker_2",
      "identity": "alice",
      "similarity": 0.9783
    },
    {
      "start": 8.03,
      "end": 9.95,
      "duration": 1.92,
      "speaker_id": "speaker_1",
      "identity": "bob",
      "similarity": 0.9305
    }
  ]
}
```

---

### 2.7 List Enrolled Speakers
`GET /speakers`

**Response (`200 OK`):**
```json
[
  {
    "speaker_id": "alice",
    "name": "Alice",
    "created_at": "2026-10-28T16:01:15.280275",
    "updated_at": "2026-10-28T16:01:15.280275",
    "sample_count": 1,
    "total_audio_duration_s": 3.26,
    "embedding_dim": 256,
    "model_version": "wespeaker-resnet34-lm",
    "metadata": {}
  }
]
```

---

### 2.8 Delete Enrolled Speaker
`DELETE /speakers/{speaker_id}`

**Response (`200 OK`):**
```json
{
  "status": "deleted",
  "speaker_id": "alice"
}
```

---

### 2.9 Runtime Telemetry & Metrics
`GET /metrics`

**Response (`200 OK`):**
```json
{
  "memory_rss_mb": 194.2,
  "memory_vms_mb": 420.5,
  "cpu_percent": 1.2,
  "threads_count": 16,
  "enrolled_speakers": 2,
  "model_version": "wespeaker-resnet34-lm",
  "embedding_dim": 256
}
```
