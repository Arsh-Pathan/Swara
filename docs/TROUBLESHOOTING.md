# Swara Troubleshooting Guide

Common issues, diagnostic procedures, and solutions for Swara.

---

## 1. Diagnostic First Step

Always run the doctor utility first to detect environment inconsistencies:

```bash
swara doctor
```

---

## 2. Common Issues & Solutions

### Issue 1: Model Checksum Mismatch or Download Failure
* **Symptom:** `ModelChecksumError` during model download or startup.
* **Cause:** Network interruption during download or incomplete file on disk.
* **Solution:**
  Force re-downloading the models:
  ```bash
  swara models download --force
  ```

---

### Issue 2: Microphone Not Detected
* **Symptom:** `swara record` fails with `sounddevice.PortAudioError: Error querying device -1`.
* **Cause:** Host has no input microphone connected, or audio permissions are denied.
* **Solution:**
  1. Verify microphone connections in OS settings.
  2. Use file-based workflows if running on a headless server:
     ```bash
     swara enroll user1 --audio path/to/pre_recorded_audio.wav
     ```

---

### Issue 3: Enrolled Voice Returning `UNKNOWN` (False Negative)
* **Symptom:** You speak your voice verification sample, but Swara returns `UNKNOWN SPEAKER`.
* **Causes:**
  1. Enrollment audio was too short (< 2 seconds) or quiet.
  2. Background noise or acoustic environment changed dramatically.
  3. The decision threshold $\theta = 0.68$ is slightly too strict for the microphone.
* **Solutions:**
  - Enroll with a longer voice sample (5 to 8 seconds) or multiple samples:
    ```bash
    swara enroll <name> --audio samples/second_sample.wav
    ```
    *(Swara automatically averages multiple samples to build a more robust centroid!)*
  - Test identification with a slightly lower threshold:
    ```bash
    swara identify --audio samples/test.wav --threshold 0.62
    ```

---

### Issue 4: Silence or Clicks Enrolled Accidentally
* **Symptom:** `AudioValidationError: Audio sample appears to be complete silence or unvoiced noise.`
* **Cause:** Microphone input volume was muted, or the file contained only room ambient noise.
* **Solution:**
  Swara's Voice Activity Detector explicitly guards against enrolling non-speech sounds. Ensure actual speech is present in the audio file.

---

### Issue 5: Port 8000 Already in Use
* **Symptom:** `[Errno 98] Address already in use` when starting `swara serve`.
* **Solution:**
  Specify an alternate port:
  ```bash
  swara serve --port 8080
  ```
