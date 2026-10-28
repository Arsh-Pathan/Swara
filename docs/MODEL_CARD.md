# Swara Model Cards & Licensing Specifications

---

## Model 1: WeSpeaker ResNet-34 Large-Margin (Speaker Encoder)

### Summary
* **Model ID:** `wespeaker_resnet34` (and `wespeaker_resnet34_int8`)
* **Architecture:** 34-layer Deep Residual Convolutional Neural Network (ResNet-34) with Large Margin fine-tuning
* **Origin Toolkit:** WeSpeaker (Wenet-e2e)
* **Dataset Trained On:** VoxCeleb1 + VoxCeleb2 (over 1 million utterances across 7,000+ speakers)
* **Input Representation:** 80-bin log Mel filterbank features with Cepstral Mean Normalization (CMN)
* **Input Shape:** `[Batch, Time, 80]` (dynamic time frames)
* **Output Representation:** 256-dimensional $L_2$-normalized floating-point vector (`[Batch, 256]`)
* **Artifact File Size:** 
  * FP32: **25.31 MB** (`wespeaker_resnet34.onnx`)
  * INT8 Quantized: **6.38 MB** (`wespeaker_resnet34_int8.onnx`)
* **SHA-256 Checksums:**
  * FP32: `3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb`
  * INT8: `6416fae0599c33bdaf7188e99695edc5fafde190a102c511c881245a109de4c3`
* **License:** **Apache License 2.0** (Permissive, unrestricted commercial use allowed)
* **License URL:** https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE

### Intended Use
* Extracting discriminative speaker voice representations for enrollment, verification, open-set identification, and diarization.

### Performance & Benchmark
* **VoxCeleb1-O EER:** 1.10%
* **Warm CPU Latency (2s audio):** 20.12 ms on Intel i7-13620H (RTF = 0.010)

---

## Model 2: Silero VAD v5 (Voice Activity Detector)

### Summary
* **Model ID:** `silero_vad`
* **Architecture:** Stateful Recurrent Neural Network (LSTM + Conv1D)
* **Origin Toolkit:** Silero AI
* **Dataset Trained On:** Multi-lingual corpus of speech and background ambient noises
* **Input Representation:** 16,000 Hz raw audio chunk of 512 samples + 64 leading context samples (576 samples total)
* **Recurrent State Shape:** `[2, Batch, 128]` float32
* **Output:** Speech probability scalar $\in [0.0, 1.0]$ + updated recurrent state
* **Artifact File Size:** **2.22 MB** (`silero_vad.onnx`)
* **SHA-256 Checksum:** `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
* **License:** **MIT License** (Permissive, unrestricted commercial use allowed)
* **License URL:** https://github.com/snakers4/silero-vad/blob/master/LICENSE

### Intended Use
* Real-time and offline speech boundary detection, speech segmentation, and silence filtering.
