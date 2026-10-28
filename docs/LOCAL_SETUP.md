# Local Setup & First-Run Guide

Follow these steps to set up, verify, and run Swara on your local machine, enroll your own voice, and run speaker identification and diarization.

---

## 1. Prerequisites

* **Operating System:** Windows 10/11, macOS (Apple Silicon / Intel), or Linux (Ubuntu 20.04+)
* **Python:** Python 3.9, 3.10, 3.11, or 3.12 (Python 3.11 recommended)
* **Microphone:** Built-in or external USB microphone (if recording live voice)

---

## 2. Installation Steps

### Step 2.1: Clone & Create Virtual Environment

```bash
# Navigate to project directory
cd Swara

# Create virtual environment (Python 3.11)
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### Step 2.2: Install Swara Package

```bash
# Upgrade pip
pip install --upgrade pip

# Install Swara in editable mode with development & microphone extras
pip install -e .[dev,microphone]
```

### Step 2.3: Verify System Diagnostics

Run the system doctor to inspect your environment, hardware, and audio drivers:

```bash
swara doctor
```

Expected output:
```
┌────────────────────────────────────────┐
│ Swara Environment & System Diagnostics │
└────────────────────────────────────────┘
                            System & Runtime Checks                            
┌───────────────────────────┬──────────┬──────────────────────────────────────┐
│ Component                 │ Status   │ Details                              │
├───────────────────────────┼──────────┼──────────────────────────────────────┤
│ Python Version            │ PASS     │ v3.11.x                              │
│ CPU Logical Cores         │ PASS     │ 16 threads available                 │
│ RAM Memory                │ PASS     │ 15.7 GB total                        │
│ ONNX Providers            │ OK       │ CPUExecutionProvider                 │
│ Microphone Support        │ PASS     │ Audio input device detected          │
│ Model: silero_vad         │ VERIFIED │ silero_vad.onnx (2.22 MB)            │
│ Model: wespeaker_resnet34 │ VERIFIED │ wespeaker_resnet34.onnx (25.31 MB)   │
└───────────────────────────┴──────────┴──────────────────────────────────────┘
All systems operational! Swara is ready for offline inference.
```

---

## 3. Acquire & Verify Model Artifacts

Download the approved ONNX models with automatic SHA-256 integrity verification:

```bash
# Download default production models (Silero VAD + WeSpeaker ResNet-34)
swara models download

# Or download all models including the ultra-compact INT8 variant:
swara models download --all
```

Check the status of downloaded models anytime:
```bash
swara models list
```

---

## 4. Test With Your Own Voice (Step-by-Step)

### Option A: Using Your Live Microphone

#### Step 1: Record your voice enrollment sample (5 to 8 seconds)
```bash
swara record --output samples/my_voice.wav --duration 6
```
*Speak naturally for 6 seconds (e.g., "Hello, my name is Arsh. I am testing the Swara speaker intelligence system.")*

#### Step 2: Enroll your voice
```bash
swara enroll arsh --audio samples/my_voice.wav --name "Arsh Pathan"
```

#### Step 3: Record a second verification sample (3 to 5 seconds)
```bash
swara record --output samples/verify_voice.wav --duration 4
```
*Speak another sentence into your microphone.*

#### Step 4: Identify your voice
```bash
swara identify --audio samples/verify_voice.wav
```
*Expected Result:*
```
┌───────────────────────┬────────────────┐
│ Field                 │ Value          │
├───────────────────────┼────────────────┤
│ Identity Result       │ arsh           │
│ Match Status          │ VERIFIED MATCH │
│ Cosine Similarity     │ 0.8842         │
│ Calibrated Confidence │ 93.6%          │
└───────────────────────┴────────────────┘
```

#### Step 5: Test Unknown Voice Rejection
Have a friend, family member, or a recorded voice from another device speak into the microphone:
```bash
swara record --output samples/stranger_voice.wav --duration 4
swara identify --audio samples/stranger_voice.wav
```
*Expected Result:*
```
┌───────────────────────┬──────────────────────────────────────────┐
│ Field                 │ Value                                    │
├───────────────────────┼──────────────────────────────────────────┤
│ Identity Result       │ UNKNOWN SPEAKER                          │
│ Match Status          │ REJECTED (Insufficient evidence)         │
└───────────────────────┴──────────────────────────────────────────┘
```

---

### Option B: Using Pre-recorded WAV Audio Files

If you do not have a working microphone or want to test programmatically:

```bash
# 1. Run the ready-to-test automated demo script
python demo.py

# 2. Or enroll and test using the included reference samples
swara enroll alice --audio samples/speechbrain_example1.wav --name "Alice"
swara identify --audio samples/alice_test.wav
swara diarize --audio samples/conversation_test.wav
```

---

## 5. Offline Operation Guarantee

Once `swara models download` has finished:
1. Disconnect your internet connection or turn on Airplane Mode.
2. Run any identification, enrollment, or diarization command:
   ```bash
   swara identify --audio samples/alice_test.wav
   ```
3. Swara executes 100% locally with zero internet calls.
