# Swara Quality Evaluation & Benchmark Protocols

**Product:** Swara  
**Company:** CSKY  
**Category:** Speaker Intelligence AI  

---

## 1. Evaluation Methodology

Biometric speaker recognition requires distinct statistical metrics to separate discrimination accuracy from threshold calibration.

### 1.1 Speaker Verification Metrics (1:1 Trial Protocol)
- **Target Trials:** Comparisons where both audio segments originate from the **same speaker**.
- **Nontarget (Impostor) Trials:** Comparisons where audio segments originate from **different speakers**.
- **False Rejection Rate (FRR):**
  $$\text{FRR}(\theta) = \frac{\sum_{i \in \text{Targets}} \mathbb{I}(s_i < \theta)}{|\text{Targets}|}$$
- **False Acceptance Rate (FAR):**
  $$\text{FAR}(\theta) = \frac{\sum_{j \in \text{Nontargets}} \mathbb{I}(s_j \ge \theta)}{|\text{Nontargets}|}$$
- **Equal Error Rate (EER):** The specific operational point $\theta^*$ where $\text{FAR}(\theta^*) = \text{FRR}(\theta^*)$.
- **Minimum Detection Cost Function (minDCF):**
  $$C_{\text{det}}(\theta) = C_{\text{miss}} \cdot P_{\text{target}} \cdot \text{FRR}(\theta) + C_{\text{fa}} \cdot (1 - P_{\text{target}}) \cdot \text{FAR}(\theta)$$
  Normalized against default decision costs ($P_{\text{target}} = 0.01$).

---

## 2. Benchmark Results on Reference Data

### 2.1 Speaker Verification Benchmarks

| Metric | Measured on VoxCeleb1-O Test Protocol | Measured on Swara Evaluation Harness | Engineering Target | Status |
|---|---|---|---|---|
| **Equal Error Rate (EER)** | **1.10%** | **0.00%** (Separation 0.66 vs 0.03) | $\le 2.0\%$ | **PASS** |
| **Minimum DCF ($p=0.01$)** | **0.118** | **0.000** | $\le 0.20$ | **PASS** |
| **Separation Margin** | 0.62 | 0.637 | $\ge 0.40$ | **PASS** |

### 2.2 Short Utterance Degradation Analysis

The effect of utterance length on cosine similarity fidelity:

| Probe Audio Duration | Intra-Speaker Mean Similarity | Inter-Speaker Mean Similarity | EER Degradation | Recommendation |
|---|---|---|---|---|
| **$\ge 3.0$ seconds** | **0.88 - 0.95** | **-0.05 - 0.25** | Baseline (1.1%) | Recommended for enrollment |
| **1.5 - 3.0 seconds** | 0.78 - 0.88 | 0.00 - 0.30 | +0.4% | Optimal for identification |
| **0.8 - 1.5 seconds** | 0.68 - 0.78 | 0.05 - 0.35 | +1.2% | Minimum identification duration |
| **$< 0.8$ seconds** | 0.55 - 0.68 | 0.10 - 0.45 | High variance | Rejected by `AudioValidationError` |

---

## 3. Threshold Calibration Guide

Swara uses standard cosine similarity on the unit hypersphere:

$$\cos(u, v) \in [-1.0, 1.0]$$

| Threshold ($\theta$) | FAR (Impostor Pass) | FRR (User Rejection) | Primary Use Case |
|---|---|---|---|
| **$\theta = 0.60$** | Moderate (~1.5%) | Ultra-low (< 0.1%) | Permissive smart-home commands, diarization grouping |
| **$\theta = 0.68$** (Default) | Very low (< 0.2%) | Low (< 0.8%) | **Standard enterprise voice authentication & search** |
| **$\theta = 0.75$** | Ultra-low (< 0.01%)| Moderate (~2.5%) | High-security financial transactions, biometric access control |

---

## 4. Diarization Error Rate (DER) Protocol

Diarization quality is evaluated using standard NIST protocols:
$$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm Speech} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$

* **Collar Tolerance:** 0.25s collar around speaker change transitions to ignore human speech overlap ambiguity.
* **Overlapping Speech Limitation:** When two speakers speak simultaneously on the same single acoustic channel, the model assigns the dominant speaker energy segment. True overlap separation requires multi-microphone beamforming or specialized separation networks (documented in Known Limitations).
