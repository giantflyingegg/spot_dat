# Hard Negative Mining & Detector Retraining Report

**Date:** 2026-02-09 15:46
**Model:** Rich CNN (158d, 25f window)
**Task:** Improve detector specificity for YouTube interpreter content

---

## 1. Negative Audit

Sampled 500 current training negatives and 500 positives.

### Activity Classification of Current Negatives
- **ACTIVE**: 90.2%
- **AMBIGUOUS**: 2.6%
- **IDLE**: 7.2%

### Wrist Velocity Comparison
| Metric | Positives | Negatives |
|--------|-----------|-----------|
| Mean | 0.1152 | 0.1149 |
| Median | 0.1098 | 0.1103 |

### Key Finding
Current negatives are **NOT idle** — 90.2% are classified as ACTIVE with similar wrist velocity distributions to positives. The 5s temporal buffer already selects active signing periods. The YouTube firing issue is more likely due to **domain shift** than negative quality.

---

## 2. Hard Negative Mining

| Method | Windows | Segments | Notes |
|--------|---------|----------|-------|
| A: Between-annotation | 666,128 | 11,593 | Gaps 2-30s, 1s buffer, 244 episodes |
| B: Subtitle-aligned | — | — | OOM killed at 200/244 episodes |
| **Total (train)** | **666,128** | | Method A only (>2x positives) |

Method B was killed by OOM at ~1.36M windows. Method A alone provided 666,128 windows, more than sufficient (>2x the 283K positives).

---

## 3. Training Results

| Variant | Val AUROC | Val F1 | Epochs |
|---------|-----------|--------|--------|
| V1_hard_only | 1.0000 | 1.0000 | 30 |
| V2_50_50 | 0.9987 | 0.9853 | 50 |
| V3_75_25 | 0.9992 | 0.9886 | 50 |
| V4_curriculum | 1.0000 | 1.0000 | 37 |

Shuffled-label sanity check (V1): AUROC = 0.6648 (expect ~0.5)

---

## 4. BOBSL Fair Evaluation (5 Held-Out Episodes)

| Model | F1@0.3 | F1@0.5 | Precision | Recall | FP/min |
|-------|--------|--------|-----------|--------|--------|
| Baseline | 0.805 | 0.733 | 0.801 | 0.809 | 0.90 |
| V1_hard_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.02 |
| V2_50_50 | 0.007 | 0.002 | 0.571 | 0.003 | 0.01 |
| V3_75_25 | 0.003 | 0.002 | 0.667 | 0.002 | 0.00 |
| V4_curriculum | 0.000 | 0.000 | 0.000 | 0.000 | 0.02 |

---

## 5. YouTube Interpreter Evaluation

| Model | HIGH | MEDIUM | LOW | HIGH/min | Mean prob | BG prob |
|-------|------|--------|-----|----------|-----------|--------|
| Baseline | 2304 | 399 | 7 | 12.1 | 0.121 | 0.074 |
| V1_hard_only | 179 | 851 | 1680 | 0.9 | 0.005 | 0.008 |
| V2_50_50 | 378 | 1402 | 930 | 2.0 | 0.009 | 0.005 |
| V3_75_25 | 137 | 785 | 1788 | 0.7 | 0.004 | 0.001 |
| V4_curriculum | 66 | 559 | 2085 | 0.3 | 0.002 | 0.001 |

### Signing vs Non-Signing Check

**Baseline:**
- 38tq2ze5BJE: interpreter=0.129, full_frame=0.085
- Ch8BkMpJ-Fg: interpreter=0.097, full_frame=0.089
- EjgVYWh2SAA: interpreter=0.122, full_frame=0.078
- ml1N7kX2DMQ: interpreter=0.137, full_frame=0.086

**V1_hard_only:**
- 38tq2ze5BJE: interpreter=0.003, full_frame=0.802
- Ch8BkMpJ-Fg: interpreter=0.010, full_frame=0.504
- EjgVYWh2SAA: interpreter=0.004, full_frame=0.711
- ml1N7kX2DMQ: interpreter=0.003, full_frame=0.654

**V2_50_50:**
- 38tq2ze5BJE: interpreter=0.009, full_frame=0.174
- Ch8BkMpJ-Fg: interpreter=0.008, full_frame=0.123
- EjgVYWh2SAA: interpreter=0.010, full_frame=0.123
- ml1N7kX2DMQ: interpreter=0.009, full_frame=0.131

**V3_75_25:**
- 38tq2ze5BJE: interpreter=0.002, full_frame=0.256
- Ch8BkMpJ-Fg: interpreter=0.008, full_frame=0.249
- EjgVYWh2SAA: interpreter=0.002, full_frame=0.239
- ml1N7kX2DMQ: interpreter=0.002, full_frame=0.302

**V4_curriculum:**
- 38tq2ze5BJE: interpreter=0.002, full_frame=0.627
- Ch8BkMpJ-Fg: interpreter=0.004, full_frame=0.466
- EjgVYWh2SAA: interpreter=0.002, full_frame=0.617
- ml1N7kX2DMQ: interpreter=0.001, full_frame=0.498

---

## 6. Figures

| Figure | Description |
|--------|-------------|
| `wrist_velocity_histogram.png` | Wrist velocity: positives vs negatives |
| `activity_scatter.png` | Wrist velocity vs hand height |
| `negative_composition_pie.png` | IDLE/ACTIVE/AMBIGUOUS split |
| `training_curves.png` | AUROC training curves for all variants |
| `bobsl_comparison.png` | BOBSL F1 comparison bar chart |
| `youtube_firing_rate.png` | YouTube HIGH/min across variants |
| `detector_trace_comparison.png` | Frame-by-frame probability trace |
| `variant_comparison.png` | Full comparison table |

---

## 7. Critical Finding: Data Source Domain Shift

All hard negative variants achieved near-zero F1 on BOBSL while dramatically reducing YouTube false positives. This reveals a **data source confound**:

- **Positives**: From per-annotation landmark files (`mediapipe_features/train/`) — 1,660 episodes
- **Hard negatives**: From full-episode landmark files (`full_episode_detection/landmarks/`) — 244 episodes
- **Evaluation**: Uses full-episode landmarks (same source as hard negatives)

The model learned to distinguish the **landmark extraction pipeline** rather than fingerspelling vs non-fingerspelling. Evidence:
1. Val AUROC = 1.0000 (trivially separable — suspiciously perfect)
2. BOBSL F1 = 0.000 (detects nothing on full-episode data)
3. YouTube mean_prob reduced 60x (0.121 to 0.002)

### Root Cause
Per-annotation landmarks in `mediapipe_features/train/` were extracted differently from full-episode landmarks, creating a distributional confound that the model exploited instead of learning the task.

### Recommendations
1. **Re-extract positives from full-episode landmarks** at annotation timestamps to eliminate the domain confound
2. **Verify pipeline consistency**: Compare per-annotation vs full-episode landmarks for the same annotations
3. **Retrain with matched sources**: Hard negative approach may work correctly once data sources are unified
4. **YouTube threshold tuning**: As an interim fix, apply higher confidence threshold for YouTube

---

*Generated by analyse_results.py*
