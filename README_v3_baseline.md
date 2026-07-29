# spot_dat — BSL Fingerspelling Detector

Binary detector for fingerspelling segments in British Sign Language (BSL) video, trained on BOBSL.

## Models

### V3 Detector (`models/variant3_75_25.pt`)

1D CNN binary classifier over per-frame landmark features.

- **Input:** 158-dim features (hand landmarks + wrist distances + motion deltas) over 25-frame windows
- **Architecture:** 1D CNN, hard-negative trained (variant3). The 75/25 ratio is the **hard/easy split within the negative class** — 92,640 hard / 30,881 easy of 123,521 negatives — not a positive/negative split. See `TRAINING_AUDIT_REPORT.md` §6.2.
- **Performance:** 85.8% precision / 92.3% recall on BOBSL validation

**Detection pipeline thresholds:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `t` | 0.87 | Frame-level confidence threshold |
| `k` | 11 | Smoothing kernel size |
| `gap` | 0.5s | Merge gap between adjacent detections |
| `min_dur` | 0.5s | Minimum segment duration |

### GBT Filter (`models/gbt_v3.pkl`)

Gradient-boosted tree post-filter for removing false-positive detections.

- **Input:** 15 segment-level features (duration, confidence stats, hand presence, motion)
- **Performance:** CV F1 = 0.850
- **Threshold:** `conf = 0.60`

## Scripts

### Training & Evaluation
| Script | Description |
|--------|-------------|
| `train_variants.py` | Original variant training (pos/neg ratio sweep) |
| `train_variants_v2.py` | V2 training with hard negatives |
| `extract_unified_dataset.py` | Build training dataset from landmarks |
| `extract_easy_neg_only.py` | Extract easy negatives for baseline |
| `mine_hard_negatives.py` | Mine hard negatives from false positives |
| `evaluate_v2.py` | Evaluate detector variants on BOBSL |
| `evaluate_bobsl.py` | BOBSL-specific evaluation |
| `evaluate_youtube.py` | YouTube domain evaluation |
| `verify_domain_shift.py` | Domain shift analysis (v1) |
| `verify_domain_shift_v2.py` | Domain shift analysis (v2) |
| `analyse_results.py` | Result analysis and metrics |
| `generate_report_v2.py` | Generate evaluation reports |
| `audit_negatives.py` | Audit negative samples |
| `plot_training_curves.py` | Plot training loss/accuracy curves |

### Detection Pipeline (V3 + GBT)
| Script | Description |
|--------|-------------|
| `frame_features.py` | Extract 158-dim frame features from landmarks |
| `run_full_detection_v3.py` | Run V3 detector on full episodes |
| `extract_segment_features.py` | Extract 15-dim segment features for GBT |
| `apply_gbt_filter.py` | Apply GBT post-filter to detections |
| `analyse_full_results_v3.py` | Analyse full detection results |
| `run_evaluation.py` | End-to-end evaluation deep dive |
