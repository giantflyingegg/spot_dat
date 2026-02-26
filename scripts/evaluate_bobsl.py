#!/usr/bin/env python3
"""
Step 4: BOBSL fair evaluation of all variants on 5 held-out episodes.

Evaluates: baseline (original rich_cnn_25f) + 4 hard-negative variants.
Uses optimised postprocessing pipeline.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

# Import feature extraction and postprocessing
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts'))
from frame_features import extract_all_frame_features, TOTAL_FEATURE_DIM
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/postprocessing_optimisation'))
from postprocessing_pipeline import PipelineConfig, run_pipeline, evaluate_segments

# ── Paths ────────────────────────────────────────────────────────────────
LANDMARK_DIR = os.path.expanduser('~/bsl_project/full_episode_detection/landmarks/')
AUTO_ANN_PATH = os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/fingerspelling-automatic-annotations.csv')
PP_CONFIG_PATH = os.path.expanduser(
    '~/bsl_project/improved_detection/postprocessing_optimisation/best_config.json')
BASELINE_MODEL = os.path.expanduser('~/bsl_project/improved_detection/models/rich_cnn_25f.pt')
VARIANT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/models/')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/evaluation/')
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HELD_OUT = [
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
]


# ── Model ────────────────────────────────────────────────────────────────

class CNN1D(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(hidden_dim // 2, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def load_model(model_path):
    """Load model and scaler from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    input_dim = checkpoint.get('input_dim', TOTAL_FEATURE_DIM)
    model = CNN1D(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('scaler_mean'), checkpoint.get('scaler_scale'), \
           checkpoint.get('window_size', 25), input_dim


def run_detector(model, landmarks, scaler_mean, scaler_scale, window_size, input_dim,
                 batch_size=512):
    """Run detector on full episode landmarks. Returns per-frame probabilities."""
    features = extract_all_frame_features(landmarks)
    n_frames = len(features)
    if n_frames < window_size:
        return np.zeros(n_frames)

    if features.shape[1] != input_dim:
        if features.shape[1] < input_dim:
            pad = np.zeros((n_frames, input_dim - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        else:
            features = features[:, :input_dim]

    if scaler_mean is not None and scaler_scale is not None:
        features = (features - scaler_mean) / (scaler_scale + 1e-8)

    windows = np.lib.stride_tricks.sliding_window_view(features, window_size, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    windows = np.ascontiguousarray(windows)

    predictions = []
    model = model.to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.FloatTensor(windows[i:i + batch_size]).to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    predictions = np.array(predictions)
    pad_start = window_size // 2
    pad_end = n_frames - len(predictions) - pad_start
    frame_probs = np.concatenate([
        np.zeros(pad_start), predictions, np.zeros(max(0, pad_end))
    ])
    return frame_probs[:n_frames]


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEP 4: BOBSL FAIR EVALUATION")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load postprocessing config
    with open(PP_CONFIG_PATH) as f:
        pp_cfg = json.load(f)
    config = PipelineConfig(
        smooth_kernel=pp_cfg['smooth_kernel'],
        threshold=pp_cfg['threshold'],
        gap_bridge_s=pp_cfg['gap_bridge_s'],
        min_duration_s=pp_cfg['min_duration_s'],
        conf_filter=pp_cfg['conf_filter'],
        boundary_extend_s=pp_cfg['boundary_extend_s'],
    )
    print(f"Post-processing: {config}")

    # Load annotations
    auto_ann = pd.read_csv(AUTO_ANN_PATH)
    auto_ann['video_id'] = auto_ann['video_id'].astype(str)

    # Models to evaluate
    models = [
        ('Baseline', BASELINE_MODEL),
        ('V1_hard_only', os.path.join(VARIANT_DIR, 'variant1_hard_only.pt')),
        ('V2_50_50', os.path.join(VARIANT_DIR, 'variant2_50_50.pt')),
        ('V3_75_25', os.path.join(VARIANT_DIR, 'variant3_75_25.pt')),
        ('V4_curriculum', os.path.join(VARIANT_DIR, 'variant4_curriculum.pt')),
    ]

    # Check which models exist
    models = [(name, path) for name, path in models if os.path.exists(path)]
    print(f"\nModels to evaluate: {len(models)}")

    all_results = []

    for model_name, model_path in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model, scaler_mean, scaler_scale, ws, input_dim = load_model(model_path)

        total_tp03, total_fp03, total_fn03 = 0, 0, 0
        total_tp05, total_fp05, total_fn05 = 0, 0, 0
        total_non_fs_duration = 0
        ep_results = []

        for ep_id in HELD_OUT:
            # Load landmarks
            lm_path = os.path.join(LANDMARK_DIR, f'{ep_id}_mediapipe.npz')
            data = np.load(lm_path)
            landmarks = data['landmarks']

            # Get GT
            ep_ann = auto_ann[auto_ann['video_id'] == ep_id]
            gt = [{'start': r['start'], 'end': r['end']} for _, r in ep_ann.iterrows()]

            # Run detector
            frame_probs = run_detector(model, landmarks, scaler_mean, scaler_scale, ws, input_dim)

            # Apply postprocessing
            segments = run_pipeline(frame_probs, config)
            preds = [{'start_time': s['start_time'], 'end_time': s['end_time'],
                       'max_prob': s.get('max_prob', 0)} for s in segments]

            # Evaluate at IoU 0.3
            tp03, fp03, fn03 = evaluate_segments(preds, gt, 0.3)[:3]
            # Evaluate at IoU 0.5
            tp05, fp05, fn05 = evaluate_segments(preds, gt, 0.5)[:3]

            total_tp03 += tp03; total_fp03 += fp03; total_fn03 += fn03
            total_tp05 += tp05; total_fp05 += fp05; total_fn05 += fn05

            # Non-FS duration for FP/min
            ep_dur = len(landmarks) / FPS
            fs_dur = sum(r['end'] - r['start'] for _, r in ep_ann.iterrows())
            non_fs_dur = ep_dur - fs_dur
            total_non_fs_duration += non_fs_dur

            recall = tp03 / len(gt) if gt else 0
            ep_results.append({
                'episode_id': ep_id, 'n_gt': len(gt),
                'tp03': tp03, 'fp03': fp03, 'fn03': fn03,
                'tp05': tp05, 'fp05': fp05, 'fn05': fn05,
                'recall03': recall, 'n_preds': len(preds),
            })
            print(f"  {ep_id}: GT={len(gt)}, TP={tp03}, FP={fp03}, FN={fn03}, R={recall:.3f}")

            del landmarks, data

        # Aggregate
        p03 = total_tp03 / (total_tp03 + total_fp03) if (total_tp03 + total_fp03) > 0 else 0
        r03 = total_tp03 / (total_tp03 + total_fn03) if (total_tp03 + total_fn03) > 0 else 0
        f1_03 = 2 * p03 * r03 / (p03 + r03) if (p03 + r03) > 0 else 0

        p05 = total_tp05 / (total_tp05 + total_fp05) if (total_tp05 + total_fp05) > 0 else 0
        r05 = total_tp05 / (total_tp05 + total_fn05) if (total_tp05 + total_fn05) > 0 else 0
        f1_05 = 2 * p05 * r05 / (p05 + r05) if (p05 + r05) > 0 else 0

        fp_per_min = total_fp03 / (total_non_fs_duration / 60) if total_non_fs_duration > 0 else 0

        result = {
            'model': model_name,
            'tp03': total_tp03, 'fp03': total_fp03, 'fn03': total_fn03,
            'precision03': p03, 'recall03': r03, 'f1_03': f1_03,
            'tp05': total_tp05, 'fp05': total_fp05, 'fn05': total_fn05,
            'precision05': p05, 'recall05': r05, 'f1_05': f1_05,
            'fp_per_min': fp_per_min,
        }
        # Add per-episode recall
        for ep_r in ep_results:
            result[f"recall_{ep_r['episode_id'][-4:]}"] = ep_r['recall03']

        all_results.append(result)
        print(f"\n  F1@0.3={f1_03:.3f} (P={p03:.3f} R={r03:.3f}), F1@0.5={f1_05:.3f}, FP/min={fp_per_min:.2f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, 'bobsl_fair_eval.csv'), index=False)
    print(f"\nSaved bobsl_fair_eval.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"BOBSL EVALUATION COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Model':<20} {'F1@0.3':>8} {'F1@0.5':>8} {'P':>8} {'R':>8} {'FP/min':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model']:<20} {r['f1_03']:>8.3f} {r['f1_05']:>8.3f} "
              f"{r['precision03']:>8.3f} {r['recall03']:>8.3f} {r['fp_per_min']:>8.2f}")


if __name__ == '__main__':
    main()
