#!/usr/bin/env python3
"""
Step 5: YouTube interpreter evaluation of all variants.

For each of 4 NHM interpreter videos × each model variant:
- Run detector on interpreter landmarks
- Re-score subtitle candidates
- Compute HIGH candidates/min, mean background probability
- Signing vs non-signing sanity check
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts'))
from frame_features import extract_all_frame_features, TOTAL_FEATURE_DIM
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/postprocessing_optimisation'))
from postprocessing_pipeline import PipelineConfig, run_pipeline

# ── Paths ────────────────────────────────────────────────────────────────
YT_LANDMARK_DIR = os.path.expanduser('~/bsl_project/youtube_mining/landmarks/')
YT_OUTPUT_DIR = os.path.expanduser('~/bsl_project/youtube_mining/output/')
PP_CONFIG_PATH = os.path.expanduser(
    '~/bsl_project/improved_detection/postprocessing_optimisation/best_config.json')
BASELINE_MODEL = os.path.expanduser('~/bsl_project/improved_detection/models/rich_cnn_25f.pt')
VARIANT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/models/')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/evaluation/')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YT_VIDEOS = ['38tq2ze5BJE', 'Ch8BkMpJ-Fg', 'EjgVYWh2SAA', 'ml1N7kX2DMQ']
FPS_CANONICAL = 25
HIGH_THRESHOLD = 0.50
SUSTAINED_THRESHOLD = 0.5  # seconds


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
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    input_dim = checkpoint.get('input_dim', TOTAL_FEATURE_DIM)
    model = CNN1D(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('scaler_mean'), checkpoint.get('scaler_scale'), \
           checkpoint.get('window_size', 25), input_dim


def run_detector(model, landmarks, scaler_mean, scaler_scale, window_size, input_dim,
                 batch_size=512):
    features = extract_all_frame_features(landmarks)
    n_frames = len(features)
    if n_frames < window_size:
        return np.zeros(n_frames)

    if features.shape[1] != input_dim:
        features = features[:, :input_dim] if features.shape[1] > input_dim else \
            np.concatenate([features, np.zeros((n_frames, input_dim - features.shape[1]))], axis=1)

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


def score_candidates(candidates_df, frame_probs, fps):
    """Re-score subtitle candidates using new frame probabilities."""
    results = []
    for _, row in candidates_df.iterrows():
        if not row.get('is_candidate', True):
            continue
        if row.get('blacklisted', False):
            continue

        sub_start = row['sub_start']
        sub_end = row['sub_end']
        word = row.get('word', '')

        # Search window: subtitle time ± 10s
        search_start = max(0, sub_start - 10)
        search_end = sub_end + 10

        f_start = int(search_start * fps)
        f_end = min(int(search_end * fps), len(frame_probs))

        if f_end <= f_start:
            continue

        window_probs = frame_probs[f_start:f_end]
        peak_prob = float(window_probs.max())
        mean_prob = float(window_probs.mean())

        # Sustained duration above 0.3
        above = window_probs >= 0.3
        sustained_frames = 0
        max_run = 0
        current_run = 0
        for v in above:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        sustained_s = max_run / fps

        # Classify tier
        if peak_prob >= HIGH_THRESHOLD and sustained_s >= SUSTAINED_THRESHOLD:
            tier = 'HIGH'
        elif peak_prob >= 0.10:
            tier = 'MEDIUM'
        else:
            tier = 'LOW'

        results.append({
            'word': word,
            'sub_start': sub_start,
            'sub_end': sub_end,
            'peak_prob': peak_prob,
            'mean_prob': mean_prob,
            'sustained_s': sustained_s,
            'tier': tier,
        })

    return pd.DataFrame(results)


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEP 5: YOUTUBE INTERPRETER EVALUATION")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load PP config
    with open(PP_CONFIG_PATH) as f:
        pp_cfg = json.load(f)

    # Models
    model_specs = [
        ('Baseline', BASELINE_MODEL),
        ('V1_hard_only', os.path.join(VARIANT_DIR, 'variant1_hard_only.pt')),
        ('V2_50_50', os.path.join(VARIANT_DIR, 'variant2_50_50.pt')),
        ('V3_75_25', os.path.join(VARIANT_DIR, 'variant3_75_25.pt')),
        ('V4_curriculum', os.path.join(VARIANT_DIR, 'variant4_curriculum.pt')),
    ]
    model_specs = [(n, p) for n, p in model_specs if os.path.exists(p)]
    print(f"Models: {len(model_specs)}")

    all_yt_results = []
    signing_check = {}

    # Pre-load interpreter landmarks (they fit in memory)
    yt_landmarks = {}
    yt_fps = {}
    for vid in YT_VIDEOS:
        lm_path = os.path.join(YT_LANDMARK_DIR, f'{vid}_interpreter_landmarks.npz')
        if not os.path.exists(lm_path):
            print(f"  WARNING: Missing landmarks for {vid}")
            continue
        data = np.load(lm_path)
        yt_landmarks[vid] = data['landmarks'] if 'landmarks' in data else data[data.files[0]]
        yt_fps[vid] = float(data.get('fps_actual', FPS_CANONICAL))
        print(f"  {vid}: {len(yt_landmarks[vid])} frames, {yt_fps[vid]:.1f} fps, "
              f"{len(yt_landmarks[vid])/yt_fps[vid]/60:.1f} min")

    # Load full-frame landmarks for signing check
    yt_full_landmarks = {}
    for vid in YT_VIDEOS:
        full_path = os.path.join(YT_LANDMARK_DIR, f'{vid}_landmarks.npz')
        if os.path.exists(full_path):
            data = np.load(full_path)
            yt_full_landmarks[vid] = data['landmarks'] if 'landmarks' in data else data[data.files[0]]

    for model_name, model_path in model_specs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model, scaler_mean, scaler_scale, ws, input_dim = load_model(model_path)

        model_results = []
        for vid in YT_VIDEOS:
            if vid not in yt_landmarks:
                continue

            landmarks = yt_landmarks[vid]
            fps = yt_fps[vid]
            duration_min = len(landmarks) / fps / 60

            # Run detector on interpreter crop
            frame_probs = run_detector(model, landmarks, scaler_mean, scaler_scale, ws, input_dim)

            # Mean probability
            mean_prob = float(frame_probs.mean())
            mean_prob_top10 = float(np.percentile(frame_probs, 90))

            # Load and re-score candidates
            cand_path = os.path.join(YT_OUTPUT_DIR, f'{vid}_interpreter', 'scored_candidates.csv')
            if os.path.exists(cand_path):
                cand_df = pd.read_csv(cand_path)
                scored = score_candidates(cand_df, frame_probs, fps)
                n_high = len(scored[scored['tier'] == 'HIGH'])
                n_medium = len(scored[scored['tier'] == 'MEDIUM'])
                n_low = len(scored[scored['tier'] == 'LOW'])
                high_per_min = n_high / duration_min
            else:
                n_high = n_medium = n_low = 0
                high_per_min = 0

            # Background probability (non-candidate periods)
            candidate_mask = np.zeros(len(frame_probs), dtype=bool)
            if os.path.exists(cand_path):
                for _, row in cand_df.iterrows():
                    if row.get('is_candidate', True) and not row.get('blacklisted', False):
                        fs = max(0, int((row['sub_start'] - 5) * fps))
                        fe = min(len(frame_probs), int((row['sub_end'] + 5) * fps))
                        candidate_mask[fs:fe] = True
            bg_probs = frame_probs[~candidate_mask]
            mean_bg_prob = float(bg_probs.mean()) if len(bg_probs) > 0 else 0

            vid_result = {
                'model': model_name,
                'video': vid,
                'duration_min': duration_min,
                'mean_prob': mean_prob,
                'mean_bg_prob': mean_bg_prob,
                'mean_prob_p90': mean_prob_top10,
                'n_high': n_high,
                'n_medium': n_medium,
                'n_low': n_low,
                'high_per_min': high_per_min,
                'n_total_candidates': n_high + n_medium + n_low,
            }
            model_results.append(vid_result)
            all_yt_results.append(vid_result)

            print(f"  {vid}: HIGH={n_high} ({high_per_min:.1f}/min), "
                  f"MEDIUM={n_medium}, LOW={n_low}, "
                  f"mean_prob={mean_prob:.3f}, bg_prob={mean_bg_prob:.3f}")

            # Signing vs non-signing check (only for first model iteration to save time)
            if model_name not in signing_check:
                signing_check[model_name] = {}
            if vid in yt_full_landmarks:
                full_lm = yt_full_landmarks[vid]
                full_probs = run_detector(model, full_lm, scaler_mean, scaler_scale, ws, input_dim)
                signing_check[model_name][vid] = {
                    'interpreter_mean_prob': mean_prob,
                    'full_frame_mean_prob': float(full_probs.mean()),
                    'full_frame_p90': float(np.percentile(full_probs, 90)),
                }

        # Aggregate across videos
        if model_results:
            total_high = sum(r['n_high'] for r in model_results)
            total_min = sum(r['duration_min'] for r in model_results)
            avg_high_per_min = total_high / total_min if total_min > 0 else 0
            avg_mean_prob = np.mean([r['mean_prob'] for r in model_results])
            avg_bg_prob = np.mean([r['mean_bg_prob'] for r in model_results])
            print(f"\n  AGGREGATE: HIGH/min={avg_high_per_min:.1f}, "
                  f"mean_prob={avg_mean_prob:.3f}, bg_prob={avg_bg_prob:.3f}")

    # Save results
    yt_df = pd.DataFrame(all_yt_results)
    yt_df.to_csv(os.path.join(OUT_DIR, 'youtube_eval.csv'), index=False)
    print(f"\nSaved youtube_eval.csv")

    with open(os.path.join(OUT_DIR, 'signing_vs_nonsigning_check.json'), 'w') as f:
        json.dump(signing_check, f, indent=2)
    print("Saved signing_vs_nonsigning_check.json")

    # Save frame probs for trace comparison (baseline vs best variant)
    # Store baseline probs for one video for later plotting
    if model_specs:
        model, sm, ss, ws, dim = load_model(model_specs[0][1])  # baseline
        vid = YT_VIDEOS[0]
        if vid in yt_landmarks:
            baseline_probs = run_detector(model, yt_landmarks[vid], sm, ss, ws, dim)
            np.savez_compressed(os.path.join(OUT_DIR, f'frame_probs_baseline_{vid}.npz'),
                                probs=baseline_probs)
        # Last variant probs
        if len(model_specs) > 1:
            model, sm, ss, ws, dim = load_model(model_specs[-1][1])
            last_probs = run_detector(model, yt_landmarks[vid], sm, ss, ws, dim)
            np.savez_compressed(os.path.join(OUT_DIR, f'frame_probs_last_variant_{vid}.npz'),
                                probs=last_probs)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"YOUTUBE EVALUATION COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
