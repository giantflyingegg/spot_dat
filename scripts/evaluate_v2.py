#!/usr/bin/env python3
"""
Step 4 (v2): Combined BOBSL + YouTube evaluation for unified-source variants.

Evaluates baseline + 4 v2 variants on:
- BOBSL: 5 held-out episodes with optimised postprocessing
- YouTube: 4 NHM interpreter videos with candidate re-scoring
"""
import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
import torch

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
VARIANT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/models_v2/')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/evaluation_v2/')
os.makedirs(OUT_DIR, exist_ok=True)

YT_LANDMARK_DIR = os.path.expanduser('~/bsl_project/youtube_mining/landmarks/')
YT_OUTPUT_DIR = os.path.expanduser('~/bsl_project/youtube_mining/output/')

FPS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HELD_OUT = [
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
]

YT_VIDEOS = ['38tq2ze5BJE', 'Ch8BkMpJ-Fg', 'EjgVYWh2SAA', 'ml1N7kX2DMQ']
FPS_CANONICAL = 25
HIGH_THRESHOLD = 0.50
SUSTAINED_THRESHOLD = 0.5


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
    """Run detector on landmarks. Returns per-frame probabilities."""
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

        search_start = max(0, sub_start - 10)
        search_end = sub_end + 10

        f_start = int(search_start * fps)
        f_end = min(int(search_end * fps), len(frame_probs))

        if f_end <= f_start:
            continue

        window_probs = frame_probs[f_start:f_end]
        peak_prob = float(window_probs.max())
        mean_prob = float(window_probs.mean())

        above = window_probs >= 0.3
        current_run = 0
        max_run = 0
        for v in above:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        sustained_s = max_run / fps

        if peak_prob >= HIGH_THRESHOLD and sustained_s >= SUSTAINED_THRESHOLD:
            tier = 'HIGH'
        elif peak_prob >= 0.10:
            tier = 'MEDIUM'
        else:
            tier = 'LOW'

        results.append({
            'word': word, 'sub_start': sub_start, 'sub_end': sub_end,
            'peak_prob': peak_prob, 'mean_prob': mean_prob,
            'sustained_s': sustained_s, 'tier': tier,
        })

    return pd.DataFrame(results)


# ── BOBSL Evaluation ─────────────────────────────────────────────────────

def evaluate_bobsl(model_specs, config, auto_ann):
    """Evaluate all models on BOBSL held-out episodes."""
    print("\n" + "=" * 70)
    print("BOBSL FAIR EVALUATION (5 Held-Out Episodes)")
    print("=" * 70)

    all_results = []

    for model_name, model_path in model_specs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model, scaler_mean, scaler_scale, ws, input_dim = load_model(model_path)

        total_tp03, total_fp03, total_fn03 = 0, 0, 0
        total_tp05, total_fp05, total_fn05 = 0, 0, 0
        total_non_fs_duration = 0

        for ep_id in HELD_OUT:
            lm_path = os.path.join(LANDMARK_DIR, f'{ep_id}_mediapipe.npz')
            if not os.path.exists(lm_path):
                print(f"  WARNING: Missing landmarks for {ep_id}")
                continue

            data = np.load(lm_path)
            landmarks = data['landmarks']

            ep_ann = auto_ann[auto_ann['video_id'] == ep_id]
            gt = [{'start': r['start'], 'end': r['end']} for _, r in ep_ann.iterrows()]

            frame_probs = run_detector(model, landmarks, scaler_mean, scaler_scale, ws, input_dim)

            segments = run_pipeline(frame_probs, config)
            preds = [{'start_time': s['start_time'], 'end_time': s['end_time'],
                       'max_prob': s.get('max_prob', 0)} for s in segments]

            tp03, fp03, fn03 = evaluate_segments(preds, gt, 0.3)[:3]
            tp05, fp05, fn05 = evaluate_segments(preds, gt, 0.5)[:3]

            total_tp03 += tp03; total_fp03 += fp03; total_fn03 += fn03
            total_tp05 += tp05; total_fp05 += fp05; total_fn05 += fn05

            ep_dur = len(landmarks) / FPS
            fs_dur = sum(r['end'] - r['start'] for _, r in ep_ann.iterrows())
            total_non_fs_duration += ep_dur - fs_dur

            recall = tp03 / len(gt) if gt else 0
            print(f"  {ep_id}: GT={len(gt)}, TP={tp03}, FP={fp03}, FN={fn03}, R={recall:.3f}")

            del landmarks, data

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
        all_results.append(result)
        print(f"\n  F1@0.3={f1_03:.3f} (P={p03:.3f} R={r03:.3f}), "
              f"F1@0.5={f1_05:.3f}, FP/min={fp_per_min:.2f}")

        del model
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, 'bobsl_fair_eval.csv'), index=False)
    print(f"\nSaved bobsl_fair_eval.csv")

    print(f"\n{'Model':<20} {'F1@0.3':>8} {'F1@0.5':>8} {'P':>8} {'R':>8} {'FP/min':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model']:<20} {r['f1_03']:>8.3f} {r['f1_05']:>8.3f} "
              f"{r['precision03']:>8.3f} {r['recall03']:>8.3f} {r['fp_per_min']:>8.2f}")

    return all_results


# ── YouTube Evaluation ───────────────────────────────────────────────────

def evaluate_youtube(model_specs):
    """Evaluate all models on YouTube interpreter videos."""
    print("\n" + "=" * 70)
    print("YOUTUBE INTERPRETER EVALUATION")
    print("=" * 70)

    # Pre-load interpreter landmarks
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

    # Pre-load full-frame landmarks for signing check
    yt_full_landmarks = {}
    for vid in YT_VIDEOS:
        full_path = os.path.join(YT_LANDMARK_DIR, f'{vid}_landmarks.npz')
        if os.path.exists(full_path):
            data = np.load(full_path)
            yt_full_landmarks[vid] = data['landmarks'] if 'landmarks' in data else data[data.files[0]]

    all_yt_results = []
    signing_check = {}

    for model_name, model_path in model_specs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model, scaler_mean, scaler_scale, ws, input_dim = load_model(model_path)
        signing_check[model_name] = {}

        model_results = []
        for vid in YT_VIDEOS:
            if vid not in yt_landmarks:
                continue

            landmarks = yt_landmarks[vid]
            fps = yt_fps[vid]
            duration_min = len(landmarks) / fps / 60

            frame_probs = run_detector(model, landmarks, scaler_mean, scaler_scale, ws, input_dim)
            mean_prob = float(frame_probs.mean())

            # Re-score candidates
            cand_path = os.path.join(YT_OUTPUT_DIR, f'{vid}_interpreter', 'scored_candidates.csv')
            if os.path.exists(cand_path):
                cand_df = pd.read_csv(cand_path)
                scored = score_candidates(cand_df, frame_probs, fps)
                n_high = len(scored[scored['tier'] == 'HIGH'])
                n_medium = len(scored[scored['tier'] == 'MEDIUM'])
                n_low = len(scored[scored['tier'] == 'LOW'])
                high_per_min = n_high / duration_min
            else:
                cand_df = None
                n_high = n_medium = n_low = 0
                high_per_min = 0

            # Background probability
            candidate_mask = np.zeros(len(frame_probs), dtype=bool)
            if cand_df is not None:
                for _, row in cand_df.iterrows():
                    if row.get('is_candidate', True) and not row.get('blacklisted', False):
                        fs = max(0, int((row['sub_start'] - 5) * fps))
                        fe = min(len(frame_probs), int((row['sub_end'] + 5) * fps))
                        candidate_mask[fs:fe] = True
            bg_probs = frame_probs[~candidate_mask]
            mean_bg_prob = float(bg_probs.mean()) if len(bg_probs) > 0 else 0

            vid_result = {
                'model': model_name, 'video': vid,
                'duration_min': duration_min, 'mean_prob': mean_prob,
                'mean_bg_prob': mean_bg_prob,
                'n_high': n_high, 'n_medium': n_medium, 'n_low': n_low,
                'high_per_min': high_per_min,
                'n_total_candidates': n_high + n_medium + n_low,
            }
            model_results.append(vid_result)
            all_yt_results.append(vid_result)

            print(f"  {vid}: HIGH={n_high} ({high_per_min:.1f}/min), "
                  f"MEDIUM={n_medium}, LOW={n_low}, "
                  f"mean_prob={mean_prob:.3f}, bg_prob={mean_bg_prob:.3f}")

            # Signing vs non-signing check
            if vid in yt_full_landmarks:
                full_lm = yt_full_landmarks[vid]
                full_probs = run_detector(model, full_lm, scaler_mean, scaler_scale, ws, input_dim)
                signing_check[model_name][vid] = {
                    'interpreter_mean_prob': mean_prob,
                    'full_frame_mean_prob': float(full_probs.mean()),
                }

        if model_results:
            total_high = sum(r['n_high'] for r in model_results)
            total_min = sum(r['duration_min'] for r in model_results)
            avg_high_per_min = total_high / total_min if total_min > 0 else 0
            avg_mean_prob = np.mean([r['mean_prob'] for r in model_results])
            avg_bg_prob = np.mean([r['mean_bg_prob'] for r in model_results])
            print(f"\n  AGGREGATE: HIGH/min={avg_high_per_min:.1f}, "
                  f"mean_prob={avg_mean_prob:.3f}, bg_prob={avg_bg_prob:.3f}")

        del model
        torch.cuda.empty_cache()

    # Save results
    yt_df = pd.DataFrame(all_yt_results)
    yt_df.to_csv(os.path.join(OUT_DIR, 'youtube_eval.csv'), index=False)
    print(f"\nSaved youtube_eval.csv")

    with open(os.path.join(OUT_DIR, 'signing_vs_nonsigning_check.json'), 'w') as f:
        json.dump(signing_check, f, indent=2)
    print("Saved signing_vs_nonsigning_check.json")

    return all_yt_results, signing_check


def main():
    t0 = time.time()
    print("=" * 70)
    print("UNIFIED SOURCE: COMBINED BOBSL + YOUTUBE EVALUATION")
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

    # Models to evaluate: baseline + v2 variants
    model_specs = [
        ('Baseline', BASELINE_MODEL),
        ('V1_hard_only', os.path.join(VARIANT_DIR, 'variant1_hard_only.pt')),
        ('V2_50_50', os.path.join(VARIANT_DIR, 'variant2_50_50.pt')),
        ('V3_75_25', os.path.join(VARIANT_DIR, 'variant3_75_25.pt')),
        ('V4_curriculum', os.path.join(VARIANT_DIR, 'variant4_curriculum.pt')),
    ]
    model_specs = [(name, path) for name, path in model_specs if os.path.exists(path)]
    print(f"\nModels to evaluate: {[n for n, _ in model_specs]}")

    if len(model_specs) <= 1:
        print("WARNING: No v2 variant models found! Only baseline will be evaluated.")

    # BOBSL evaluation
    bobsl_results = evaluate_bobsl(model_specs, config, auto_ann)

    gc.collect()
    torch.cuda.empty_cache()

    # YouTube evaluation
    yt_results, signing_check = evaluate_youtube(model_specs)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
