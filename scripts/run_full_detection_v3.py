#!/usr/bin/env python3
"""
Full BOBSL detection run: V3 Hard-Negative CNN + optimised postprocessing.

Processes all 249 BOBSL episodes using the V3 model (75/25 hard/easy negatives,
trained on unified full-episode source data with 158-dim rich features).

Usage:
    python3 -u run_full_detection_v3.py [--start N] [--end N]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts/'))
sys.path.insert(0, os.path.expanduser(
    '~/bsl_project/improved_detection/postprocessing_optimisation/'))

from frame_features import extract_all_frame_features
from train_models import CNN1D
from postprocessing_pipeline import PipelineConfig, run_pipeline

# ══════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

LANDMARKS_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_episode_detection/landmarks/'))
V3_MODEL_PATH = Path(os.path.expanduser(
    '~/bsl_project/hard_negative_detection/models_v2/variant3_75_25.pt'))
OUT_DIR = Path(os.path.expanduser('~/bsl_project/full_detection_v3/'))

FPS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024

HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}

# Best postprocessing config (same as V2 optimised and 5-episode eval)
PP_CONFIG = PipelineConfig(
    smooth_kernel=11, threshold=0.30, gap_bridge_s=0.5,
    min_duration_s=0.5, conf_filter=0.60, boundary_extend_s=0.2,
)


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_v3_model():
    checkpoint = torch.load(V3_MODEL_PATH, map_location='cpu', weights_only=False)
    model = CNN1D(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(DEVICE)
    return (model, checkpoint['scaler_mean'], checkpoint['scaler_scale'],
            checkpoint['window_size'], checkpoint['input_dim'])


# ══════════════════════════════════════════════════════════════════════════
# FRAME-LEVEL INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def run_cnn_detector(model, landmarks, smean, sscale, ws):
    """Run V3 CNN on full episode -> raw per-frame probabilities."""
    features = extract_all_frame_features(landmarks)
    n_frames = len(features)

    if n_frames < ws:
        return np.zeros(n_frames)

    # Normalise
    features_norm = (features - smean) / (sscale + 1e-8)

    # Sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(features_norm, ws, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    windows = np.ascontiguousarray(windows)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            batch = torch.FloatTensor(windows[i:i + BATCH_SIZE]).to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    predictions = np.array(predictions)
    pad_start = ws // 2
    pad_end = n_frames - len(predictions) - pad_start
    frame_probs = np.concatenate([
        np.zeros(pad_start), predictions, np.zeros(max(0, pad_end))
    ])
    return frame_probs[:n_frames]


# ══════════════════════════════════════════════════════════════════════════
# EPISODE PROCESSING
# ══════════════════════════════════════════════════════════════════════════

def process_episode(ep_id, model, smean, sscale, ws):
    """Process a single episode -> save detection JSON."""
    lm_path = LANDMARKS_DIR / f'{ep_id}_mediapipe.npz'
    if not lm_path.exists():
        return {'status': 'missing', 'ep_id': ep_id}

    t0 = time.time()

    # Load landmarks
    data = np.load(str(lm_path))
    landmarks = data['landmarks']
    n_frames = len(landmarks)
    duration_s = n_frames / FPS
    is_held_out = ep_id in HELD_OUT

    # Run CNN detector
    raw_probs = run_cnn_detector(model, landmarks, smean, sscale, ws)

    # Apply optimised post-processing
    segments = run_pipeline(raw_probs, PP_CONFIG)

    elapsed = time.time() - t0

    # Build output dict (compatible with V2 format)
    output = {
        'episode_id': ep_id,
        'n_frames': int(n_frames),
        'duration_sec': float(duration_s),
        'fps': FPS,
        'n_detections': len(segments),
        'variant': 'v3_hard_neg',
        'is_held_out': is_held_out,
        'pipeline_config': PP_CONFIG.to_dict(),
        'processing_time_s': float(elapsed),
        'detections': [{
            'start_frame': int(d.get('start_frame', int(d['start_time'] * FPS))),
            'end_frame': int(d.get('end_frame', int(d['end_time'] * FPS))),
            'start_time': round(d['start_time'], 4),
            'end_time': round(d['end_time'], 4),
            'duration': round(d['duration'], 4),
            'frame_confidence': round(d['mean_prob'], 6),
            'max_confidence': round(d['max_prob'], 6),
            'combined_confidence': round(d['mean_prob'], 6),
        } for d in segments],
    }

    # Save
    out_path = OUT_DIR / 'detections' / f'{ep_id}_detections.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=1)

    # Free memory
    del landmarks, data, raw_probs

    return {
        'status': 'ok',
        'ep_id': ep_id,
        'n_frames': n_frames,
        'duration_min': duration_s / 60,
        'n_detections': len(segments),
        'time_s': elapsed,
        'is_held_out': is_held_out,
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=999)
    args = parser.parse_args()

    print("=" * 70)
    print("FULL BOBSL DETECTION — V3 Hard-Negative CNN")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {V3_MODEL_PATH}")
    print(f"Post-processing: {PP_CONFIG.label()}")

    # Load model
    print("\nLoading V3 model...")
    model, smean, sscale, ws, input_dim = load_v3_model()
    print(f"  input_dim={input_dim}, window={ws}")

    # Get episode list
    ep_files = sorted([f.stem.replace('_mediapipe', '')
                       for f in LANDMARKS_DIR.glob('*_mediapipe.npz')])
    ep_files = ep_files[args.start:args.end]

    # Check for already-processed episodes (resume support)
    done_dir = OUT_DIR / 'detections'
    already_done = set()
    for f in done_dir.glob('*_detections.json'):
        already_done.add(f.stem.replace('_detections', ''))

    remaining = [e for e in ep_files if e not in already_done]
    n_total = len(ep_files)
    n_remaining = len(remaining)

    if already_done:
        print(f"\nResume: {len(already_done)} already done, {n_remaining} remaining")
    print(f"Processing {n_remaining} episodes (of {n_total} total)")

    # Process
    log_path = OUT_DIR / 'logs' / 'processing_log.txt'
    results = []
    total_dets = 0
    t_start = time.time()

    for idx, ep_id in enumerate(remaining):
        result = process_episode(ep_id, model, smean, sscale, ws)
        results.append(result)

        if result['status'] == 'ok':
            total_dets += result['n_detections']
            elapsed_total = time.time() - t_start
            eps_per_s = (idx + 1) / elapsed_total
            eta = (n_remaining - idx - 1) / eps_per_s if eps_per_s > 0 else 0
            held_flag = " [HELD-OUT]" if result['is_held_out'] else ""

            msg = (f"[{idx+1}/{n_remaining}] {ep_id}: "
                   f"{result['n_frames']} frames ({result['duration_min']:.1f}min), "
                   f"{result['n_detections']} detections, "
                   f"{result['time_s']:.1f}s"
                   f"{held_flag}"
                   f"  [ETA: {eta/60:.1f}min]")
        else:
            msg = f"[{idx+1}/{n_remaining}] {ep_id}: MISSING"

        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

        # Batch summary every 50 episodes
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            batch_summary = {
                'episodes_processed': idx + 1 + len(already_done),
                'total_detections': total_dets,
                'elapsed_min': elapsed / 60,
                'eps_per_min': (idx + 1) / (elapsed / 60),
            }
            batch_path = OUT_DIR / 'results' / 'batch_summaries' / f'batch_{idx+1}.json'
            with open(batch_path, 'w') as f:
                json.dump(batch_summary, f, indent=2)

    # Final summary
    elapsed = time.time() - t_start
    ok_results = [r for r in results if r['status'] == 'ok']
    missing = [r for r in results if r['status'] == 'missing']

    # Count all detections (including previously done)
    total_all_dets = 0
    all_done = list(done_dir.glob('*_detections.json'))
    for f in all_done:
        with open(f) as fh:
            d = json.load(fh)
        total_all_dets += d['n_detections']

    print(f"\n{'=' * 70}")
    print(f"COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"  Episodes processed this run: {len(ok_results)}")
    print(f"  Episodes missing:            {len(missing)}")
    print(f"  Total episodes done:         {len(all_done)}")
    print(f"  Detections this run:         {total_dets:,}")
    print(f"  Total detections (all):      {total_all_dets:,}")
    print(f"  Held-out episodes:           {sum(1 for r in ok_results if r['is_held_out'])}")

    # Save run summary
    summary = {
        'total_episodes': len(all_done),
        'missing_episodes': [r['ep_id'] for r in missing],
        'total_detections': total_all_dets,
        'total_time_min': elapsed / 60,
        'avg_time_per_episode_s': elapsed / max(len(ok_results), 1),
        'pipeline_config': PP_CONFIG.to_dict(),
        'model_path': str(V3_MODEL_PATH),
        'held_out_episodes': list(HELD_OUT),
        'device': str(DEVICE),
    }
    with open(OUT_DIR / 'results' / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
