#!/usr/bin/env python3
"""
Extract GBT segment features for all V3 detections across 249 episodes.

For each detection, computes 15 features using raw landmarks + V3 CNN frame probs:
  - Duration, confidence stats (mean, max, std), conf_ramp
  - Wrist height, wrist velocity, hand detection rate
  - Finger spread (L/R/mean), fingertip velocity (L/R)
  - Mouth opening, isolation ratio

Also labels each detection against GT annotations (IoU >= 0.3).

Usage:
    python3 -u extract_segment_features.py [--start N] [--end N]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts/'))
sys.path.insert(0, os.path.expanduser(
    '~/bsl_project/improved_detection/postprocessing_optimisation/'))

from frame_features import (
    extract_all_frame_features,
    _pose_xyz, _hand_xyz, _hand_all_xyz, _dist, _face_xyz,
    P_NOSE, P_L_SHOULDER, P_R_SHOULDER, P_L_WRIST, P_R_WRIST,
    HAND_L_START, HAND_R_START, FINGERTIP_INDICES,
    F_UPPER_LIP_TOP, F_LOWER_LIP_BOTTOM,
)
from train_models import CNN1D

# ── Paths ─────────────────────────────────────────────────────────────────
LANDMARKS_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_episode_detection/landmarks/'))
V3_MODEL_PATH = Path(os.path.expanduser(
    '~/bsl_project/hard_negative_detection/models_v2/variant3_75_25.pt'))
V3_DET_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_detection_v3/detections/'))
ANN_PATH = Path(os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/'
    'fingerspelling-automatic-annotations.csv'))
OUT_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_detection_v3/gbt_filtering/'))

FPS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
IOU_THRESHOLD = 0.3

HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}


# ══════════════════════════════════════════════════════════════════════════
# V3 CNN INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def load_v3_model():
    checkpoint = torch.load(V3_MODEL_PATH, map_location='cpu', weights_only=False)
    model = CNN1D(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(DEVICE)
    return (model, checkpoint['scaler_mean'], checkpoint['scaler_scale'],
            checkpoint['window_size'])


def run_v3_cnn(model, landmarks, smean, sscale, ws):
    """Run V3 CNN → per-frame probabilities."""
    features = extract_all_frame_features(landmarks)
    n_frames = len(features)
    if n_frames < ws:
        return np.zeros(n_frames)

    features_norm = (features - smean) / (sscale + 1e-8)
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
# IoU MATCHING
# ══════════════════════════════════════════════════════════════════════════

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0


def label_detections(detections, gt_list):
    """Match detections to GT, return labels (1=TP, 0=FP) and best IoU."""
    n_det = len(detections)
    if n_det == 0:
        return np.array([]), np.array([])
    if not gt_list:
        return np.zeros(n_det, dtype=int), np.zeros(n_det)

    # Build IoU matrix
    iou_mat = np.zeros((n_det, len(gt_list)))
    for i, d in enumerate(detections):
        for j, g in enumerate(gt_list):
            iou_mat[i, j] = compute_iou(
                d['start_time'], d['end_time'], g['start'], g['end'])

    # Greedy matching
    labels = np.zeros(n_det, dtype=int)
    best_iou = np.zeros(n_det)
    matched_g = set()

    # Sort by highest IoU for greedy assignment
    pairs = []
    for i in range(n_det):
        for j in range(len(gt_list)):
            if iou_mat[i, j] >= IOU_THRESHOLD:
                pairs.append((iou_mat[i, j], i, j))
    pairs.sort(reverse=True)

    matched_d = set()
    for iou_val, i, j in pairs:
        if i in matched_d or j in matched_g:
            continue
        labels[i] = 1
        best_iou[i] = iou_val
        matched_d.add(i)
        matched_g.add(j)

    # Best IoU for unmatched detections
    for i in range(n_det):
        if i not in matched_d:
            best_iou[i] = float(iou_mat[i].max()) if len(gt_list) > 0 else 0.0

    return labels, best_iou


# ══════════════════════════════════════════════════════════════════════════
# SEGMENT FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def extract_segment_features(seg, landmarks, raw_probs, all_segments, episode_duration):
    """Extract the 15 GBT features for a single detection segment."""
    sf = seg['start_frame']
    ef = min(seg['end_frame'], len(landmarks))
    seg_lm = landmarks[sf:ef]
    N = len(seg_lm)

    if N == 0:
        return None

    feats = {}

    # Duration
    feats['duration'] = seg['duration']

    # Confidence features (from raw frame probs)
    prob_slice = raw_probs[sf:ef]
    feats['mean_confidence'] = float(prob_slice.mean())
    feats['max_confidence'] = float(prob_slice.max())
    feats['confidence_std'] = float(prob_slice.std())
    feats['conf_ramp'] = feats['mean_confidence']

    # Shoulder width for normalisation
    l_shoulder = _pose_xyz(seg_lm, P_L_SHOULDER)
    r_shoulder = _pose_xyz(seg_lm, P_R_SHOULDER)
    sw = np.maximum(_dist(l_shoulder, r_shoulder), 1e-6)

    # Wrist height (relative to shoulder centre, normalised)
    l_wrist = _pose_xyz(seg_lm, P_L_WRIST)
    r_wrist = _pose_xyz(seg_lm, P_R_WRIST)
    shoulder_c = (l_shoulder + r_shoulder) / 2
    l_height = (shoulder_c[:, 1] - l_wrist[:, 1]) / sw
    r_height = (shoulder_c[:, 1] - r_wrist[:, 1]) / sw
    feats['mean_wrist_height'] = float(np.mean(np.maximum(l_height, r_height)))

    # Wrist velocity
    l_vel = np.diff(l_wrist, axis=0, prepend=l_wrist[:1])
    r_vel = np.diff(r_wrist, axis=0, prepend=r_wrist[:1])
    l_speed = np.sqrt(np.sum(l_vel ** 2, axis=-1))
    r_speed = np.sqrt(np.sum(r_vel ** 2, axis=-1))
    feats['mean_wrist_velocity'] = float(np.mean(np.maximum(l_speed, r_speed)))

    # Hand detection rate
    lh = seg_lm[:, 1566:1629].reshape(N, 21, 3)
    rh = seg_lm[:, 1629:1692].reshape(N, 21, 3)
    lh_det = np.any(lh != 0, axis=(1, 2)).astype(float)
    rh_det = np.any(rh != 0, axis=(1, 2)).astype(float)
    feats['hand_detection_rate'] = float(np.mean(np.maximum(lh_det, rh_det)))

    # Finger spread (L/R)
    for hand_start, prefix in [(1566, 'l'), (1629, 'r')]:
        all_xyz = _hand_all_xyz(seg_lm, hand_start)
        tip_xyz = all_xyz[:, FINGERTIP_INDICES, :]
        max_spread = np.zeros(N)
        for i in range(5):
            for j in range(i + 1, 5):
                d = _dist(tip_xyz[:, i], tip_xyz[:, j])
                max_spread = np.maximum(max_spread, d)
        feats[f'{prefix}_finger_spread'] = float(np.mean(max_spread / sw))
    feats['mean_finger_spread'] = (feats['l_finger_spread'] + feats['r_finger_spread']) / 2

    # Mouth opening
    upper = _face_xyz(seg_lm, F_UPPER_LIP_TOP)
    lower = _face_xyz(seg_lm, F_LOWER_LIP_BOTTOM)
    opening = _dist(upper, lower)
    feats['mean_mouth_opening'] = float(np.mean(opening / sw))

    # Isolation ratio
    seg_center = (seg['start_time'] + seg['end_time']) / 2
    min_gap = episode_duration
    for other in all_segments:
        if other is seg:
            continue
        other_center = (other['start_time'] + other['end_time']) / 2
        gap = abs(seg_center - other_center)
        if gap < min_gap:
            min_gap = gap
    feats['isolation_ratio'] = seg['duration'] / max(min_gap, 0.1)

    # Fingertip velocities (R and L)
    for hand_start, prefix in [(1629, 'r'), (1566, 'l')]:
        tips_vel = []
        for tip in FINGERTIP_INDICES:
            pos = _hand_xyz(seg_lm, hand_start, tip)
            vel = np.diff(pos, axis=0, prepend=pos[:1])
            speed = np.sqrt(np.sum(vel ** 2, axis=-1))
            tips_vel.append(speed)
        feats[f'{prefix}_fingertip_vel_mean'] = float(np.mean(tips_vel))

    return feats


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=999)
    args = parser.parse_args()

    print("=" * 70)
    print("SEGMENT FEATURE EXTRACTION FOR V3 GBT FILTERING")
    print("=" * 70)

    # Load V3 model
    print("Loading V3 CNN model...")
    model, smean, sscale, ws = load_v3_model()
    print(f"  Device: {DEVICE}, window_size: {ws}")

    # Load annotations
    print("Loading annotations...")
    ann = pd.read_csv(ANN_PATH)
    ann['video_id'] = ann['video_id'].astype(str)
    print(f"  {len(ann)} total annotations")

    # Get episode list from V3 detections
    det_files = sorted(V3_DET_DIR.glob('*_detections.json'))
    ep_ids = [f.stem.replace('_detections', '') for f in det_files]
    ep_ids = ep_ids[args.start:args.end]
    print(f"  {len(ep_ids)} episodes to process")

    # Process all episodes
    all_rows = []
    t_start = time.time()

    for idx, ep_id in enumerate(ep_ids):
        # Load V3 detections
        det_path = V3_DET_DIR / f'{ep_id}_detections.json'
        with open(det_path) as f:
            det_data = json.load(f)
        detections = det_data['detections']
        ep_duration = det_data['duration_sec']

        if not detections:
            continue

        # Load landmarks
        lm_path = LANDMARKS_DIR / f'{ep_id}_mediapipe.npz'
        if not lm_path.exists():
            print(f"  [{idx+1}/{len(ep_ids)}] {ep_id}: landmarks missing, skipping")
            continue

        data = np.load(str(lm_path))
        landmarks = data['landmarks']

        # Run V3 CNN for frame probs
        raw_probs = run_v3_cnn(model, landmarks, smean, sscale, ws)

        # Get GT for this episode
        ep_ann = ann[ann['video_id'] == ep_id]
        gt_list = [{'start': r['start'], 'end': r['end']}
                   for _, r in ep_ann.iterrows()]

        # Label detections
        labels, best_ious = label_detections(detections, gt_list)

        # Extract features for each detection
        n_ok = 0
        for det_idx, det in enumerate(detections):
            feats = extract_segment_features(
                det, landmarks, raw_probs, detections, ep_duration)
            if feats is None:
                continue
            row = {
                'episode_id': ep_id,
                'detection_idx': det_idx,
                'start_time': det['start_time'],
                'end_time': det['end_time'],
                'is_held_out': ep_id in HELD_OUT,
                'label': int(labels[det_idx]),
                'best_iou': float(best_ious[det_idx]),
                **feats,
            }
            all_rows.append(row)
            n_ok += 1

        n_tp = int(labels.sum())
        elapsed = time.time() - t_start
        eps_done = idx + 1
        eta = (len(ep_ids) - eps_done) / (eps_done / elapsed) if eps_done > 0 else 0

        held_flag = " [HELD-OUT]" if ep_id in HELD_OUT else ""
        print(f"  [{eps_done}/{len(ep_ids)}] {ep_id}: "
              f"{len(detections)} dets, {n_ok} feats, "
              f"{n_tp} TP / {len(detections)-n_tp} FP, "
              f"{len(gt_list)} GT{held_flag}"
              f"  [ETA: {eta/60:.1f}min]")

        # Free memory
        del landmarks, data, raw_probs

    # Save
    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / 'segment_features.csv'
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Total segments: {len(df)}")
    print(f"  TP: {df['label'].sum()}, FP: {(1-df['label']).sum():.0f}")
    print(f"  Held-out segments: {df['is_held_out'].sum()}")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
