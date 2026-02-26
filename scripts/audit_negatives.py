#!/usr/bin/env python3
"""
Step 1: Audit current training negatives.

Samples 500 negatives and 500 positives from the pre-extracted per-annotation
MediaPipe landmarks. Computes hand activity metrics and classifies negatives
as IDLE / ACTIVE / AMBIGUOUS to confirm whether the model trained against
non-signing rather than non-fingerspelling signing.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────
POS_DIR = os.path.expanduser('~/bsl_project/mediapipe_features/train/')
NEG_DIR = os.path.expanduser('~/bsl_project/mediapipe_features/train_negatives/')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/audit/')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_SAMPLES = 500
FPS = 25

# ── MediaPipe landmark indices ───────────────────────────────────────────
# Pose landmarks (33×4): x,y,z,visibility per landmark
P_L_SHOULDER = 11  # dims 44-47
P_R_SHOULDER = 12  # dims 48-51
P_L_ELBOW = 13     # dims 52-55
P_R_ELBOW = 14     # dims 56-59
P_L_WRIST = 15     # dims 60-63
P_R_WRIST = 16     # dims 64-67
P_L_HIP = 23       # dims 92-95
P_R_HIP = 24       # dims 96-99
P_NOSE = 0          # dims 0-3

# Hand landmarks (21×3): x,y,z per landmark
# Left hand starts at dim 1566, right hand at 1629
LH_START = 1566
RH_START = 1629
# Fingertip landmark indices within hand (each ×3 for offset)
TIPS = [4, 8, 12, 16, 20]


def get_pose_xyz(landmarks, joint_idx):
    """Extract xyz from pose landmarks. landmarks shape: (T, 1692)."""
    base = joint_idx * 4
    return landmarks[:, base:base+3]


def get_hand_tip_xyz(landmarks, hand_start, tip_idx):
    """Extract xyz for a fingertip."""
    base = hand_start + tip_idx * 3
    return landmarks[:, base:base+3]


def compute_activity_metrics(landmarks, frame_count, detection_flags=None):
    """Compute hand activity metrics for one annotation window.

    Args:
        landmarks: (max_frames, 1692)
        frame_count: actual number of valid frames
        detection_flags: (max_frames, 4) if available
    """
    lm = landmarks[:frame_count]
    T = len(lm)
    if T < 3:
        return None

    # Wrist positions
    l_wrist = get_pose_xyz(lm, P_L_WRIST)
    r_wrist = get_pose_xyz(lm, P_R_WRIST)

    # Shoulder positions (for height reference)
    l_shoulder = get_pose_xyz(lm, P_L_SHOULDER)
    r_shoulder = get_pose_xyz(lm, P_R_SHOULDER)
    shoulder_mid_y = (l_shoulder[:, 1] + r_shoulder[:, 1]) / 2

    # Hip positions (for waist reference)
    l_hip = get_pose_xyz(lm, P_L_HIP)
    r_hip = get_pose_xyz(lm, P_R_HIP)
    hip_mid_y = (l_hip[:, 1] + r_hip[:, 1]) / 2

    # Wrist velocity (frame-to-frame displacement)
    l_wrist_vel = np.linalg.norm(np.diff(l_wrist, axis=0), axis=1)
    r_wrist_vel = np.linalg.norm(np.diff(r_wrist, axis=0), axis=1)
    mean_wrist_vel = (l_wrist_vel.mean() + r_wrist_vel.mean()) / 2
    max_wrist_vel = max(l_wrist_vel.max(), r_wrist_vel.max())

    # Fingertip velocity
    tip_vels = []
    for tip_idx in TIPS:
        for hand_start in [LH_START, RH_START]:
            tip_xyz = get_hand_tip_xyz(lm, hand_start, tip_idx)
            vel = np.linalg.norm(np.diff(tip_xyz, axis=0), axis=1)
            tip_vels.append(vel.mean())
    mean_fingertip_vel = np.mean(tip_vels)

    # Hand height: proportion of frames where wrist is above hip (y < hip_y in MediaPipe)
    l_above = (l_wrist[:, 1] < hip_mid_y).mean()
    r_above = (r_wrist[:, 1] < hip_mid_y).mean()
    hand_above_waist = max(l_above, r_above)

    # Hand above shoulder
    l_above_shoulder = (l_wrist[:, 1] < shoulder_mid_y).mean()
    r_above_shoulder = (r_wrist[:, 1] < shoulder_mid_y).mean()
    hand_above_shoulder = max(l_above_shoulder, r_above_shoulder)

    # Mean wrist height relative to shoulder (negative = above)
    l_wrist_height = (l_wrist[:, 1] - shoulder_mid_y).mean()
    r_wrist_height = (r_wrist[:, 1] - shoulder_mid_y).mean()
    mean_wrist_height = min(l_wrist_height, r_wrist_height)  # most elevated

    # Hand detection rate
    if detection_flags is not None:
        df = detection_flags[:frame_count]
        l_hand_det = df[:, 2].mean() if df.shape[1] > 2 else 0
        r_hand_det = df[:, 3].mean() if df.shape[1] > 3 else 0
        any_hand_det = ((df[:, 2] > 0) | (df[:, 3] > 0)).mean() if df.shape[1] > 3 else 0
    else:
        # Estimate from hand landmark values (all zeros = not detected)
        l_hand_lm = lm[:, LH_START:LH_START+63]
        r_hand_lm = lm[:, RH_START:RH_START+63]
        l_hand_det = (np.abs(l_hand_lm).sum(axis=1) > 0.01).mean()
        r_hand_det = (np.abs(r_hand_lm).sum(axis=1) > 0.01).mean()
        any_hand_det = max(l_hand_det, r_hand_det)

    return {
        'mean_wrist_vel': float(mean_wrist_vel),
        'max_wrist_vel': float(max_wrist_vel),
        'mean_fingertip_vel': float(mean_fingertip_vel),
        'hand_above_waist': float(hand_above_waist),
        'hand_above_shoulder': float(hand_above_shoulder),
        'mean_wrist_height': float(mean_wrist_height),
        'l_hand_detection_rate': float(l_hand_det),
        'r_hand_detection_rate': float(r_hand_det),
        'any_hand_detection_rate': float(any_hand_det),
        'frame_count': int(frame_count),
        'duration_s': float(frame_count / FPS),
    }


def sample_annotations(data_dir, n_samples, rng):
    """Sample n random annotations from per-episode .npz files."""
    files = sorted(glob(os.path.join(data_dir, '*.npz')))
    if not files:
        raise RuntimeError(f"No .npz files in {data_dir}")

    # Build index: (file_idx, annotation_idx, frame_count)
    index = []
    for fi, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        fc = data['frame_counts']
        for ai in range(len(fc)):
            if fc[ai] >= 5:  # need at least 5 frames
                index.append((fi, ai, int(fc[ai])))

    print(f"  Total annotations available: {len(index)}")
    chosen = rng.choice(len(index), size=min(n_samples, len(index)), replace=False)

    results = []
    loaded_cache = {}
    for idx in chosen:
        fi, ai, fc = index[idx]
        if fi not in loaded_cache:
            data = np.load(files[fi], allow_pickle=True)
            loaded_cache[fi] = data
            # Keep cache small — evict entries we don't need right now
            if len(loaded_cache) > 20:
                to_evict = [k for k in loaded_cache if k != fi]
                if to_evict:
                    del loaded_cache[to_evict[0]]

        data = loaded_cache[fi]
        lm = data['landmarks'][ai]
        det_flags = data['detection_flags'][ai] if 'detection_flags' in data else None
        ep_id = os.path.basename(files[fi]).replace('.npz', '')

        metrics = compute_activity_metrics(lm, fc, det_flags)
        if metrics is not None:
            metrics['episode_id'] = ep_id
            metrics['annotation_idx'] = ai
            results.append(metrics)

    return results


def classify_negative(m, pos_p25_vel):
    """Classify a negative as IDLE, ACTIVE, or AMBIGUOUS."""
    low_vel = m['mean_wrist_vel'] < pos_p25_vel * 0.3
    no_hands = m['any_hand_detection_rate'] < 0.3
    high_vel = m['mean_wrist_vel'] >= pos_p25_vel * 0.5
    hands_tracked = m['any_hand_detection_rate'] >= 0.5
    above_waist = m['hand_above_waist'] >= 0.3

    if low_vel or no_hands:
        return 'IDLE'
    elif high_vel and hands_tracked and above_waist:
        return 'ACTIVE'
    else:
        return 'AMBIGUOUS'


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    print("=" * 70)
    print("STEP 1: AUDIT CURRENT TRAINING NEGATIVES")
    print("=" * 70)

    # Sample positives
    print("\nSampling 500 positives...")
    pos_metrics = sample_annotations(POS_DIR, N_SAMPLES, rng)
    print(f"  Got {len(pos_metrics)} positive samples")

    # Sample negatives
    print("\nSampling 500 negatives...")
    neg_metrics = sample_annotations(NEG_DIR, N_SAMPLES, rng)
    print(f"  Got {len(neg_metrics)} negative samples")

    # Compute reference thresholds from positives
    pos_vels = np.array([m['mean_wrist_vel'] for m in pos_metrics])
    pos_p25 = np.percentile(pos_vels, 25)
    pos_p50 = np.percentile(pos_vels, 50)
    print(f"\n  Positive wrist velocity: median={pos_p50:.4f}, P25={pos_p25:.4f}")

    # Classify negatives
    for m in neg_metrics:
        m['classification'] = classify_negative(m, pos_p25)

    counts = {'IDLE': 0, 'ACTIVE': 0, 'AMBIGUOUS': 0}
    for m in neg_metrics:
        counts[m['classification']] += 1

    total = len(neg_metrics)
    print(f"\n  NEGATIVE CLASSIFICATION:")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls}: {cnt} ({cnt/total:.1%})")

    # Save CSVs
    pos_df = pd.DataFrame(pos_metrics)
    pos_df['label'] = 'positive'
    pos_df.to_csv(os.path.join(OUT_DIR, 'positive_audit.csv'), index=False)

    neg_df = pd.DataFrame(neg_metrics)
    neg_df['label'] = 'negative'
    neg_df.to_csv(os.path.join(OUT_DIR, 'negative_audit.csv'), index=False)

    # Summary JSON
    neg_vels = np.array([m['mean_wrist_vel'] for m in neg_metrics])
    neg_hand_det = np.array([m['any_hand_detection_rate'] for m in neg_metrics])
    pos_hand_det = np.array([m['any_hand_detection_rate'] for m in pos_metrics])

    summary = {
        'n_positives': len(pos_metrics),
        'n_negatives': len(neg_metrics),
        'classification': counts,
        'classification_pct': {k: round(v/total*100, 1) for k, v in counts.items()},
        'positive_wrist_vel': {
            'mean': float(pos_vels.mean()),
            'median': float(np.median(pos_vels)),
            'p25': float(pos_p25),
            'p75': float(np.percentile(pos_vels, 75)),
        },
        'negative_wrist_vel': {
            'mean': float(neg_vels.mean()),
            'median': float(np.median(neg_vels)),
            'p25': float(np.percentile(neg_vels, 25)),
            'p75': float(np.percentile(neg_vels, 75)),
        },
        'positive_hand_detection': {
            'mean': float(pos_hand_det.mean()),
            'above_50pct': float((pos_hand_det >= 0.5).mean()),
        },
        'negative_hand_detection': {
            'mean': float(neg_hand_det.mean()),
            'above_50pct': float((neg_hand_det >= 0.5).mean()),
        },
    }

    with open(os.path.join(OUT_DIR, 'audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved audit_summary.json")

    # ── Visualisations ────────────────────────────────────────────────
    print("\nGenerating figures...")

    # 1. Wrist velocity histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(pos_vels.max(), neg_vels.max()) * 1.1, 60)
    ax.hist(pos_vels, bins=bins, alpha=0.6, label=f'Positives (n={len(pos_vels)})',
            color='#e15759', density=True)
    ax.hist(neg_vels, bins=bins, alpha=0.6, label=f'Negatives (n={len(neg_vels)})',
            color='#4e79a7', density=True)
    ax.axvline(pos_p25, color='#e15759', linestyle='--', alpha=0.7, label=f'Pos P25={pos_p25:.4f}')
    ax.set_xlabel('Mean Wrist Velocity (normalized units/frame)')
    ax.set_ylabel('Density')
    ax.set_title('Wrist Velocity: Positives vs Current Negatives')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'wrist_velocity_histogram.png'), dpi=150)
    plt.close()
    print("  Saved wrist_velocity_histogram.png")

    # 2. Activity scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for m in pos_metrics:
        ax.scatter(m['mean_wrist_vel'], m['mean_wrist_height'],
                   c='#e15759', alpha=0.3, s=10, zorder=2)
    colors = {'IDLE': '#4e79a7', 'ACTIVE': '#59a14f', 'AMBIGUOUS': '#edc948'}
    for m in neg_metrics:
        ax.scatter(m['mean_wrist_vel'], m['mean_wrist_height'],
                   c=colors[m['classification']], alpha=0.4, s=10, zorder=3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e15759', markersize=8, label='Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4e79a7', markersize=8, label='Neg: IDLE'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#59a14f', markersize=8, label='Neg: ACTIVE'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#edc948', markersize=8, label='Neg: AMBIGUOUS'),
    ]
    ax.legend(handles=handles)
    ax.set_xlabel('Mean Wrist Velocity')
    ax.set_ylabel('Mean Wrist Height (rel. to shoulders, negative = above)')
    ax.set_title('Hand Activity: Positives vs Negatives')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'activity_scatter.png'), dpi=150)
    plt.close()
    print("  Saved activity_scatter.png")

    # 3. Pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    labels_pie = list(counts.keys())
    sizes = [counts[k] for k in labels_pie]
    colors_pie = ['#4e79a7', '#59a14f', '#edc948']
    pcts = [f'{s/total:.1%}' for s in sizes]
    wedges, texts, autotexts = ax.pie(sizes, labels=[f'{l}\n({p})' for l, p in zip(labels_pie, pcts)],
                                        colors=colors_pie, autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 12})
    ax.set_title(f'Current Negative Composition (n={total})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'negative_composition_pie.png'), dpi=150)
    plt.close()
    print("  Saved negative_composition_pie.png")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"AUDIT COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
