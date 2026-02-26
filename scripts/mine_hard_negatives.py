#!/usr/bin/env python3
"""
Step 2: Mine hard negatives from between-annotation gaps and subtitle-aligned periods.

Methods:
  A: Between-annotation gaps (2-30s, with 1s buffer) — confirmed active signing near FS
  B: Subtitle-aligned non-annotation periods — interpreter actively translating speech
  C: False positive mining — eval-only, from held-out episodes

Processes 244 non-held-out episodes with full landmarks, one at a time.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import webvtt
from pathlib import Path

# Import feature extraction
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts'))
from frame_features import extract_all_frame_features, TOTAL_FEATURE_DIM

# ── Paths ────────────────────────────────────────────────────────────────
LANDMARK_DIR = os.path.expanduser('~/bsl_project/full_episode_detection/landmarks/')
SUBTITLE_DIR = os.path.expanduser('~/bsl_project/data/subtitles/')
AUTO_ANN_PATH = os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/fingerspelling-automatic-annotations.csv')
FP_ANALYSIS_PATH = os.path.expanduser(
    '~/bsl_project/improved_detection/error_analysis/fp_analysis.csv')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/hard_negatives/')
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 25
WINDOW_SIZE = 25
STRIDE = 5
SEED = 42

HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}

# Activity threshold: reject windows with very low wrist velocity
MIN_WRIST_VEL = 0.005


def extract_windows(seq, window_size=WINDOW_SIZE, stride=STRIDE):
    """Extract sliding windows from a sequence. Returns (n_windows, window_size, D)."""
    N, D = seq.shape
    if N < window_size:
        return np.empty((0, window_size, D), dtype=np.float32)
    starts = list(range(0, N - window_size + 1, stride))
    return np.stack([seq[s:s + window_size] for s in starts]).astype(np.float32)


def compute_wrist_velocity(landmarks):
    """Compute mean wrist velocity from raw 1692-dim landmarks."""
    if len(landmarks) < 3:
        return 0.0
    l_wrist = landmarks[:, 60:63]
    r_wrist = landmarks[:, 64:67]
    l_vel = np.linalg.norm(np.diff(l_wrist, axis=0), axis=1).mean()
    r_vel = np.linalg.norm(np.diff(r_wrist, axis=0), axis=1).mean()
    return (l_vel + r_vel) / 2


def parse_subtitle_time(t_str):
    """Convert HH:MM:SS.mmm to seconds."""
    parts = t_str.split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def load_subtitles(ep_id):
    """Load subtitles for an episode. Returns list of (start_s, end_s, text)."""
    vtt_path = os.path.join(SUBTITLE_DIR, f'{ep_id}.vtt')
    if not os.path.exists(vtt_path):
        return []
    try:
        captions = webvtt.read(vtt_path)
    except Exception:
        return []

    segments = []
    for c in captions:
        start = parse_subtitle_time(c.start)
        end = parse_subtitle_time(c.end)
        text = c.text.strip()
        if text and end > start:
            segments.append((start, end, text))
    return segments


def merge_intervals(intervals):
    """Merge overlapping/adjacent (start, end) intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def find_valid_gaps(ann_times, min_gap=2.0, max_gap=30.0, buffer=1.0):
    """Find between-annotation gaps suitable for hard negatives.

    Args:
        ann_times: list of (start, end) sorted by start
        min_gap: minimum gap duration after buffer exclusion
        max_gap: maximum gap duration
        buffer: seconds to exclude on each side of annotations
    """
    if len(ann_times) < 2:
        return []

    gaps = []
    for i in range(len(ann_times) - 1):
        gap_start = ann_times[i][1] + buffer
        gap_end = ann_times[i + 1][0] - buffer
        gap_dur = gap_end - gap_start
        if min_gap <= gap_dur <= max_gap:
            gaps.append((gap_start, gap_end))
    return gaps


def find_subtitle_segments(subtitles, ann_times, ann_buffer=5.0):
    """Find subtitle-aligned segments NOT near any annotation.

    Returns intervals where subtitles exist but no annotation is within ann_buffer.
    """
    if not subtitles:
        return []

    # Build annotation exclusion zones
    exclusions = [(s - ann_buffer, e + ann_buffer) for s, e in ann_times]
    exclusions = merge_intervals(exclusions)

    # Merge subtitle intervals
    sub_intervals = merge_intervals([(s, e) for s, e, _ in subtitles])

    # Subtract exclusions from subtitle intervals
    result = []
    for ss, se in sub_intervals:
        remaining = [(ss, se)]
        for es, ee in exclusions:
            new_remaining = []
            for rs, re in remaining:
                if ee <= rs or es >= re:
                    new_remaining.append((rs, re))
                else:
                    if es > rs:
                        new_remaining.append((rs, min(es, re)))
                    if ee < re:
                        new_remaining.append((max(ee, rs), re))
            remaining = new_remaining
        result.extend(remaining)

    # Filter by minimum duration
    return [(s, e) for s, e in result if e - s >= WINDOW_SIZE / FPS]


def process_segments(landmarks, segments, method_label):
    """Extract feature windows from time segments.

    Args:
        landmarks: (N, 1692) full episode landmarks
        segments: list of (start_s, end_s)
        method_label: 'A' or 'B' for logging

    Returns:
        all_windows: np array (n_windows, WINDOW_SIZE, TOTAL_FEATURE_DIM)
        metadata: list of dicts
    """
    all_windows = []
    metadata = []
    n_rejected = 0

    for seg_start, seg_end in segments:
        frame_start = int(seg_start * FPS)
        frame_end = min(int(seg_end * FPS), len(landmarks))

        if frame_end - frame_start < WINDOW_SIZE:
            continue

        seg_lm = landmarks[frame_start:frame_end]

        # Activity check
        wrist_vel = compute_wrist_velocity(seg_lm)
        if wrist_vel < MIN_WRIST_VEL:
            n_rejected += 1
            continue

        # Compute features
        try:
            feats = extract_all_frame_features(seg_lm)
        except Exception:
            continue

        # Extract windows
        windows = extract_windows(feats)
        if len(windows) == 0:
            continue

        all_windows.append(windows)
        metadata.append({
            'start_time': float(seg_start),
            'end_time': float(seg_end),
            'duration': float(seg_end - seg_start),
            'mean_wrist_vel': float(wrist_vel),
            'n_windows': len(windows),
            'method': method_label,
        })

    if all_windows:
        all_windows = np.vstack(all_windows)
    else:
        all_windows = np.empty((0, WINDOW_SIZE, TOTAL_FEATURE_DIM), dtype=np.float32)

    return all_windows, metadata, n_rejected


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEP 2: MINE HARD NEGATIVES")
    print("=" * 70)

    # Load annotations
    auto_ann = pd.read_csv(AUTO_ANN_PATH)
    auto_ann['video_id'] = auto_ann['video_id'].astype(str)
    print(f"Loaded {len(auto_ann)} auto annotations")

    # Get eligible episodes (have full landmarks, not held-out)
    landmark_files = sorted(Path(LANDMARK_DIR).glob('*_mediapipe.npz'))
    all_ep_ids = [f.stem.replace('_mediapipe', '') for f in landmark_files]
    eligible = [e for e in all_ep_ids if e not in HELD_OUT]
    print(f"Eligible episodes: {len(eligible)} (excluding {len(HELD_OUT)} held-out)")

    # ══════════════════════════════════════════════════════════════════
    # METHOD A: Between-annotation mining
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("METHOD A: Between-annotation gaps")
    print(f"{'='*60}")

    all_a_windows = []
    all_a_meta = []
    total_a_rejected = 0

    for idx, ep_id in enumerate(eligible):
        ep_ann = auto_ann[auto_ann['video_id'] == ep_id].sort_values('start')
        ann_times = list(zip(ep_ann['start'].values, ep_ann['end'].values))

        gaps = find_valid_gaps(ann_times, min_gap=2.0, max_gap=30.0, buffer=1.0)
        if not gaps:
            if (idx + 1) % 50 == 0:
                n_w = sum(len(w) for w in all_a_windows)
                print(f"  [{idx+1}/{len(eligible)}] {n_w:,} windows, {len(all_a_meta)} segments")
            continue

        # Load landmarks
        lm_path = os.path.join(LANDMARK_DIR, f'{ep_id}_mediapipe.npz')
        data = np.load(lm_path)
        landmarks = data['landmarks']

        windows, meta, rejected = process_segments(landmarks, gaps, 'A')
        for m in meta:
            m['episode_id'] = ep_id
        all_a_windows.append(windows)
        all_a_meta.extend(meta)
        total_a_rejected += rejected

        del landmarks, data

        if (idx + 1) % 50 == 0:
            n_w = sum(len(w) for w in all_a_windows)
            print(f"  [{idx+1}/{len(eligible)}] {n_w:,} windows, {len(all_a_meta)} segments, "
                  f"{total_a_rejected} rejected")

    if all_a_windows:
        a_windows = np.vstack([w for w in all_a_windows if len(w) > 0])
    else:
        a_windows = np.empty((0, WINDOW_SIZE, TOTAL_FEATURE_DIM), dtype=np.float32)
    print(f"\n  Method A result: {len(a_windows):,} windows from {len(all_a_meta)} segments")
    print(f"  Rejected (low activity): {total_a_rejected}")

    # Save Method A
    np.savez_compressed(os.path.join(OUT_DIR, 'method_a_windows.npz'),
                        windows=a_windows)
    pd.DataFrame(all_a_meta).to_csv(os.path.join(OUT_DIR, 'method_a_metadata.csv'), index=False)

    # ══════════════════════════════════════════════════════════════════
    # METHOD B: Subtitle-aligned mining
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("METHOD B: Subtitle-aligned non-annotation periods")
    print(f"{'='*60}")

    all_b_windows = []
    all_b_meta = []
    total_b_rejected = 0
    eps_no_subs = 0

    for idx, ep_id in enumerate(eligible):
        subtitles = load_subtitles(ep_id)
        if not subtitles:
            eps_no_subs += 1
            continue

        ep_ann = auto_ann[auto_ann['video_id'] == ep_id].sort_values('start')
        ann_times = list(zip(ep_ann['start'].values, ep_ann['end'].values))

        sub_segments = find_subtitle_segments(subtitles, ann_times, ann_buffer=5.0)
        if not sub_segments:
            continue

        # Load landmarks
        lm_path = os.path.join(LANDMARK_DIR, f'{ep_id}_mediapipe.npz')
        data = np.load(lm_path)
        landmarks = data['landmarks']

        windows, meta, rejected = process_segments(landmarks, sub_segments, 'B')
        for m in meta:
            m['episode_id'] = ep_id
        all_b_windows.append(windows)
        all_b_meta.extend(meta)
        total_b_rejected += rejected

        del landmarks, data

        if (idx + 1) % 50 == 0:
            n_w = sum(len(w) for w in all_b_windows)
            print(f"  [{idx+1}/{len(eligible)}] {n_w:,} windows, {len(all_b_meta)} segments")

    if all_b_windows:
        b_windows = np.vstack([w for w in all_b_windows if len(w) > 0])
    else:
        b_windows = np.empty((0, WINDOW_SIZE, TOTAL_FEATURE_DIM), dtype=np.float32)
    print(f"\n  Method B result: {len(b_windows):,} windows from {len(all_b_meta)} segments")
    print(f"  Rejected (low activity): {total_b_rejected}")
    print(f"  Episodes without subtitles: {eps_no_subs}")

    # Save Method B
    np.savez_compressed(os.path.join(OUT_DIR, 'method_b_windows.npz'),
                        windows=b_windows)
    pd.DataFrame(all_b_meta).to_csv(os.path.join(OUT_DIR, 'method_b_metadata.csv'), index=False)

    # ══════════════════════════════════════════════════════════════════
    # METHOD C: FP mining (eval only)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("METHOD C: False positive mining (eval-only)")
    print(f"{'='*60}")

    fp_df = pd.read_csv(FP_ANALYSIS_PATH)
    fp_df['episode_id'] = fp_df['episode_id'].astype(str)
    # Only use FPs from held-out episodes
    fp_held = fp_df[fp_df['episode_id'].isin(HELD_OUT)]
    print(f"  FPs from held-out episodes: {len(fp_held)}")

    all_c_windows = []
    all_c_meta = []

    for ep_id in HELD_OUT:
        ep_fps = fp_held[fp_held['episode_id'] == ep_id]
        if len(ep_fps) == 0:
            continue

        lm_path = os.path.join(LANDMARK_DIR, f'{ep_id}_mediapipe.npz')
        if not os.path.exists(lm_path):
            continue

        data = np.load(lm_path)
        landmarks = data['landmarks']

        segments = list(zip(ep_fps['pred_start'].values, ep_fps['pred_end'].values))
        windows, meta, _ = process_segments(landmarks, segments, 'C')
        for m in meta:
            m['episode_id'] = ep_id
            m['eval_only'] = True
        all_c_windows.append(windows)
        all_c_meta.extend(meta)

        del landmarks, data

    if all_c_windows:
        c_windows = np.vstack([w for w in all_c_windows if len(w) > 0])
    else:
        c_windows = np.empty((0, WINDOW_SIZE, TOTAL_FEATURE_DIM), dtype=np.float32)
    print(f"  Method C result: {len(c_windows):,} windows from {len(all_c_meta)} segments")

    # Save Method C
    np.savez_compressed(os.path.join(OUT_DIR, 'method_c_windows.npz'),
                        windows=c_windows)
    pd.DataFrame(all_c_meta).to_csv(os.path.join(OUT_DIR, 'method_c_metadata.csv'), index=False)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    stats = {
        'method_a': {
            'n_windows': int(len(a_windows)),
            'n_segments': len(all_a_meta),
            'n_rejected': total_a_rejected,
            'mean_wrist_vel': float(np.mean([m['mean_wrist_vel'] for m in all_a_meta])) if all_a_meta else 0,
        },
        'method_b': {
            'n_windows': int(len(b_windows)),
            'n_segments': len(all_b_meta),
            'n_rejected': total_b_rejected,
            'eps_without_subs': eps_no_subs,
            'mean_wrist_vel': float(np.mean([m['mean_wrist_vel'] for m in all_b_meta])) if all_b_meta else 0,
        },
        'method_c': {
            'n_windows': int(len(c_windows)),
            'n_segments': len(all_c_meta),
            'eval_only': True,
        },
        'total_train_windows': int(len(a_windows) + len(b_windows)),
        'total_eval_windows': int(len(c_windows)),
    }

    with open(os.path.join(OUT_DIR, 'hard_negative_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MINING COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Method A: {len(a_windows):,} windows ({len(all_a_meta)} segments)")
    print(f"  Method B: {len(b_windows):,} windows ({len(all_b_meta)} segments)")
    print(f"  Method C: {len(c_windows):,} windows ({len(all_c_meta)} segments, eval-only)")
    print(f"  Total for training: {len(a_windows) + len(b_windows):,}")


if __name__ == '__main__':
    main()
