#!/usr/bin/env python3
"""
Extract ONLY easy negatives + split hard negatives.
Positives were already saved by the previous (OOM-killed) run.

This script skips positive extraction and only does:
1. Easy negative extraction from full-episode landmarks (subsampled to 250K train / 60K val)
2. Hard negative splitting by episode
3. Save dataset_stats.json
"""
import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts'))
from frame_features import extract_all_frame_features, TOTAL_FEATURE_DIM

# ── Paths ────────────────────────────────────────────────────────────────
LANDMARK_DIR = os.path.expanduser('~/bsl_project/full_episode_detection/landmarks/')
AUTO_ANN_PATH = os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/fingerspelling-automatic-annotations.csv')
HARD_NEG_PATH = os.path.expanduser(
    '~/bsl_project/hard_negative_detection/hard_negatives/method_a_windows.npz')
HARD_NEG_META_PATH = os.path.expanduser(
    '~/bsl_project/hard_negative_detection/hard_negatives/method_a_metadata.csv')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/unified_data/')
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 25
WINDOW_SIZE = 25
NEG_STRIDE = 5
SEED = 42
EASY_NEG_BUFFER = 5.0
EASY_NEG_MIN_DUR = 2.0
MIN_WRIST_VEL = 0.005

MAX_EASY_NEG_TRAIN = 250_000
MAX_EASY_NEG_VAL = 60_000

HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}

VAL_FRACTION = 0.2


def extract_windows(seq, window_size=WINDOW_SIZE, stride=NEG_STRIDE):
    N, D = seq.shape
    if N < window_size:
        return np.empty((0, window_size, D), dtype=np.float32)
    starts = list(range(0, N - window_size + 1, stride))
    return np.stack([seq[s:s + window_size] for s in starts]).astype(np.float32)


def compute_wrist_velocity(landmarks):
    if len(landmarks) < 3:
        return 0.0
    l_wrist = landmarks[:, 60:63]
    r_wrist = landmarks[:, 64:67]
    l_vel = np.linalg.norm(np.diff(l_wrist, axis=0), axis=1).mean()
    r_vel = np.linalg.norm(np.diff(r_wrist, axis=0), axis=1).mean()
    return (l_vel + r_vel) / 2


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def find_easy_negative_intervals(ann_times, episode_duration):
    exclusions = []
    for start, end in ann_times:
        excl_start = max(0, start - EASY_NEG_BUFFER)
        excl_end = min(episode_duration, end + EASY_NEG_BUFFER)
        exclusions.append((excl_start, excl_end))
    exclusions = merge_intervals(exclusions)

    valid = []
    prev_end = 0
    for excl_start, excl_end in exclusions:
        if excl_start - prev_end >= EASY_NEG_MIN_DUR:
            valid.append((prev_end, excl_start))
        prev_end = excl_end
    if episode_duration - prev_end >= EASY_NEG_MIN_DUR:
        valid.append((prev_end, episode_duration))
    return valid


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXTRACT EASY NEGATIVES + SPLIT HARD NEGATIVES")
    print("(Positives already extracted)")
    print("=" * 70)

    # Load annotations
    auto_ann = pd.read_csv(AUTO_ANN_PATH)
    auto_ann['video_id'] = auto_ann['video_id'].astype(str)

    # Find episodes with landmarks (excluding held-out)
    landmark_files = [f for f in os.listdir(LANDMARK_DIR) if f.endswith('_mediapipe.npz')]
    ep_ids_with_landmarks = set()
    for f in landmark_files:
        ep_id = f.replace('_mediapipe.npz', '')
        if ep_id not in HELD_OUT:
            ep_ids_with_landmarks.add(ep_id)

    ann_with_landmarks = auto_ann[auto_ann['video_id'].isin(ep_ids_with_landmarks)]
    n_eps = len(ann_with_landmarks['video_id'].unique())
    print(f"Episodes with landmarks (excl held-out): {n_eps}")

    # Reproduce the SAME episode-level split
    all_eps = sorted(ep_ids_with_landmarks)
    rng = np.random.RandomState(SEED)
    rng.shuffle(all_eps)
    split_idx = int(len(all_eps) * (1 - VAL_FRACTION))
    train_eps = set(all_eps[:split_idx])
    val_eps = set(all_eps[split_idx:])
    print(f"Episode split: {len(train_eps)} train, {len(val_eps)} val")

    # ── Extract easy negatives (episode by episode, track counts) ──
    easy_neg_train = []
    easy_neg_val = []
    n_easy_neg_segments = 0
    n_pos_ann_processed = 0
    n_pos_ann_skipped_short = 0

    # Count annotations for stats (don't re-extract positives)
    for _, row in ann_with_landmarks.iterrows():
        f_start = int(row['start'] * FPS)
        f_end = int(row['end'] * FPS)
        if f_end - f_start < WINDOW_SIZE:
            n_pos_ann_skipped_short += 1
        else:
            n_pos_ann_processed += 1

    print(f"Annotations: {n_pos_ann_processed} processed, {n_pos_ann_skipped_short} skipped")

    for i, ep_id in enumerate(sorted(ep_ids_with_landmarks)):
        lm_path = os.path.join(LANDMARK_DIR, f'{ep_id}_mediapipe.npz')
        data = np.load(lm_path)
        landmarks = data['landmarks']
        n_frames = len(landmarks)
        ep_dur = n_frames / FPS
        is_train = ep_id in train_eps

        ep_ann = ann_with_landmarks[ann_with_landmarks['video_id'] == ep_id]
        ann_times = [(row['start'], row['end']) for _, row in ep_ann.iterrows()]
        easy_intervals = find_easy_negative_intervals(ann_times, ep_dur)

        ep_easy_windows = []
        for seg_start, seg_end in easy_intervals:
            f_start = int(seg_start * FPS)
            f_end = min(int(seg_end * FPS), n_frames)
            if f_end - f_start < WINDOW_SIZE:
                continue

            seg_lm = landmarks[f_start:f_end]
            wrist_vel = compute_wrist_velocity(seg_lm)
            if wrist_vel < MIN_WRIST_VEL:
                continue

            feats = extract_all_frame_features(seg_lm)
            windows = extract_windows(feats, WINDOW_SIZE, NEG_STRIDE)
            if len(windows) > 0:
                ep_easy_windows.append(windows)
                n_easy_neg_segments += 1

        if ep_easy_windows:
            ep_easy = np.concatenate(ep_easy_windows)
            if is_train:
                easy_neg_train.append(ep_easy)
            else:
                easy_neg_val.append(ep_easy)

        del landmarks, data
        gc.collect()

        if (i + 1) % 25 == 0 or i == 0:
            n_neg_t = sum(len(w) for w in easy_neg_train)
            n_neg_v = sum(len(w) for w in easy_neg_val)
            print(f"  [{i+1}/{len(ep_ids_with_landmarks)}] "
                  f"easy_neg: {n_neg_t+n_neg_v:,} (t:{n_neg_t:,} v:{n_neg_v:,})")

    # ── Subsample and save easy negatives ──
    print("\nSaving easy negatives (subsampled to avoid OOM)...")
    rng2 = np.random.RandomState(SEED + 1)

    if easy_neg_train:
        total_easy_train = sum(len(w) for w in easy_neg_train)
        print(f"  Total easy_neg_train windows: {total_easy_train:,}")

        if total_easy_train > MAX_EASY_NEG_TRAIN:
            keep_ratio = MAX_EASY_NEG_TRAIN / total_easy_train
            subsampled = []
            for chunk in easy_neg_train:
                n_keep = max(1, int(len(chunk) * keep_ratio))
                idx = rng2.choice(len(chunk), size=n_keep, replace=False)
                subsampled.append(chunk[idx])
            del easy_neg_train
            gc.collect()
            ent = np.concatenate(subsampled)
            del subsampled
        else:
            ent = np.concatenate(easy_neg_train)
            del easy_neg_train

        gc.collect()
        np.savez_compressed(os.path.join(OUT_DIR, 'easy_neg_train.npz'), windows=ent)
        print(f"  easy_neg_train: {ent.shape}")
        del ent
        gc.collect()

    if easy_neg_val:
        total_easy_val = sum(len(w) for w in easy_neg_val)
        print(f"  Total easy_neg_val windows: {total_easy_val:,}")

        if total_easy_val > MAX_EASY_NEG_VAL:
            keep_ratio = MAX_EASY_NEG_VAL / total_easy_val
            subsampled = []
            for chunk in easy_neg_val:
                n_keep = max(1, int(len(chunk) * keep_ratio))
                idx = rng2.choice(len(chunk), size=n_keep, replace=False)
                subsampled.append(chunk[idx])
            del easy_neg_val
            gc.collect()
            env = np.concatenate(subsampled)
            del subsampled
        else:
            env = np.concatenate(easy_neg_val)
            del easy_neg_val

        gc.collect()
        np.savez_compressed(os.path.join(OUT_DIR, 'easy_neg_val.npz'), windows=env)
        print(f"  easy_neg_val: {env.shape}")
        del env
        gc.collect()

    # ── Split existing hard negatives by episode ──
    print("\nSplitting existing hard negatives by episode...")
    if os.path.exists(HARD_NEG_META_PATH) and os.path.exists(HARD_NEG_PATH):
        meta = pd.read_csv(HARD_NEG_META_PATH)
        hard_data = np.load(HARD_NEG_PATH)
        hard_windows = hard_data['windows']
        print(f"  Hard negatives loaded: {hard_windows.shape}")

        meta_cols = meta.columns.tolist()
        print(f"  Metadata columns: {meta_cols}")

        if 'episode_id' in meta_cols and 'n_windows' in meta_cols:
            meta['episode_id'] = meta['episode_id'].astype(str)
            ep_ids_per_window = []
            for _, row in meta.iterrows():
                ep_ids_per_window.extend([row['episode_id']] * int(row['n_windows']))

            if len(ep_ids_per_window) == len(hard_windows):
                ep_ids_arr = np.array(ep_ids_per_window)
                train_mask = np.isin(ep_ids_arr, list(train_eps))
                val_mask = np.isin(ep_ids_arr, list(val_eps))

                ht = hard_windows[train_mask]
                hv = hard_windows[val_mask]
                np.savez_compressed(os.path.join(OUT_DIR, 'hard_neg_train.npz'), windows=ht)
                np.savez_compressed(os.path.join(OUT_DIR, 'hard_neg_val.npz'), windows=hv)
                print(f"  hard_neg_train: {ht.shape}")
                print(f"  hard_neg_val: {hv.shape}")
                del ht, hv
            else:
                print(f"  WARNING: metadata window count ({len(ep_ids_per_window)}) != "
                      f"actual windows ({len(hard_windows)}). Using random split.")
                perm = np.random.RandomState(SEED).permutation(len(hard_windows))
                split = int(len(hard_windows) * (1 - VAL_FRACTION))
                np.savez_compressed(os.path.join(OUT_DIR, 'hard_neg_train.npz'),
                                    windows=hard_windows[perm[:split]])
                np.savez_compressed(os.path.join(OUT_DIR, 'hard_neg_val.npz'),
                                    windows=hard_windows[perm[split:]])
                print(f"  hard_neg_train: {split:,}, hard_neg_val: {len(hard_windows)-split:,}")
        else:
            print(f"  WARNING: Missing expected columns. Using random 80/20 split.")
            perm = np.random.RandomState(SEED).permutation(len(hard_windows))
            split = int(len(hard_windows) * (1 - VAL_FRACTION))
            np.savez_compressed(os.path.join(OUT_DIR, 'hard_neg_train.npz'),
                                windows=hard_windows[perm[:split]])
            np.savez_compressed(os.path.join(OUT_DIR, 'hard_neg_val.npz'),
                                windows=hard_windows[perm[split:]])

        del hard_windows, hard_data
        gc.collect()
    else:
        print("  WARNING: Hard negative files not found!")

    # Save split info and stats
    stats = {
        'n_pos_ann_processed': n_pos_ann_processed,
        'n_pos_ann_skipped_short': n_pos_ann_skipped_short,
        'n_easy_neg_segments': n_easy_neg_segments,
        'train_episodes': sorted(train_eps),
        'val_episodes': sorted(val_eps),
        'n_train_episodes': len(train_eps),
        'n_val_episodes': len(val_eps),
    }
    with open(os.path.join(OUT_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Easy negative segments: {n_easy_neg_segments}")


if __name__ == '__main__':
    main()
