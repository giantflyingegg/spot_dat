#!/usr/bin/env python3
"""
Flip-augmentation data builder for the V3 retrain.

Produces, for every banked training window (positives / easy-neg / hard-neg,
train+val), an INDEX-ALIGNED horizontally-flipped counterpart in feature space,
by flipping the segment's RAW landmarks (transforms.hflip) and re-running the
UNMODIFIED frame_features.py — never transforming the 158-dim vector directly.

Fidelity choices (so flip[i] differs from banked orig[i] by the flip ALONE):
  * Per-SEGMENT feature extraction, exactly as the banked extractors do
    (extract_unified_dataset.py / mine_hard_negatives.py). The original recipe
    is per-segment, so a per-segment flip is the faithful, confound-free pairing
    — NOT a whole-episode re-window, which would give the flip branch a different
    Group-D temporal-boundary treatment than the banked orig branch and let the
    net cheat. (Reflection axis is per-frame either way; within-segment Group-D
    context is exactly what the banked model was trained on.)
  * Segment KEEP/SKIP decisions (short-segment skip, wrist-velocity filter, the
    easy-neg subsample RNG) are taken from the ORIGINAL landmarks — identical set
    of segments as banked; only the features differ.

Reuses the banked helpers (no reimplementation): extract_unified_dataset.py for
positives/easy-neg, the hard-neg metadata for hard-neg provenance.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))          # transforms.py
sys.path.insert(0, os.path.expanduser('~/bsl_project/hard_negative_detection/scripts'))
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts'))

import transforms as T
import extract_unified_dataset as exu   # reuse: helpers + constants + split logic
from frame_features import extract_all_frame_features as EF

FLIP_DIR = os.path.join(HERE, 'flipped_data')
os.makedirs(FLIP_DIR, exist_ok=True)
UNIFIED = os.path.expanduser('~/bsl_project/hard_negative_detection/unified_data/')
HARD_META = os.path.expanduser(
    '~/bsl_project/hard_negative_detection/hard_negatives/method_a_metadata.csv')
LM_DIR = exu.LANDMARK_DIR
FPS, W = exu.FPS, exu.WINDOW_SIZE


def _episode_split():
    """Reproduce extract_unified_dataset's episode set + train/val split."""
    files = [f for f in os.listdir(LM_DIR) if f.endswith('_mediapipe.npz')]
    eps = set(f.replace('_mediapipe.npz', '') for f in files) - exu.HELD_OUT
    auto = pd.read_csv(exu.AUTO_ANN_PATH); auto['video_id'] = auto['video_id'].astype(str)
    ann = auto[auto['video_id'].isin(eps)]
    all_eps = sorted(eps)
    rng = np.random.RandomState(exu.SEED); rng.shuffle(all_eps)
    split = int(len(all_eps) * (1 - exu.VAL_FRACTION))
    return sorted(eps), set(all_eps[:split]), set(all_eps[split:]), ann


def _flip_feats(seg_lm):
    """Per-segment flipped features (the core op): hflip raw -> frame_features."""
    return EF(T.hflip(seg_lm))


def _orig_feats(seg_lm):
    return EF(seg_lm)


# ── positives + easy negatives (one pass over sorted episodes) ──────────────

def build_pos_easy(flip=True, verify_n=0, max_eps=None):
    eps_sorted, train_eps, val_eps, ann = _episode_split()
    feat = _flip_feats if flip else _orig_feats
    pos_tr, pos_va, easy_tr, easy_va = [], [], [], []
    vchecks = []
    for i, ep in enumerate(eps_sorted):
        if max_eps is not None and i >= max_eps:
            break
        lm = np.load(os.path.join(LM_DIR, f'{ep}_mediapipe.npz'))['landmarks']
        n = len(lm); is_tr = ep in train_eps
        ea = ann[ann['video_id'] == ep]
        # positives
        ep_pos = []
        for _, row in ea.iterrows():
            fs = int(row['start'] * FPS); fe = min(int(row['end'] * FPS), n)
            if fe - fs < W:
                continue
            win = exu.extract_windows(feat(lm[fs:fe]), W, exu.POS_STRIDE)
            if len(win):
                ep_pos.append(win)
        if ep_pos:
            (pos_tr if is_tr else pos_va).append(np.concatenate(ep_pos))
        # easy negatives (keep/skip from ORIG; same intervals + wrist-vel filter)
        ann_times = [(r['start'], r['end']) for _, r in ea.iterrows()]
        ep_easy = []
        for s, e in exu.find_easy_negative_intervals(ann_times, n / FPS):
            fs = int(s * FPS); fe = min(int(e * FPS), n)
            if fe - fs < W:
                continue
            seg = lm[fs:fe]
            if exu.compute_wrist_velocity(seg) < exu.MIN_WRIST_VEL:
                continue
            win = exu.extract_windows(feat(seg), W, exu.NEG_STRIDE)
            if len(win):
                ep_easy.append(win)
        if ep_easy:
            (easy_tr if is_tr else easy_va).append(np.concatenate(ep_easy))
        if verify_n and i < verify_n and ep_pos and is_tr:
            vchecks.append((ep, np.concatenate(ep_pos)))
        del lm; gc.collect()
        if (i + 1) % 50 == 0:
            print(f"  pos/easy {i+1}/{len(eps_sorted)}")
    return pos_tr, pos_va, easy_tr, easy_va, vchecks


def _subsample_like_banked(chunks, max_n, seed):
    """Reproduce extract_unified_dataset's per-chunk keep-ratio subsample."""
    total = sum(len(c) for c in chunks)
    rng2 = np.random.RandomState(seed)
    if total <= max_n:
        return np.concatenate(chunks)
    keep = max_n / total
    out = []
    for c in chunks:
        nk = max(1, int(len(c) * keep))
        idx = rng2.choice(len(c), size=nk, replace=False)
        out.append(c[idx])
    return np.concatenate(out)


# ── hard negatives (metadata-CSV order, episode-cached) ─────────────────────

def build_hard(flip=True, verify_rows=0):
    _, train_eps, val_eps, _ = _episode_split()
    meta = pd.read_csv(HARD_META); meta['episode_id'] = meta['episode_id'].astype(str)
    feat = _flip_feats if flip else _orig_feats
    all_win, ep_per_win, adj = [], [], []
    cur_ep, cur_lm = None, None
    vcheck = []
    for ri, row in meta.iterrows():
        if verify_rows and ri >= verify_rows:
            break
        ep = row['episode_id']
        if ep != cur_ep:
            del cur_lm; gc.collect()
            cur_lm = np.load(os.path.join(LM_DIR, f'{ep}_mediapipe.npz'))['landmarks']
            cur_ep = ep
        fs = int(row['start_time'] * FPS); fe = min(int(row['end_time'] * FPS), len(cur_lm))
        win = exu.extract_windows(feat(cur_lm[fs:fe]), W, exu.NEG_STRIDE)
        nw = int(row['n_windows'])
        # The banked unified hard pool (method_a_windows.npz) is FROZEN and stored exactly
        # `nw` windows/row; train_v3 loads it as the orig branch, so the flip pool MUST keep
        # per-row counts == nw to stay globally index-aligned (htf[i] == flip(hard_tr[i])).
        # Diagnostic over all 11594 rows: orig and flip extracts agree everywhere (0
        # mismatches), and exactly ONE row (6620, ep 6164207930460576679) now extracts 130
        # vs banked 131 — that episode's landmarks drifted by one window since the banked
        # build. Pad (repeat last) / truncate to nw to preserve alignment + the V3-subset
        # RNG length. A padded slot is a benign near-duplicate flipped NEGATIVE (label
        # unchanged); verify_rows path is left exact (asserts) to keep the orig==banked check.
        if verify_rows:
            assert len(win) == nw, f"row {ri} ep {ep}: {len(win)} != metadata {nw}"
        elif len(win) != nw:
            adj.append((ri, ep, len(win), nw))
            if len(win) > nw:
                win = win[:nw]
            elif len(win) > 0:
                win = np.concatenate([win, np.repeat(win[-1:], nw - len(win), axis=0)])
            else:
                raise RuntimeError(f"row {ri} ep {ep}: 0 windows, cannot pad to {nw}")
        all_win.append(win)
        ep_per_win.extend([ep] * len(win))
        if verify_rows and ri < verify_rows:
            vcheck.append((ri, win))
        if (ri + 1) % 2000 == 0:
            print(f"  hard {ri+1}/{len(meta)}")
    if adj:
        print(f"  ALIGN-PAD: adjusted {len(adj)} row(s) to banked counts: {adj}")
    allw = np.vstack(all_win)
    ep_arr = np.array(ep_per_win)
    tr = allw[np.isin(ep_arr, list(train_eps))]
    va = allw[np.isin(ep_arr, list(val_eps))]
    return tr, va, vcheck


# ── verification: regenerate ORIG samples, compare to banked ────────────────

def verify():
    print("=" * 60 + "\nVERIFY: regenerated ORIG == banked unified_data\n" + "=" * 60)
    # positives: first few episodes (no subsampling -> exact alignment)
    pos_tr, _, _, _, vchecks = build_pos_easy(flip=False, verify_n=2, max_eps=2)
    banked_pos = np.load(UNIFIED + 'positives_train.npz')['windows']
    regen = np.concatenate([w for _, w in vchecks])
    head = banked_pos[:len(regen)]
    ok_pos = np.allclose(regen, head, atol=1e-5)
    print(f"  positives_train: regen {regen.shape} vs banked head {head.shape}  "
          f"max|Δ|={np.max(np.abs(regen-head)):.2e}  ALIGNED={ok_pos}")
    # hard negatives: first 50 metadata rows compared to the PRE-SPLIT window file
    # (method_a_windows.npz), which is the exact order build_hard concatenates in
    # before applying the identical train/val isin mask. (Comparing to the
    # train-filtered hard_neg_train is wrong: rows 0..49 may span a val episode.)
    _, _, vcheck = build_hard(flip=False, verify_rows=50)
    maw = np.load(os.path.expanduser('~/bsl_project/hard_negative_detection/'
                                     'hard_negatives/method_a_windows.npz'))['windows']
    regen_h = np.concatenate([w for _, w in vcheck])
    head_h = maw[:len(regen_h)]
    ok_hard = np.allclose(regen_h, head_h, atol=1e-5)
    print(f"  hard (vs method_a_windows): regen {regen_h.shape} vs head {head_h.shape}  "
          f"max|Δ|={np.max(np.abs(regen_h-head_h)):.2e}  ALIGNED={ok_hard}")
    print(f"\n  VERIFY {'PASS' if (ok_pos and ok_hard) else 'FAIL'}")
    return ok_pos and ok_hard


def build_all():
    print("=" * 60 + "\nBUILD FLIPPED POOLS (index-aligned to banked)\n" + "=" * 60)
    print("[1/2] positives + easy negatives (flipped)...")
    pos_tr, pos_va, easy_tr, easy_va, _ = build_pos_easy(flip=True)
    np.savez_compressed(f'{FLIP_DIR}/positives_train_flip.npz',
                        windows=np.concatenate(pos_tr))
    np.savez_compressed(f'{FLIP_DIR}/positives_val_flip.npz',
                        windows=np.concatenate(pos_va))
    et = _subsample_like_banked(easy_tr, 250_000, exu.SEED + 1)
    ev = _subsample_like_banked(easy_va, 60_000, exu.SEED + 1)
    np.savez_compressed(f'{FLIP_DIR}/easy_neg_train_flip.npz', windows=et)
    np.savez_compressed(f'{FLIP_DIR}/easy_neg_val_flip.npz', windows=ev)
    print(f"  pos_train_flip {sum(len(w) for w in pos_tr)}, "
          f"easy_train_flip {len(et)}")
    del pos_tr, pos_va, easy_tr, easy_va, et, ev; gc.collect()
    print("[2/2] hard negatives (flipped)...")
    htr, hva, _ = build_hard(flip=True)
    np.savez_compressed(f'{FLIP_DIR}/hard_neg_train_flip.npz', windows=htr)
    np.savez_compressed(f'{FLIP_DIR}/hard_neg_val_flip.npz', windows=hva)
    print(f"  hard_train_flip {htr.shape}, hard_val_flip {hva.shape}")
    # shape cross-check vs banked
    for nm in ['positives_train', 'positives_val', 'easy_neg_train', 'easy_neg_val',
               'hard_neg_train', 'hard_neg_val']:
        b = np.load(UNIFIED + nm + '.npz')['windows'].shape
        f = np.load(f'{FLIP_DIR}/{nm}_flip.npz')['windows'].shape
        print(f"  {nm}: banked {b}  flip {f}  {'OK' if b == f else 'MISMATCH'}")
    print("BUILD COMPLETE")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--build', action='store_true')
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.build:
        build_all()
