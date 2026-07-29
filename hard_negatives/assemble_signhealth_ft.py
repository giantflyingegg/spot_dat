#!/usr/bin/env python3
"""
Stage 3b-2 assembly: build the SignHealth fine-tune train/val arrays with an
index-aligned FLIP counterpart (flip-aug), from the v1 window pools + manifests.

Convention: flip[i] = flip(orig[i]) EXACTLY. orig windows were sliced from
whole-video feats = extract_all_frame_features(lm_dsw); the flip is therefore
extract_all_frame_features(hflip(lm_dsw)) sliced at the SAME offset. Per-video
whole-video flip-feats are cached, then sliced for every window of that video.

Ratio (train): pos all : flank all(<=4x) : easy 2x (subsample seed 42).
Val: pos all : flank all : easy 2x. Val gets ORIG only (flip-aug is train-time).
Labels: pos=1, flank/easy=0. New files only; BOBSL pool untouched.
"""
import os, sys, json, hashlib
import numpy as np

sys.path.insert(0, '/home/gfe/bsl_project/youtube_mining')
sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
sys.path.insert(0, '/home/gfe/bsl_project/robustness_diag')
import regen_transforms as RT
import regen_signhealth_flipaug as R
from frame_features import extract_all_frame_features as EF
import transforms as T

OUT = '/home/gfe/bsl_project/youtube_mining/output_signhealth'
HND = '/home/gfe/bsl_project/hard_negative_detection'
VAL_JSON = '/home/gfe/bsl_project/analysis/retrain_val_videos_PROPOSED.json'
FPS = 25; W = 25; SEED = 42
KEEP = R.load_keepset()
VAL = {v['video_id'] for v in json.load(open(VAL_JSON))['val_videos']}
_flip_cache = {}


def row_split(m):
    """Split is a whole-video property (val = the 7 held-out videos). Robust across
    all pools — the flanks manifest (Stage 2b) predates the 'split' key."""
    return 'val' if m['video_id'] in VAL else 'train'


def lm_dsw(vid):
    p = f'{OUT}/{vid}/{vid}_landmarks_25fps_iso_deswap.npz'
    if os.path.exists(p):
        return np.load(p)['landmarks']
    raw = np.load(f'{OUT}/{vid}/{vid}_landmarks.npz'); lm, fps = raw['landmarks'], float(raw['fps_actual'])
    vp = KEEP.get(vid); wh = None
    if vp and os.path.exists(vp):
        _, _, wh = R.source_wh(vp)
    else:
        wh = np.float32(16.0 / 9.0)
    return RT.transform_chain(lm, fps, wh)


def flip_feats(vid):
    if vid not in _flip_cache:
        _flip_cache[vid] = EF(T.hflip(lm_dsw(vid))).astype(np.float32)
    return _flip_cache[vid]


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:12]


def build(pool, want_flip, subsample_to=None, split='train'):
    """Return (orig[K,25,158], flip[K,25,158] or None, rows) for the given pool/split."""
    W_all = np.load(f'{HND}/signhealth_{pool}_v1.npz')['windows']
    man = json.load(open(f'{HND}/signhealth_{pool}_v1_manifest.json'))['windows']
    idx = [i for i, m in enumerate(man) if row_split(m) == split]
    if subsample_to is not None and len(idx) > subsample_to:
        rng = np.random.RandomState(SEED)
        idx = sorted(rng.choice(idx, size=subsample_to, replace=False).tolist())
    orig = W_all[idx]
    rows = [man[i] for i in idx]
    flip = None
    if want_flip:
        flip = np.empty_like(orig)
        n_fallback = 0
        for k, m in enumerate(rows):
            s = int(round(m['t_start'] * FPS))
            ff = flip_feats(m['video_id'])
            seg = ff[s:s + W]
            if len(seg) < W:                       # video-end guard (rare)
                seg = np.vstack([seg, np.zeros((W - len(seg), 158), np.float32)])
            # hflip can introduce a feats outlier (~0.1% of windows) at a frame that
            # was clean in orig -> fall back to the orig window (that sample stays
            # un-flipped) so no 1e6 value poisons the pool.
            if np.abs(seg).max() > 100.0:
                seg = orig[k]; n_fallback += 1
            flip[k] = seg
        if n_fallback:
            print(f"    [{pool}] flip-outlier fallback to orig: {n_fallback} windows", flush=True)
    return orig, flip, rows


def main():
    print("assembling SignHealth FT arrays (orig + flip)...", flush=True)
    # sizes
    pos_tr_n = sum(1 for m in json.load(open(f'{HND}/signhealth_positives_v1_manifest.json'))['windows'] if row_split(m) == 'train')
    pos_va_n = sum(1 for m in json.load(open(f'{HND}/signhealth_positives_v1_manifest.json'))['windows'] if row_split(m) == 'val')
    # TRAIN
    po, pf, _ = build('positives', True, split='train')
    fo, ff, _ = build('flanks', True, split='train')                 # all
    eo, ef, _ = build('easyneg', True, subsample_to=2 * pos_tr_n, split='train')
    Xtr = np.concatenate([po, fo, eo]); Xtrf = np.concatenate([pf, ff, ef])
    ytr = np.concatenate([np.ones(len(po), np.float32), np.zeros(len(fo) + len(eo), np.float32)])
    # sanity: orig vs recomputed flip alignment (flip differs from orig, same shape)
    assert Xtr.shape == Xtrf.shape and not np.array_equal(po[0], pf[0])
    # VAL (orig only)
    pvo, _, _ = build('positives', False, split='val')
    fvo, _, _ = build('flanks', False, split='val')
    evo, _, _ = build('easyneg', False, subsample_to=2 * pos_va_n, split='val')
    Xva = np.concatenate([pvo, fvo, evo])
    yva = np.concatenate([np.ones(len(pvo), np.float32), np.zeros(len(fvo) + len(evo), np.float32)])

    np.savez_compressed(f'{HND}/sh_ft_train.npz', X=Xtr, Xf=Xtrf, y=ytr)
    np.savez_compressed(f'{HND}/sh_ft_val.npz', X=Xva, y=yva)
    prov = {
        'seed': SEED,
        'train': {'pos': int(len(po)), 'flank': int(len(fo)), 'easy': int(len(eo)), 'total': int(len(Xtr)),
                  'pos_frac': round(float(ytr.mean()), 4)},
        'val': {'pos': int(len(pvo)), 'flank': int(len(fvo)), 'easy': int(len(evo)), 'total': int(len(Xva)),
                'pos_frac': round(float(yva.mean()), 4)},
        'pool_hashes': {p: md5(f'{HND}/signhealth_{p}_v1.npz') for p in ('positives', 'flanks', 'easyneg')},
        'flip': 'extract_all_frame_features(transforms.hflip(lm_dsw)) sliced at same offset',
    }
    json.dump(prov, open(f'{HND}/sh_ft_assembly.json', 'w'), indent=2)
    print(f"TRAIN X={Xtr.shape} pos%={100*ytr.mean():.1f}  | VAL X={Xva.shape} pos%={100*yva.mean():.1f}")
    print(f"train: pos {len(po)} flank {len(fo)} easy {len(eo)} = {len(Xtr)}")
    print(f"val  : pos {len(pvo)} flank {len(fvo)} easy {len(evo)} = {len(Xva)}")
    print(f"orig/flip value check: orig mean={Xtr.mean():.3f} std={Xtr.std():.3f}  flip mean={Xtrf.mean():.3f} std={Xtrf.std():.3f}")
    print(f"wrote sh_ft_train.npz, sh_ft_val.npz, sh_ft_assembly.json")


if __name__ == '__main__':
    main()
