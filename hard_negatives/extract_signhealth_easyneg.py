#!/usr/bin/env python3
"""
Stage 3b-1: SignHealth EASY-NEGATIVE harvest — BOBSL convention.

HARD CONSTRAINT: done non-test videos ONLY. A non-done video's "far-from-span"
region may contain unannotated fingerspelling (nobody looked), so its negatives
would be poisoned — excluded entirely.

BOBSL easy-neg convention (extract_unified_dataset.py): intervals >5.0s from any
span, >=2.0s duration, mean wrist-velocity >= 0.005; windows at stride 5. Uses the
corrected landmarks (wrist velocity) + whole-video feats (windows). New files only.
"""
import os, sys, json, sqlite3
from datetime import datetime, timezone
import numpy as np

OUT = '/home/gfe/bsl_project/youtube_mining/output_signhealth'
DB = '/home/gfe/bsl_project/wife_drypass/wife_drypass.db'
OUT_DIR = '/home/gfe/bsl_project/hard_negative_detection'
FREEZE = ('/home/gfe/bsl_project/analysis/frozen_test_videos.json',
          '/home/gfe/bsl_project/analysis/frozen_test_L_supplement.json')
VAL_JSON = '/home/gfe/bsl_project/analysis/retrain_val_videos_PROPOSED.json'
FPS = 25; W = 25; NEG_STRIDE = 5
BUFFER = 5.0; MIN_DUR = 2.0; MIN_WRIST_VEL = 0.005
OUT_NPZ = f'{OUT_DIR}/signhealth_easyneg_v1.npz'
OUT_MAN = f'{OUT_DIR}/signhealth_easyneg_v1_manifest.json'


def easy_intervals(ann, dur):
    excl = sorted((max(0, a - BUFFER), min(dur, b + BUFFER)) for a, b in ann)
    merged = []
    for s, e in excl:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out = []; cur = 0.0
    for s, e in merged:
        if s - cur >= MIN_DUR:
            out.append((cur, s))
        cur = e
    if dur - cur >= MIN_DUR:
        out.append((cur, dur))
    return out


def wrist_vel(lm):
    l = lm[:, 60:63]; r = lm[:, 64:67]
    lv = np.linalg.norm(np.diff(l, axis=0), axis=1).mean() if len(lm) > 1 else 0
    rv = np.linalg.norm(np.diff(r, axis=0), axis=1).mean() if len(lm) > 1 else 0
    return (lv + rv) / 2


def main():
    test = set()
    for p in FREEZE:
        test |= {v['video_id'] for v in json.load(open(p))['videos']}
    val = {v['video_id'] for v in json.load(open(VAL_JSON))['val_videos']}
    con = sqlite3.connect(DB)
    done = [r[0] for r in con.execute("SELECT video_id FROM video_status WHERE status='done'") if r[0] not in test]

    wins, man = [], []
    per = {}; skip_lowvel = 0
    for vid in sorted(done):
        fp = f'{OUT}/{vid}/{vid}_feats158_25fps_iso_deswap.npy'
        lp = f'{OUT}/{vid}/{vid}_landmarks_25fps_iso_deswap.npz'
        if not (os.path.exists(fp) and os.path.exists(lp)):
            continue
        feats = np.load(fp); lm = np.load(lp)['landmarks']
        N = min(len(feats), len(lm)); dur = N / FPS
        fout = np.abs(feats).max(axis=1) > 100.0          # R4: bad-landmark frames (pool hygiene)
        split = 'val' if vid in val else 'train'
        spans = con.execute("SELECT t_start_s,t_end_s FROM spans WHERE video_id=? AND t_end_s>t_start_s", (vid,)).fetchall()
        cnt = 0
        for a, b in easy_intervals([(x, y) for x, y in spans], dur):
            f0 = int(a * FPS); f1 = min(int(b * FPS), N)
            if f1 - f0 < W:
                continue
            if wrist_vel(lm[f0:f1]) < MIN_WRIST_VEL:
                skip_lowvel += 1; continue
            for s in range(f0, f1 - W + 1, NEG_STRIDE):
                if fout[s:s + W].any():                    # reject window with an outlier frame
                    continue
                wins.append(feats[s:s + W])
                man.append({'video_id': vid, 't_start': round(s / FPS, 3), 'split': split})
                cnt += 1
        per[vid] = cnt
    con.close()
    Wn = np.asarray(wins, dtype=np.float32) if wins else np.empty((0, W, 158), np.float32)
    n_tr = sum(1 for m in man if m['split'] == 'train'); n_va = len(man) - n_tr
    prov = {'artifact': 'signhealth_easyneg_v1', 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'scope': 'done non-test videos ONLY (no easy-neg from non-done — poisoning risk)',
            'convention': f'BOBSL: >{BUFFER}s from span, >={MIN_DUR}s, wrist_vel>={MIN_WRIST_VEL}, stride {NEG_STRIDE}',
            'n_windows': int(len(Wn)), 'n_train': n_tr, 'n_val': n_va, 'skipped_lowvel_intervals': skip_lowvel}
    np.savez_compressed(OUT_NPZ, windows=Wn)
    json.dump({'provenance': prov, 'windows': man}, open(OUT_MAN, 'w'))
    print(f"easy-neg windows: {len(Wn)}  train={n_tr} val={n_va}  (from {sum(1 for c in per.values() if c) } done videos)")
    print(f"value stats: mean={Wn.mean():.3f} std={Wn.std():.3f} min={Wn.min():.3f} max={Wn.max():.3f}")
    print(f"wrote {OUT_NPZ}")


if __name__ == '__main__':
    main()
