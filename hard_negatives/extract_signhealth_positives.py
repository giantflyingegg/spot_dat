#!/usr/bin/env python3
"""
Stage 3b-1: SignHealth POSITIVE-window extraction under the DUAL convention.

All non-test spanned videos with corrected iso_deswap features. Whole-video feats
(the .npy = extract_all_frame_features on corrected landmarks) are sliced — this
matches run_v3_cnn inference AND supplies real context frames for short spans.

Per span (id, t_start, t_end); f=int(t*25); L = f_end - f_start:
  L <  5 frames (0.2s)  -> DROP (below PP min_dur floor)
  5 <= L < 25 frames    -> CONTAINMENT-JITTER (SignHealth-side deviation):
        25-frame windows that fully CONTAIN the span, real frames only (never
        zero-pad), jittered at stride 3 across every offset where span ⊆ window.
        FS-frame fraction = L/25 (impure label — deliberate: a window containing
        a spell should fire; PP bounds the event afterwards).
  L >= 25 frames        -> STANDARD (BOBSL, unchanged): windows ⊆ span, stride 3,
        FS-frame fraction = 1.0.

The 7 val videos' windows go to split='val' (whole-video hold-out). BOBSL pool is
untouched. New files only.
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
FPS = 25; W = 25; STRIDE = 3; MIN_SPAN_F = 5
OUT_NPZ = f'{OUT_DIR}/signhealth_positives_v1.npz'
OUT_MAN = f'{OUT_DIR}/signhealth_positives_v1_manifest.json'


def main():
    test = set()
    for p in FREEZE:
        test |= {v['video_id'] for v in json.load(open(p))['videos']}
    val = {v['video_id'] for v in json.load(open(VAL_JSON))['val_videos']}

    con = sqlite3.connect(DB)
    spans_dump = con.execute("SELECT id,segment_idx,t_start_s,t_end_s,video_id FROM spans ORDER BY id").fetchall()
    import hashlib
    spans_hash = hashlib.sha256(json.dumps(spans_dump, default=str).encode()).hexdigest()
    vids = [r[0] for r in con.execute("SELECT DISTINCT video_id FROM spans WHERE video_id IS NOT NULL")]
    vids = sorted(v for v in vids if v not in test)

    wins, man = [], []
    by_conv = {'standard': 0, 'containment': 0}
    dropped_lt5 = 0
    per_video = {}
    per_video_val = {}
    for vid in vids:
        fp = f'{OUT}/{vid}/{vid}_feats158_25fps_iso_deswap.npy'
        if not os.path.exists(fp):
            continue
        feats = np.load(fp); N = len(feats)
        split = 'val' if vid in val else 'train'
        spans = con.execute("SELECT id,t_start_s,t_end_s FROM spans WHERE video_id=? "
                            "AND t_end_s>t_start_s ORDER BY t_start_s", (vid,)).fetchall()
        cnt = 0
        for sid, a, b in spans:
            f0 = int(a * FPS); f1 = min(int(b * FPS), N)
            L = f1 - f0
            if L < MIN_SPAN_F:
                dropped_lt5 += 1; continue
            if L < W:                                   # containment-jitter
                lo = max(0, f1 - W); hi = min(f0, N - W)
                conv = 'containment'; fs = round(L / W, 3)
            else:                                       # standard (window subset span)
                lo = f0; hi = f1 - W
                conv = 'standard'; fs = 1.0
            if hi < lo:
                continue
            for s in range(lo, hi + 1, STRIDE):
                wins.append(feats[s:s + W])
                man.append({'video_id': vid, 'span_id': sid, 'convention': conv,
                            'fs_frac': fs, 'split': split, 't_start': round(s / FPS, 3)})
                by_conv[conv] += 1; cnt += 1
        per_video[vid] = per_video.get(vid, 0) + cnt
        if split == 'val':
            per_video_val[vid] = cnt
    con.close()

    Wn = np.asarray(wins, dtype=np.float32) if wins else np.empty((0, W, 158), np.float32)
    n_train = sum(1 for m in man if m['split'] == 'train')
    n_val = sum(1 for m in man if m['split'] == 'val')
    prov = {
        'artifact': 'signhealth_positives_v1', 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generator': 'extract_signhealth_positives.py',
        'source_db_spans_sha256': spans_hash,
        'convention': 'DUAL: standard(window subset span,stride3,L>=25f) + containment-jitter(span subset window,stride3,5<=L<25f)',
        'feature_space': 'whole-video extract_all_frame_features (matches inference), sliced',
        'window_frames': W, 'stride_frames': STRIDE, 'min_span_frames': MIN_SPAN_F,
        'val_videos': sorted(val), 'n_windows': int(len(Wn)),
        'by_convention': by_conv, 'dropped_spans_lt5frames': dropped_lt5,
        'n_train': n_train, 'n_val': n_val, 'per_video_val': per_video_val,
    }
    np.savez_compressed(OUT_NPZ, windows=Wn)
    json.dump({'provenance': prov, 'windows': man}, open(OUT_MAN, 'w'))

    print("=" * 64)
    print(f"total positive windows: {len(Wn)}  shape={Wn.shape}")
    print(f"  by convention: standard={by_conv['standard']}  containment={by_conv['containment']}")
    print(f"  short-span (containment) contribution: {by_conv['containment']} "
          f"({100*by_conv['containment']/(len(Wn) or 1):.1f}% of positives)")
    print(f"  dropped spans <5 frames (0.2s floor): {dropped_lt5}")
    print(f"  train={n_train}  val={n_val}  (val = {100*n_val/(len(Wn) or 1):.1f}%)")
    print(f"1F44 positive windows: {per_video.get('1F44ac0vqac',0)}  (was 195 standard-only)")
    print("val per-video:", per_video_val)
    print(f"value stats: mean={Wn.mean():.3f} std={Wn.std():.3f} min={Wn.min():.3f} max={Wn.max():.3f} (pool ~0.12/0.59)")
    print(f"\nwrote {OUT_NPZ}\nwrote {OUT_MAN}")


if __name__ == '__main__':
    main()
