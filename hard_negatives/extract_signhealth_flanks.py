#!/usr/bin/env python3
"""
Stage 2b — SignHealth hard-negative FLANK extraction.

Scope: done videos ONLY, minus the 19 frozen TEST videos (freeze JSONs = single
source of truth). Windowing matches the detector's training pipeline exactly:
25-frame (1.0s @ 25fps) windows, stride 5, 158-dim corrected features — the same
extract_all_frame_features space as unified_data/*.npz.

A window is a HARD-NEGATIVE flank iff ALL hold:
  R1  near signing : window within 5.0s of a span edge (nearest-span dist <= 5s)
  R2  not signing  : zero overlap with ANY span, with a 0.5s safety margin off
                     each span boundary (spans dilated by 0.5s must not overlap)
  R3  signing hand : hand landmarks present in >= 50% of the 25 frames
                     (>=1 hand block wrist != (0,0))
  R4  clean data   : no frame with a feats outlier (max|feat| > 100)  [pool hygiene]

Windows failing R3 (no hand) are logged as EASY-NEG candidates, not included.
Windows failing R1/R2 are not candidates. R4 failures are excluded + counted.

Outputs (NEW files):
  signhealth_flanks_v1.npz            windows=(M,25,158) float32
  signhealth_flanks_v1_manifest.json  per-window {video_id,t_start,t_end,
      hand_presence,nearest_span_dist,feature_run} + provenance header.
Read-only w.r.t. the DB.
"""
import os, sys, json, glob, sqlite3, hashlib
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
from frame_features import extract_all_frame_features  # noqa (only for dim ref)

DB = '/home/gfe/bsl_project/wife_drypass/wife_drypass.db'
OUT_DIR = '/home/gfe/bsl_project/hard_negative_detection'
SIG_OUT = f'{OUT_DIR}/output_signhealth'  # placeholder; real dir below
OUTPUT_SIGNHEALTH = '/home/gfe/bsl_project/youtube_mining/output_signhealth'
FREEZE_TEST_JSONS = (
    '/home/gfe/bsl_project/analysis/frozen_test_videos.json',
    '/home/gfe/bsl_project/analysis/frozen_test_L_supplement.json',
)
FPS = 25.0
WINDOW = 25            # frames (1.0s) — train_v3.WINDOW_SIZE
STRIDE = 5             # frames (0.2s) — pool NEG stride
NEAR_S = 5.0          # R1
MARGIN_S = 0.5        # R2 safety margin
HAND_FRAC = 0.5       # R3
OUTLIER = 100.0       # R4
LA, LB = 1566, 1629   # hand-block wrist offsets in the 1692-dim layout

OUT_NPZ = f'{OUT_DIR}/signhealth_flanks_v1.npz'
OUT_MAN = f'{OUT_DIR}/signhealth_flanks_v1_manifest.json'


def load_excluded():
    ids = set()
    for p in FREEZE_TEST_JSONS:
        doc = json.load(open(p))
        ids.update(v['video_id'] for v in doc['videos'])
    return ids


def load_spans(con, vid):
    rows = con.execute("SELECT t_start_s,t_end_s FROM spans WHERE video_id=? "
                       "AND t_end_s>t_start_s", (vid,)).fetchall()
    return sorted((float(a), float(b)) for a, b in rows)


def nearest_span_dist(w0, w1, spans):
    """Min gap from window [w0,w1] to any span interval; 0.0 if it overlaps one."""
    best = np.inf
    for a, b in spans:
        if w1 > a and w0 < b:
            return 0.0
        gap = a - w1 if a >= w1 else w0 - b
        if gap < best:
            best = gap
    return best


def overlaps_dilated(w0, w1, spans, m):
    for a, b in spans:
        if w1 > (a - m) and w0 < (b + m):
            return True
    return False


def feature_run(vid):
    return ('2026-07-19' if os.path.exists(
        f'{OUTPUT_SIGNHEALTH}/{vid}/{vid}_iso_deswap_provenance.json') else '2026-07-13')


def main():
    excluded = load_excluded()
    con = sqlite3.connect(DB)
    spans_dump = con.execute(
        "SELECT id,segment_idx,t_start_s,t_end_s,video_id FROM spans ORDER BY id").fetchall()
    spans_hash = hashlib.sha256(json.dumps(spans_dump, default=str).encode()).hexdigest()
    done = [r[0] for r in con.execute("SELECT video_id FROM video_status WHERE status='done'")]
    inscope = sorted(v for v in done if v not in excluded)
    print(f"done={len(done)}  excluded(test)={len(excluded)}  in-scope={len(inscope)}", flush=True)

    wins, man = [], []
    per_video = {}
    sub = {'no_feats': 0, 'r1_far': 0, 'r2_overlap': 0, 'r3_nohand_easyneg': 0, 'r4_outlier': 0}
    test_hits = 0

    for vid in inscope:
        d = f'{OUTPUT_SIGNHEALTH}/{vid}'
        fp = f'{d}/{vid}_feats158_25fps_iso_deswap.npy'
        lp = f'{d}/{vid}_landmarks_25fps_iso_deswap.npz'
        if not (os.path.exists(fp) and os.path.exists(lp)):
            sub['no_feats'] += 1
            continue
        feats = np.load(fp)                       # (N,158) raw pool-space
        lm = np.load(lp)['landmarks']             # (N,1692)
        n = min(len(feats), len(lm))
        spans = load_spans(con, vid)
        if not spans:
            continue
        # per-frame hand presence + feats outlier
        hand = ((lm[:, LA] != 0) | (lm[:, LA + 1] != 0) |
                (lm[:, LB] != 0) | (lm[:, LB + 1] != 0))
        fout = np.abs(feats).max(axis=1) > OUTLIER
        cnt = 0
        for s in range(0, n - WINDOW + 1, STRIDE):
            e = s + WINDOW
            w0, w1 = s / FPS, e / FPS
            nd = nearest_span_dist(w0, w1, spans)
            if nd > NEAR_S:            # R1
                sub['r1_far'] += 1; continue
            if overlaps_dilated(w0, w1, spans, MARGIN_S):   # R2
                sub['r2_overlap'] += 1; continue
            hf = float(hand[s:e].mean())
            if hf < HAND_FRAC:         # R3 -> easy-neg candidate, excluded
                sub['r3_nohand_easyneg'] += 1; continue
            if fout[s:e].any():        # R4
                sub['r4_outlier'] += 1; continue
            wins.append(feats[s:e])
            man.append({'video_id': vid, 't_start': round(w0, 3), 't_end': round(w1, 3),
                        'hand_presence': round(hf, 3), 'nearest_span_dist': round(float(nd), 3),
                        'feature_run': feature_run(vid)})
            cnt += 1
            if vid in excluded:
                test_hits += 1
        per_video[vid] = cnt

    con.close()
    W = np.asarray(wins, dtype=np.float32) if wins else np.empty((0, WINDOW, 158), np.float32)
    prov = {
        'artifact': 'signhealth_flanks_v1',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generator': 'extract_signhealth_flanks.py',
        'scope': 'done videos minus 19 frozen TEST videos',
        'source_db': DB, 'source_db_spans_sha256': spans_hash,
        'freeze_jsons_honoured': list(FREEZE_TEST_JSONS),
        'n_excluded_test_videos': len(excluded),
        'window_frames': WINDOW, 'stride_frames': STRIDE, 'fps': FPS, 'feat_dim': 158,
        'feature_space': 'extract_all_frame_features on 25fps+iso+deswap landmarks (matches unified_data pools)',
        'rules': {'R1_near_span_s': NEAR_S, 'R2_no_overlap_margin_s': MARGIN_S,
                  'R3_min_hand_presence_frac': HAND_FRAC, 'R4_max_abs_feat': OUTLIER},
        'n_windows': int(len(W)),
        'n_videos_contributing': int(sum(1 for c in per_video.values() if c > 0)),
        'subchecks_failed': sub,
        'test_windows_leaked': test_hits,
        'per_video_counts': per_video,
    }
    np.savez_compressed(OUT_NPZ, windows=W)
    json.dump({'provenance': prov, 'windows': man}, open(OUT_MAN, 'w'))

    # ── report ──
    print("=" * 64)
    print(f"windows: {len(W)}  shape={W.shape}")
    print(f"contributing videos: {prov['n_videos_contributing']}/{len(inscope)}")
    print(f"TEST windows leaked: {test_hits}  (MUST be 0)")
    tot_r3 = sub['r3_nohand_easyneg']
    cand = len(W) + tot_r3 + sub['r4_outlier']
    print(f"hand-presence pass rate: {len(W)}/{cand if cand else 1} "
          f"= {100*len(W)/(cand if cand else 1):.1f}% of R1&R2 survivors (R3+R4 applied)")
    print(f"sub-checks failed: {sub}")
    top = sorted(per_video.items(), key=lambda kv: -kv[1])[:10]
    print("top 10 videos by window count:")
    for v, c in top:
        flag = '   <-- 1F44 (letter-gap flanks)' if v == '1F44ac0vqac' else ''
        print(f"  {v:<14} {c:>6}{flag}")
    if '1F44ac0vqac' in per_video:
        share = 100 * per_video['1F44ac0vqac'] / (len(W) or 1)
        print(f"1F44 share of pool: {per_video['1F44ac0vqac']}/{len(W)} = {share:.1f}%")
    print(f"\nwrote {OUT_NPZ}\nwrote {OUT_MAN}")


if __name__ == '__main__':
    main()
