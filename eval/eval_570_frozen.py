#!/usr/bin/env python3
"""
eval_570_frozen.py — freeze-and-forward variant of eval_570.py.

Difference from eval_570.py (which is left UNMODIFIED):
  * GT is read from test570_gt_frozen_20260723.json (a snapshot of the exact
    live eval query: 19 videos, t_end_s>t_start_s, 570 spans) instead of the
    live wife_drypass.db. GT no longer drifts with annotation edits.
  * Inference is NOT re-run. The cached 2026-07-19 probs
    (test570_probs_variant{3,4}.npz) are loaded as-is, so results are a pure
    function of (frozen GT, archived probs).
  * Output is written to test570_results_frozen.json (a NEW file). The archived
    test570_results.json is never touched, so it can be diffed byte-for-byte.

Operating points, IoU, matching and metric code are identical to eval_570.py
(imported from the same postprocessing module).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/postprocessing_optimisation')
from postprocessing_pipeline import PipelineConfig, run_pipeline, evaluate_segments

HND = '/home/gfe/bsl_project/hard_negative_detection'
GT_JSON = f'{HND}/test570_gt_frozen_20260723.json'
IOU = 0.3
V4_LOCK = PipelineConfig(smooth_kernel=5, threshold=0.6, gap_bridge_s=0.0, min_duration_s=0.3, conf_filter=0.0, boundary_extend_s=0.1)
V3_LOCK = PipelineConfig(smooth_kernel=3, threshold=0.8, gap_bridge_s=0.0, min_duration_s=0.5, conf_filter=0.0, boundary_extend_s=0.0)
B4 = PipelineConfig(smooth_kernel=7, threshold=0.5, gap_bridge_s=0.0, min_duration_s=0.2, conf_filter=0.5, boundary_extend_s=0.1)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def pooled(vids, probs, gt, cfg):
    agg = [0, 0, 0]
    for v in vids:
        segs = run_pipeline(probs[v], cfg)
        tp, fp, fn = evaluate_segments(segs, gt[v], IOU)[:3]
        agg[0] += tp; agg[1] += fp; agg[2] += fn
    return prf(*agg), agg


def main():
    # ── frozen GT ──
    G = json.load(open(GT_JSON))
    tR = G['R_videos']; tL = G['L_videos']; allv = tR + tL
    gt = {v: [{'start': s['start'], 'end': s['end']} for s in G['videos'][v]] for v in allv}

    # ── cached probs (NO inference) ──
    probs = {}
    for mk in ('variant3', 'variant4'):
        d = np.load(f'{HND}/test570_probs_{mk}.npz')
        probs[mk] = {v: d[v] for v in allv}

    R = {}
    def line(tag, mk, cfg):
        (f1, p, r), _ = pooled(allv, probs[mk], gt, cfg)
        (f1R, pR, rR), aR = pooled(tR, probs[mk], gt, cfg)
        (f1L, pL, rL), aL = pooled(tL, probs[mk], gt, cfg)
        R[tag] = {'all': (f1, p, r), 'R': (f1R, pR, rR), 'L': (f1L, pL, rL)}
        print(f"{tag:22s} overall F1={f1:.4f} P={p:.4f} R={r:.4f} | R F1={f1R:.4f} R={rR:.4f} | L F1={f1L:.4f} R={rL:.4f}")

    print("=" * 90)
    print("LINE 1 — best-vs-best")
    line('v4@V4_LOCK', 'variant4', V4_LOCK)
    line('v3@V3_LOCK', 'variant3', V3_LOCK)
    print("LINE 2 — continuity")
    line('v3@B4', 'variant3', B4)
    print("LINE 3 — fixed-config (B4 = v3's banked home turf)")
    line('v4@B4', 'variant4', B4)

    # ── LINE 4: duration-stratified recall (both at their lock) ──
    buckets = [('<0.5s', 0.0, 0.5), ('0.5-1.0s', 0.5, 1.0), ('>=1.0s', 1.0, 1e9)]
    print("\nLINE 4 — duration-stratified recall (GT span recalled = IoU-matched at 0.3)")
    dur_tab = {}
    for mk, cfg, lab in [('variant4', V4_LOCK, 'v4@lock'), ('variant3', V3_LOCK, 'v3@lock')]:
        rec = {b[0]: [0, 0] for b in buckets}
        for v in allv:
            segs = run_pipeline(probs[mk][v], cfg)
            _, _, _, matched, _, _, _, _ = evaluate_segments(segs, gt[v], IOU)
            matched_gt = {gj for _, gj in matched}
            for gi, g in enumerate(gt[v]):
                d = g['end'] - g['start']
                for name, lo, hi in buckets:
                    if lo <= d < hi:
                        rec[name][1] += 1
                        if gi in matched_gt: rec[name][0] += 1
                        break
        dur_tab[lab] = {k: (v[0], v[1], (v[0] / v[1] if v[1] else 0.0)) for k, v in rec.items()}
        print(f"  {lab}: " + "  ".join(f"{k} {v[0]}/{v[1]}={v[0]/v[1]*100 if v[1] else 0:.0f}%" for k, v in rec.items()))

    # ── new-only detections (v4@lock fires, no GT match, no overlapping v3@lock pred) ──
    print("\nNEW-ONLY detections (v4@lock fires, no GT match, no overlap with v3@lock pred):")
    newonly = {}
    tot_new = 0
    for v in allv:
        s4 = run_pipeline(probs['variant4'][v], V4_LOCK)
        s3 = run_pipeline(probs['variant3'][v], V3_LOCK)
        _, _, _, m4, unmatched4, _, _, _ = evaluate_segments(s4, gt[v], IOU)
        n = 0
        for pi in unmatched4:
            a, b = s4[pi]['start_time'], s4[pi]['end_time']
            if not any(min(b, s['end_time']) - max(a, s['start_time']) > 0 for s in s3):
                n += 1
        newonly[v] = n; tot_new += n
    print(f"  total new-only across 19 test videos: {tot_new}")
    top = sorted(newonly.items(), key=lambda kv: -kv[1])[:6]
    print("  top: " + ", ".join(f"{v}={n}" for v, n in top))

    json.dump({'lines': R, 'duration_recall': dur_tab, 'new_only': newonly, 'new_only_total': tot_new,
               'iou': IOU, 'R_videos': tR, 'L_videos': tL}, open(f'{HND}/test570_results_frozen.json', 'w'), indent=2)
    print(f"\nwrote test570_results_frozen.json (GT from frozen JSON, cached probs; archive untouched)")


if __name__ == '__main__':
    main()
