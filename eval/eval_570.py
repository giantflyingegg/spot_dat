#!/usr/bin/env python3
"""
Stage 3c — the ONE gated frozen-570 pass. Inference ONCE per model over the 19
frozen TEST videos (16 R + 3 L supplement); all operating points applied POST-HOC
to cached probs. Probs archived with provenance (paper artifact).

PRE-REGISTERED report lines (closed — nothing else is confirmatory):
  1. best-vs-best : v4@V4_LOCK vs v3@V3_LOCK        (R / L / overall)
  2. continuity   : v3@B4                            (v3's banked operating point)
  3. fixed-config : v3@B4 vs v4@B4                    (model-only; B4 = v3's home turf)
  4. duration-stratified recall (<0.5 / 0.5-1.0 / >=1.0s), both models at their lock
  + new-only detections per video (v4 fires, no GT match, no overlapping v3 pred)
Event matching: greedy IoU 0.3. THE 570 IS TOUCHED ONCE (this inference pass).
"""
import os, sys, json, sqlite3, hashlib
from datetime import datetime, timezone
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/postprocessing_optimisation')
from train_models import CNN1D
from postprocessing_pipeline import PipelineConfig, run_pipeline, evaluate_segments

DB = '/home/gfe/bsl_project/wife_drypass/wife_drypass.db'
OUT = '/home/gfe/bsl_project/youtube_mining/output_signhealth'
RET = '/home/gfe/bsl_project/robustness_diag/retrain'
HND = '/home/gfe/bsl_project/hard_negative_detection'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FPS = 25; WS = 25; IOU = 0.3
MODELS = {'variant3': f'{RET}/variant3_flipaug.pt', 'variant4': f'{RET}/variant4_signhealth_ft.pt'}
V4_LOCK = PipelineConfig(smooth_kernel=5, threshold=0.6, gap_bridge_s=0.0, min_duration_s=0.3, conf_filter=0.0, boundary_extend_s=0.1)
V3_LOCK = PipelineConfig(smooth_kernel=3, threshold=0.8, gap_bridge_s=0.0, min_duration_s=0.5, conf_filter=0.0, boundary_extend_s=0.0)
B4 = PipelineConfig(smooth_kernel=7, threshold=0.5, gap_bridge_s=0.0, min_duration_s=0.2, conf_filter=0.5, boundary_extend_s=0.1)


def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = CNN1D(input_dim=158).to(DEVICE); m.load_state_dict(ck['model_state_dict']); m.eval()
    return m, np.asarray(ck['scaler_mean'], np.float32), np.asarray(ck['scaler_scale'], np.float32)


def frame_probs(model, feats, smean, sscale):
    fn = (feats - smean) / (sscale + 1e-8); n = len(fn)
    if n < WS: return np.zeros(n)
    win = np.moveaxis(sliding_window_view(fn, WS, axis=0), -1, 1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(win), 4096):
            xb = torch.from_numpy(np.ascontiguousarray(win[i:i + 4096])).float().to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    preds = np.concatenate(preds); ps = WS // 2; pe = n - len(preds) - ps
    return np.concatenate([np.zeros(ps), preds, np.zeros(max(0, pe))])


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
    tR = [v['video_id'] for v in json.load(open('/home/gfe/bsl_project/analysis/frozen_test_videos.json'))['videos']]
    tL = [v['video_id'] for v in json.load(open('/home/gfe/bsl_project/analysis/frozen_test_L_supplement.json'))['videos']]
    allv = tR + tL
    con = sqlite3.connect(DB)
    gt = {v: [{'start': a, 'end': b} for a, b in
              con.execute("SELECT t_start_s,t_end_s FROM spans WHERE video_id=? AND t_end_s>t_start_s", (v,))]
          for v in allv}
    con.close()
    feat_run = {v: ('2026-07-19' if os.path.exists(f'{OUT}/{v}/{v}_iso_deswap_provenance.json') else '2026-07-13') for v in allv}

    # ── ONE inference pass per model; archive probs + provenance ──
    probs = {}
    for mk, mp in MODELS.items():
        model, sm, ss = load_model(mp)
        probs[mk] = {}
        for v in allv:
            probs[mk][v] = frame_probs(model, np.load(f'{OUT}/{v}/{v}_feats158_25fps_iso_deswap.npy'), sm, ss).astype(np.float32)
        np.savez_compressed(f'{HND}/test570_probs_{mk}.npz', **probs[mk])
    prov = {'artifact': 'frozen-570 cached inference probs', 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'iou': IOU, 'n_R': len(tR), 'n_L': len(tL),
            'checkpoints': {mk: {'path': mp, 'md5': md5(mp)} for mk, mp in MODELS.items()},
            'feature_run_per_video': feat_run,
            'operating_points': {'V4_LOCK': V4_LOCK.to_dict() if hasattr(V4_LOCK, 'to_dict') else str(V4_LOCK),
                                 'V3_LOCK': str(V3_LOCK), 'B4': str(B4)},
            'note': 'All reported operating points derive from this one archived inference pass.'}
    json.dump(prov, open(f'{HND}/test570_probs_provenance.json', 'w'), indent=2, default=str)

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
    # v3@B4 already computed as continuity

    # ── LINE 4: duration-stratified recall (both at their lock) ──
    buckets = [('<0.5s', 0.0, 0.5), ('0.5-1.0s', 0.5, 1.0), ('>=1.0s', 1.0, 1e9)]
    print("\nLINE 4 — duration-stratified recall (GT span recalled = IoU-matched at 0.3)")
    dur_tab = {}
    for mk, cfg, lab in [('variant4', V4_LOCK, 'v4@lock'), ('variant3', V3_LOCK, 'v3@lock')]:
        rec = {b[0]: [0, 0] for b in buckets}  # [recalled, total]
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
        for pi in unmatched4:                       # v4 preds with no GT match
            a, b = s4[pi]['start_time'], s4[pi]['end_time']
            if not any(min(b, s['end_time']) - max(a, s['start_time']) > 0 for s in s3):  # no v3 overlap
                n += 1
        newonly[v] = n; tot_new += n
    print(f"  total new-only across 19 test videos: {tot_new}")
    top = sorted(newonly.items(), key=lambda kv: -kv[1])[:6]
    print("  top: " + ", ".join(f"{v}={n}" for v, n in top))

    json.dump({'lines': R, 'duration_recall': dur_tab, 'new_only': newonly, 'new_only_total': tot_new,
               'iou': IOU, 'R_videos': tR, 'L_videos': tL}, open(f'{HND}/test570_results.json', 'w'), indent=2)
    print(f"\nwrote test570_results.json + cached probs + provenance")


if __name__ == '__main__':
    main()
