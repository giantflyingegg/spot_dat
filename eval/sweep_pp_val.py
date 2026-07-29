#!/usr/bin/env python3
"""
Stage 3c (val sweep): PP operating-point sweep on the 7 VAL videos ONLY, for
variant3_flipaug.pt AND variant4_signhealth_ft.pt. Locks ONE pooled config per
model (best pooled event-F1). R/L reported as DIAGNOSTICS (not separate op-points).
Also evaluates v3 at its banked B4_recall config as a continuity line.

Event matching: greedy IoU 0.3 (postprocessing_pipeline.evaluate_segments) — the
13-July convention. Inference = run_v3_cnn windowing from precomputed whole-video
iso_deswap feats (matches deployment). THE FROZEN 570 IS NOT TOUCHED.
"""
import os, sys, json, sqlite3, itertools
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/postprocessing_optimisation')
from train_models import CNN1D
from postprocessing_pipeline import PipelineConfig, run_pipeline, evaluate_segments

DB = '/home/gfe/bsl_project/wife_drypass/wife_drypass.db'
OUT = '/home/gfe/bsl_project/youtube_mining/output_signhealth'
VAL_JSON = '/home/gfe/bsl_project/analysis/retrain_val_videos_PROPOSED.json'
RET = '/home/gfe/bsl_project/robustness_diag/retrain'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FPS = 25; WS = 25; IOU = 0.3
MODELS = {'variant3': f'{RET}/variant3_flipaug.pt', 'variant4': f'{RET}/variant4_signhealth_ft.pt'}
B4 = PipelineConfig(smooth_kernel=7, threshold=0.5, gap_bridge_s=0.0,
                    min_duration_s=0.2, conf_filter=0.5, boundary_extend_s=0.1)
GRID = dict(smooth_kernel=[3, 5, 7, 9, 11], threshold=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            gap_bridge_s=[0.0, 0.25, 0.5], min_duration_s=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            conf_filter=[0.0, 0.5, 0.6], boundary_extend_s=[0.0, 0.1, 0.2])


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = CNN1D(input_dim=158).to(DEVICE); m.load_state_dict(ck['model_state_dict']); m.eval()
    return m, np.asarray(ck['scaler_mean'], np.float32), np.asarray(ck['scaler_scale'], np.float32)


def frame_probs(model, feats, smean, sscale):
    fn = (feats - smean) / (sscale + 1e-8)
    n = len(fn)
    if n < WS:
        return np.zeros(n)
    win = np.moveaxis(sliding_window_view(fn, WS, axis=0), -1, 1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(win), 4096):
            xb = torch.from_numpy(np.ascontiguousarray(win[i:i + 4096])).float().to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    preds = np.concatenate(preds)
    ps = WS // 2; pe = n - len(preds) - ps
    return np.concatenate([np.zeros(ps), preds, np.zeros(max(0, pe))])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    val = [v['video_id'] for v in json.load(open(VAL_JSON))['val_videos']]
    con = sqlite3.connect(DB)
    hand = {v: con.execute("SELECT signer_hand FROM video_status WHERE video_id=?", (v,)).fetchone()[0] for v in val}
    gt = {v: [{'start': a, 'end': b} for a, b in
              con.execute("SELECT t_start_s,t_end_s FROM spans WHERE video_id=? AND t_end_s>t_start_s", (v,))]
          for v in val}
    con.close()

    # cache per-frame probs per (model,video)
    probs = {}
    for mk, mp in MODELS.items():
        model, sm, ss = load_model(mp)
        for v in val:
            feats = np.load(f'{OUT}/{v}/{v}_feats158_25fps_iso_deswap.npy')
            probs[(mk, v)] = frame_probs(model, feats, sm, ss)
        print(f"[{mk}] probs cached for {len(val)} val videos", flush=True)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"sweeping {len(combos)} PP configs x {len(MODELS)} models...", flush=True)

    def score(mk, cfg):
        agg = {'all': [0, 0, 0], 'R': [0, 0, 0], 'L': [0, 0, 0]}
        for v in val:
            segs = run_pipeline(probs[(mk, v)], cfg)
            tp, fp, fn = evaluate_segments(segs, gt[v], IOU)[:3]
            for scope in ('all', hand[v]):
                agg[scope][0] += tp; agg[scope][1] += fp; agg[scope][2] += fn
        return agg

    results = {}
    all_scores = {}
    for mk in MODELS:
        best = None
        rows = []
        for combo in combos:
            cfg = PipelineConfig(**dict(zip(keys, combo)))
            agg = score(mk, cfg)
            f1 = prf(*agg['all'])[0]
            rows.append((f1, dict(zip(keys, combo)), prf(*agg['R'])[0], prf(*agg['L'])[0]))
            if best is None or f1 > best['f1']:
                best = {'f1': f1, 'cfg': dict(zip(keys, combo)), 'agg': agg}
        rows.sort(key=lambda r: -r[0])
        all_scores[mk] = rows
        results[mk] = best
        top = rows[0][0]
        within01 = sum(1 for r in rows if top - r[0] <= 0.01)
        within02 = sum(1 for r in rows if top - r[0] <= 0.02)
        print(f"\n[{mk}] top F1={top:.4f}; {within01} configs within 0.01, {within02} within 0.02 of best")
        print(f"  top-8 configs:")
        for f1, cfg, rf1, lf1 in rows[:8]:
            print(f"    F1={f1:.4f} R={rf1:.4f} L={lf1:.4f}  k{cfg['smooth_kernel']} th{cfg['threshold']} "
                  f"gap{cfg['gap_bridge_s']} min{cfg['min_duration_s']} cf{cfg['conf_filter']} ext{cfg['boundary_extend_s']}")
        # B4 continuity for v3
        b4 = score(mk, B4)
        results[mk]['b4'] = {'agg': b4}
        print(f"\n[{mk}] BEST pooled F1={best['f1']:.4f}  cfg={best['cfg']}")
        for scope in ('all', 'R', 'L'):
            f1, p, r = prf(*best['agg'][scope]); tp, fp, fn = best['agg'][scope]
            print(f"    {scope:>3}: F1={f1:.4f} P={p:.4f} R={r:.4f} (tp{tp} fp{fp} fn{fn})")
        f1b, pb, rb = prf(*b4['all'])
        print(f"    [B4 banked] pooled F1={f1b:.4f} P={pb:.4f} R={rb:.4f}")

    out = {'iou': IOU, 'val_videos': val, 'hand': hand, 'grid': GRID,
           'results': {mk: {'best_f1': r['f1'], 'best_cfg': r['cfg'],
                            'best_agg': {s: {'tp': r['agg'][s][0], 'fp': r['agg'][s][1], 'fn': r['agg'][s][2],
                                             'F1': prf(*r['agg'][s])[0], 'P': prf(*r['agg'][s])[1], 'R': prf(*r['agg'][s])[2]}
                                         for s in ('all', 'R', 'L')},
                            'b4_all': {'tp': r['b4']['agg']['all'][0], 'fp': r['b4']['agg']['all'][1],
                                       'fn': r['b4']['agg']['all'][2], 'F1': prf(*r['b4']['agg']['all'])[0]}}
                       for mk, r in results.items()}}
    json.dump(out, open('/home/gfe/bsl_project/hard_negative_detection/pp_sweep_val_results.json', 'w'), indent=2)
    print("\nwrote pp_sweep_val_results.json  (570 NOT touched)")


if __name__ == '__main__':
    main()
