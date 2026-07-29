#!/usr/bin/env python3
"""
Regenerate SignHealth detections on the CORRECTED pipeline (cache-reuse).

Per video:  raw cached landmarks  (output_signhealth/<vid>/<vid>_landmarks.npz)
  -> resample to 25fps
  -> isotropy correction (x,z *= W/H; W/H from ffprobe of the source video)
  -> per-frame geometric hand de-swap (blocks 1566/1629 by pose-wrist nearest)
  -> feats158 + variant3_flipaug.pt CNN
  -> post-processing at the RECALL operating point (B4): k7/th0.5/gap0/min0.2/cf0.5/ext0.1, GBT OFF
  -> write output_signhealth/<vid>/detections_v3_flipaug.json  (NEVER _gbt.json)

The transform chain (regen_transforms.py) was reverse-engineered and validated
bit-exact against the 53 known-good *_iso_deswap artifacts (the lost 2026-07-13 run).

Read-only w.r.t. every existing artifact: writes ONLY detections_v3_flipaug.json.
Does not touch detections_v3_gbt.json, checkpoints, landmark cache, or the DB.
"""
import sys, types
sys.modules.setdefault('webvtt', types.ModuleType('webvtt'))  # subtitle lib, unused here

import os, csv, json, glob, subprocess, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

BASE = Path('/home/gfe/bsl_project/youtube_mining')
sys.path.insert(0, str(BASE))
import run_signhealth_pipeline as P          # run_v3_cnn, run_pipeline, PipelineConfig, CNN1D, DEVICE
import regen_transforms as RT

OUTPUT_DIR   = BASE / 'output_signhealth'
KEEPSET_CSV  = BASE / 'signhealth_keepset.csv'
FLIP_CKPT    = Path('/home/gfe/bsl_project/robustness_diag/retrain/variant3_flipaug.pt')
OUT_NAME     = 'detections_v3_flipaug.json'

# Recall operating point (B4): flip-aug CNN + tuned PP, GBT OFF.
PP_B4 = P.PipelineConfig(smooth_kernel=7, threshold=0.5, gap_bridge_s=0.0,
                         min_duration_s=0.2, conf_filter=0.5, boundary_extend_s=0.1)


def load_keepset():
    m = {}
    with open(KEEPSET_CSV) as f:
        for r in csv.DictReader(f):
            m[r['video_id']] = r['local_path']
    return m


def source_wh(video_path):
    """W/H = float32(width/height) from ffprobe of the source video (coded dims)."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path],
        capture_output=True, text=True, timeout=60).stdout.strip()
    w, h = out.split('x')
    return int(w), int(h), np.float32(int(w) / int(h))


def load_flip_model():
    ck = torch.load(str(FLIP_CKPT), map_location='cpu', weights_only=False)
    model = P.CNN1D(input_dim=ck['input_dim']).to(P.DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, ck['scaler_mean'], ck['scaler_scale'], ck['window_size']


def process_video(vid, video_path, model, smean, sscale, ws):
    vdir = OUTPUT_DIR / vid
    raw = np.load(str(vdir / f'{vid}_landmarks.npz'))
    lm, fps = raw['landmarks'], float(raw['fps_actual'])
    if video_path and os.path.exists(video_path):
        w, h, wh = source_wh(video_path)
        wh_source, src_res = 'ffprobe', f'{w}x{h}'
    else:
        # source video deleted -> fall back to the dominant SignHealth aspect ratio.
        wh = np.float32(16.0 / 9.0)
        wh_source, src_res = 'assumed_16:9_no_source', 'unknown'

    lm25 = RT.resample_25fps(lm, fps)          # -> 25fps
    lm_iso = RT.iso_correct(lm25, wh)          # -> isotropic
    lm_dsw = RT.deswap(lm_iso)                  # -> hand blocks de-swapped
    n_frames = len(lm_dsw)
    duration_s = n_frames / 25.0

    raw_probs = P.run_v3_cnn(model, lm_dsw, smean, sscale, ws)   # flip-aug CNN
    segments = P.run_pipeline(raw_probs, PP_B4)                   # GBT OFF

    dets = [{
        'start_time': round(float(s['start_time']), 4),
        'end_time':   round(float(s['end_time']), 4),
        'duration':   round(float(s['end_time'] - s['start_time']), 4),
        'mean_prob':  round(float(s['mean_prob']), 6),
        'max_prob':   round(float(s['max_prob']), 6),
    } for s in segments]

    out = {
        # ── provenance header (do not repeat the lost-provenance mistake) ──
        'provenance': {
            'generator': 'regen_signhealth_flipaug.py',
            'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'checkpoint': FLIP_CKPT.name,                       # variant3_flipaug.pt
            'gbt': 'OFF',
            'operating_point': 'B4_recall',
            'pp_config': {'smooth_kernel': PP_B4.smooth_kernel, 'threshold': PP_B4.threshold,
                          'gap_bridge_s': PP_B4.gap_bridge_s, 'min_duration_s': PP_B4.min_duration_s,
                          'conf_filter': PP_B4.conf_filter, 'boundary_extend_s': PP_B4.boundary_extend_s},
            'transforms': {'resample_25fps': True, 'iso_correction': True,
                           'hand_deswap': True, 'W_over_H': float(wh),
                           'wh_source': wh_source, 'source_res': src_res, 'orig_fps': fps},
            'transform_module': 'regen_transforms.py',
        },
        'video_id': vid,
        'detector': 'v3_flipaug_deswap_iso',
        'n_frames': n_frames,
        'fps': 25.0,
        'duration_sec': round(duration_s, 2),
        'mean_frame_prob': float(raw_probs.mean()),
        'n_detections': len(dets),
        'detections_per_min': len(dets) / (duration_s / 60) if duration_s > 0 else 0,
        'detections': dets,
    }
    with open(vdir / OUT_NAME, 'w') as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='*', help='specific video_ids; default = all with a landmark cache')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    keep = load_keepset()
    if args.videos:
        vids = args.videos
    else:
        vids = sorted(os.path.basename(os.path.dirname(f))
                      for f in glob.glob(str(OUTPUT_DIR / '*' / '*_landmarks.npz')))
    if args.limit:
        vids = vids[:args.limit]

    model, smean, sscale, ws = load_flip_model()
    print(f"loaded {FLIP_CKPT.name}; DEVICE={P.DEVICE}; {len(vids)} videos", flush=True)
    ok = skip = 0
    for k, vid in enumerate(vids, 1):
        vp = keep.get(vid)
        o = process_video(vid, vp, model, smean, sscale, ws)   # vp None/missing -> 16:9 fallback
        ok += 1
        if o['provenance']['transforms']['wh_source'] != 'ffprobe':
            skip += 1  # counts videos that used the fallback W/H
        if k <= 3 or k % 100 == 0:
            print(f"  [{k}/{len(vids)}] {vid}: {o['n_detections']} dets, "
                  f"mean_frame_prob={o['mean_frame_prob']:.4f}", flush=True)
    print(f"done: {ok} written ({skip} used assumed-16:9 W/H fallback)", flush=True)


if __name__ == '__main__':
    main()
