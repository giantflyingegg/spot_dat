#!/usr/bin/env python3
"""
Stage 4b: v4 DEPLOYMENT detection run over all NON-TEST corpus videos.

Same corrected-feature pipeline as the v3 detection run (raw landmarks ->
resample25 -> iso -> deswap -> feats158 -> CNN), but with:
  * checkpoint = variant4_signhealth_ft.pt
  * PP = the deployment operating point (k3/th.3/gap0/min.2/cf0/ext0)
  * output = detections_v4_signhealth_ft.json  (NEVER overwrites v3's file)

The 19 frozen TEST videos are SKIPPED — their detection files are never written
(the builder lockout is the belt; this is the braces). Fast path: reuse cached
iso_deswap feats (.npy) where present; else compute lm_dsw on the fly.
Read-only w.r.t. DB, checkpoints, v3 detections, landmark cache.
"""
import os, sys, json, glob, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, '/home/gfe/bsl_project/youtube_mining')
sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/postprocessing_optimisation')
import regen_transforms as RT
import regen_signhealth_flipaug as R          # load_keepset, source_wh
from frame_features import extract_all_frame_features as EF
from postprocessing_pipeline import PipelineConfig, run_pipeline

OUT_DIR = '/home/gfe/bsl_project/youtube_mining/output_signhealth'
CKPT = '/home/gfe/bsl_project/robustness_diag/retrain/variant4_signhealth_ft.pt'
OUT_NAME = 'detections_v4_signhealth_ft.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WS = 25
DEPLOY = PipelineConfig(smooth_kernel=3, threshold=0.3, gap_bridge_s=0.0,
                        min_duration_s=0.2, conf_filter=0.0, boundary_extend_s=0.0)
DEPLOY_RATIONALE = ("min_dur floor aligned to the training span floor (0.2s = 5 frames); "
                    "min_dur is a structural exclusion not a soft knob, so a 0.3s floor would "
                    "re-blind the queue to sub-0.3s single letters (the class the containment "
                    "convention taught v4 to find). Recall-priority per FN-cost principle. "
                    "val: R=0.827 P=0.612.")


def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]


def main():
    test = set()
    for p in ('/home/gfe/bsl_project/analysis/frozen_test_videos.json',
              '/home/gfe/bsl_project/analysis/frozen_test_L_supplement.json'):
        test |= {v['video_id'] for v in json.load(open(p))['videos']}

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    from train_models import CNN1D
    model = CNN1D(input_dim=158).to(DEVICE); model.load_state_dict(ck['model_state_dict']); model.eval()
    smean = np.asarray(ck['scaler_mean'], np.float32); sscale = np.asarray(ck['scaler_scale'], np.float32)
    keep = R.load_keepset()
    ck_md5 = md5(CKPT)

    vids = sorted(os.path.basename(os.path.dirname(f))
                  for f in glob.glob(f'{OUT_DIR}/*/*_landmarks.npz'))
    vids = [v for v in vids if v not in test]
    print(f"v4 deploy detection over {len(vids)} non-test videos ({len(test)} test SKIPPED)", flush=True)

    def probs_from_feats(feats):
        fn = (feats - smean) / (sscale + 1e-8); n = len(fn)
        if n < WS: return np.zeros(n)
        win = np.moveaxis(sliding_window_view(fn, WS, axis=0), -1, 1); pr = []
        with torch.no_grad():
            for i in range(0, len(win), 4096):
                pr.append(torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(win[i:i + 4096])).float().to(DEVICE))).cpu().numpy())
        pr = np.concatenate(pr); ps = WS // 2; pe = n - len(pr) - ps
        return np.concatenate([np.zeros(ps), pr, np.zeros(max(0, pe))])

    ok = cached = computed = 0
    for k, vid in enumerate(vids, 1):
        d = f'{OUT_DIR}/{vid}'
        npy = f'{d}/{vid}_feats158_25fps_iso_deswap.npy'
        if os.path.exists(npy):
            feats = np.load(npy); n_frames = len(feats); src = 'cached_feats'
            wh_source = 'cached'; cached += 1
        else:
            raw = np.load(f'{d}/{vid}_landmarks.npz'); lm, fps = raw['landmarks'], float(raw['fps_actual'])
            vp = keep.get(vid)
            if vp and os.path.exists(vp):
                _, _, wh = R.source_wh(vp); wh_source = 'ffprobe'
            else:
                wh = np.float32(16.0 / 9.0); wh_source = 'assumed_16:9'
            feats = EF(RT.transform_chain(lm, fps, wh)).astype(np.float32); n_frames = len(feats); src = 'onthefly'; computed += 1
        probs = probs_from_feats(feats)
        segs = run_pipeline(probs, DEPLOY)
        dur = n_frames / 25.0
        out = {
            'provenance': {
                'generator': 'regen_signhealth_v4_deploy.py', 'purpose': 'DEPLOYMENT (not eval)',
                'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'checkpoint': 'variant4_signhealth_ft.pt', 'checkpoint_md5': ck_md5,
                'deployment_choice': DEPLOY.to_dict() if hasattr(DEPLOY, 'to_dict') else str(DEPLOY),
                'deployment_rationale': DEPLOY_RATIONALE,
                'feature_source': src, 'wh_source': wh_source,
                'note': 'v4 fine-tuned (BOBSL+SignHealth). Test videos never written (lockout braces).',
            },
            'video_id': vid, 'detector': 'v4_signhealth_ft', 'n_frames': n_frames, 'fps': 25.0,
            'duration_sec': round(dur, 2), 'mean_frame_prob': float(probs.mean()), 'n_detections': len(segs),
            'detections': [{'start_time': round(float(s['start_time']), 4), 'end_time': round(float(s['end_time']), 4),
                            'duration': round(float(s['end_time'] - s['start_time']), 4),
                            'mean_prob': round(float(s['mean_prob']), 6), 'max_prob': round(float(s['max_prob']), 6)}
                           for s in segs],
        }
        json.dump(out, open(f'{d}/{OUT_NAME}', 'w'), indent=2)
        ok += 1
        if k <= 3 or k % 150 == 0:
            print(f"  [{k}/{len(vids)}] {vid}: {len(segs)} dets ({src})", flush=True)
    print(f"done: {ok} written ({cached} from cached feats, {computed} on-the-fly). Test videos skipped.", flush=True)


if __name__ == '__main__':
    main()
