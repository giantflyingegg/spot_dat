#!/usr/bin/env python3
"""
Reusable acceptance evaluator for a retrained detector checkpoint.

CNN-only (PP, no GBT) OR full V3+GBT, under identity / flip / scale(3D-sim) /
speed transforms, on the 5 held-out test episodes. Reuses run_fair_eval.py
components (load_model, run_inference, detect_segments_optimised+PP_CONFIG,
apply_gbt_filter, evaluate_segments_iou) and transforms.py — nothing reimplemented.

Flip is purely spatial: GT event times are UNTOUCHED. Speed rescales GT by 1/v.
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.expanduser('~/bsl_project/evaluation/v3_gbt_fair_eval/'))
import transforms as T
import run_fair_eval as fe

FPS = fe.FPS


def load_episodes():
    auto = pd.read_csv(fe.AUTO_ANN_PATH); auto['video_id'] = auto['video_id'].astype(str)
    eps = {}
    for ep in fe.HELD_OUT_EPISODES:
        lm = np.load(str(fe.LANDMARKS_DIR / f'{ep}_mediapipe.npz'))['landmarks']
        gt = [{'start': r['start'], 'end': r['end']}
              for _, r in auto[auto['video_id'] == ep].iterrows()]
        eps[ep] = {'lm': lm, 'gt': gt}
    return eps


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r


def evaluate(model_bundle, eps, transform=None, gt_transform=None,
             gbt=None, iou=0.3):
    model, smean, sscale, ws, idim = model_bundle
    transform = transform or (lambda lm: lm)
    tp = fp = fn = 0
    per_ep = {}
    for ep, d in eps.items():
        lm_t = transform(d['lm'])
        probs = fe.run_inference(model, lm_t, smean, sscale, ws, idim, feat_type='rich')
        segs = fe.detect_segments_optimised(probs, fe.PP_CONFIG)
        if gbt is not None:
            segs, _ = fe.apply_gbt_filter(segs, lm_t, probs, len(lm_t) / FPS,
                                          gbt['model'], gbt['scaler'],
                                          gbt['feature_names'], gbt_threshold=0.5)
        gt = d['gt'] if gt_transform is None else gt_transform(d['gt'])
        etp, efp, efn, _ = fe.evaluate_segments_iou(segs, gt, iou)
        tp += etp; fp += efp; fn += efn
        per_ep[ep] = (etp, efp, efn)
    f1, p, r = _f1(tp, fp, fn)
    return {'f1': f1, 'p': p, 'r': r, 'tp': tp, 'fp': fp, 'fn': fn, 'per_ep': per_ep}


def rows_from(axis, value, label, res):
    out = [dict(model=label, axis=axis, value=value, scope='pooled',
                F1=res['f1'], P=res['p'], R=res['r'],
                tp=res['tp'], fp=res['fp'], fn=res['fn'])]
    for ep, (tp, fp, fn) in res['per_ep'].items():
        f1, p, r = _f1(tp, fp, fn)
        out.append(dict(model=label, axis=axis, value=value, scope=ep,
                        F1=f1, P=p, R=r, tp=tp, fp=fp, fn=fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gbt', default=None)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--scale', default='')   # e.g. "0.6,0.8,1.0,1.25,1.5"
    ap.add_argument('--speed', default='')   # e.g. "0.85,1.0,1.15"
    a = ap.parse_args()

    bundle = fe.load_model(os.path.expanduser(a.ckpt))
    gbt = None
    if a.gbt:
        gbt = pickle.load(open(os.path.expanduser(a.gbt), 'rb'))
    eps = load_episodes()
    print(f"[{a.label}] ckpt loaded ws={bundle[3]} dim={bundle[4]} gbt={'yes' if gbt else 'no'}")

    rows = []
    # identity
    r = evaluate(bundle, eps, gbt=gbt)
    rows += rows_from('identity', 1.0, a.label, r)
    print(f"  identity: F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f} "
          f"tp={r['tp']} fp={r['fp']} fn={r['fn']}")
    if a.flip:
        r = evaluate(bundle, eps, transform=lambda lm: T.hflip(lm), gbt=gbt)
        rows += rows_from('flip', 'flipped', a.label, r)
        print(f"  flipped : F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f} "
              f"tp={r['tp']} fp={r['fp']} fn={r['fn']}")
    for s in [float(x) for x in a.scale.split(',') if x]:
        r = evaluate(bundle, eps, transform=lambda lm, s=s: T.scale3d(lm, s), gbt=gbt)
        rows += rows_from('scale', s, a.label, r)
        print(f"  scale s={s}: F1={r['f1']:.4f}")
    for v in [float(x) for x in a.speed.split(',') if x]:
        r = evaluate(bundle, eps, transform=lambda lm, v=v: T.tresample(lm, v),
                     gt_transform=lambda gt, v=v: T.rescale_gt(gt, v), gbt=gbt)
        rows += rows_from('speed', v, a.label, r)
        print(f"  speed v={v}: F1={r['f1']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.expanduser(a.out), index=False)
    print(f"  saved {a.out}")


if __name__ == '__main__':
    main()
