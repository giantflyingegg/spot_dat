#!/usr/bin/env python3
"""
Stage 2 — refit the V3 GBT over BOTH orientations (original + flipped) of the
flip-aug CNN (variant3_flipaug.pt), so the second-stage filter is handedness-robust.

Faithful to the banked recipe (full_detection_v3/gbt_filtering/scripts/apply_gbt_filter.py):
  - 15 alphabetical GBT_FEATURE_NAMES, labels = greedy IoU>=0.3 to auto-GT.
  - StandardScaler + GradientBoostingClassifier(n_estimators=200, max_depth=4,
    learning_rate=0.1, subsample=0.8, random_state=42).
  - GroupKFold(5) by episode for the CV number.
  - Train = 244 non-held-out episodes (same split the banked GBT used); the 5
    held-out episodes stay OUT (they are the Stage-3 test).

The ONLY changes vs banked:
  * features come from variant3_flipaug.pt (not banked CNN), and
  * each training episode contributes TWICE — original landmarks AND T.hflip(landmarks)
    — so the GBT sees both handedness orientations. GT event times are flip-invariant
    (flip is spatial), so labels are computed against the SAME GT for both orientations.
    GroupKFold groups on episode_id so an episode's orig+flip stay in the same fold.

Features are extracted through the EXACT Stage-3 application path (run_fair_eval's
run_inference -> detect_segments_optimised(PP_CONFIG) -> extract_gbt_segment_features),
so the GBT is trained on the same feature distribution it will filter at test time.

banked gbt_v3.pkl is NOT touched; output is gbt_v3_flipaug.pkl.
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))                                   # transforms.py
sys.path.insert(0, os.path.expanduser('~/bsl_project/evaluation/v3_gbt_fair_eval/'))
sys.path.insert(0, os.path.expanduser('~/bsl_project/full_detection_v3/gbt_filtering/scripts/'))

import transforms as T
import run_fair_eval as fe
from extract_segment_features import label_detections           # banked greedy IoU>=0.3 labeller

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SEED = 42
FPS = fe.FPS
CKPT = os.path.join(HERE, 'variant3_flipaug.pt')
DET_DIR = os.path.expanduser('~/bsl_project/full_detection_v3/detections/')
OUT_PKL = os.path.join(HERE, 'gbt_v3_flipaug.pkl')
OUT_FEATS = os.path.join('..', 'results', 'retrain', 'stage2_segment_features_flipaug.csv')
HELD_OUT = set(fe.HELD_OUT_EPISODES)
FEATS = fe.GBT_FEATURE_NAMES


def train_episodes():
    eps = sorted(f.replace('_detections.json', '')
                 for f in os.listdir(DET_DIR) if f.endswith('_detections.json'))
    return [e for e in eps if e not in HELD_OUT]                  # 244 non-held-out


def segs_and_labels(bundle, lm_t, gt_list):
    model, smean, sscale, ws, idim = bundle
    probs = fe.run_inference(model, lm_t, smean, sscale, ws, idim, feat_type='rich')
    segs = fe.detect_segments_optimised(probs, fe.PP_CONFIG)
    if not segs:
        return []
    labels, best_ious = label_detections(segs, gt_list)
    dur = len(lm_t) / FPS
    rows = []
    for k, seg in enumerate(segs):
        feats = fe.extract_gbt_segment_features(seg, lm_t, probs, segs, dur)
        if feats is None:
            continue
        feats['label'] = int(labels[k])
        feats['best_iou'] = float(best_ious[k])
        rows.append(feats)
    return rows


def build_features():
    bundle = fe.load_model(CKPT)
    print(f"flip-aug CNN loaded: ws={bundle[3]} dim={bundle[4]}")
    auto = pd.read_csv(fe.AUTO_ANN_PATH); auto['video_id'] = auto['video_id'].astype(str)
    eps = train_episodes()
    print(f"training episodes (non-held-out): {len(eps)}")
    all_rows = []
    t0 = time.time()
    for i, ep in enumerate(eps):
        lm = np.load(str(fe.LANDMARKS_DIR / f'{ep}_mediapipe.npz'))['landmarks']
        gt_list = [{'start': r['start'], 'end': r['end']}
                   for _, r in auto[auto['video_id'] == ep].iterrows()]
        for orient, lm_t in (('orig', lm), ('flip', T.hflip(lm))):
            rows = segs_and_labels(bundle, lm_t, gt_list)
            for r in rows:
                r['episode_id'] = ep
                r['orientation'] = orient
            all_rows.extend(rows)
        del lm
        if (i + 1) % 25 == 0 or i + 1 == len(eps):
            el = time.time() - t0
            eta = (len(eps) - i - 1) / ((i + 1) / el)
            print(f"  [{i+1}/{len(eps)}] {ep}: total rows={len(all_rows)} "
                  f"[{el/60:.1f}min elapsed, ETA {eta/60:.1f}min]")
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_FEATS), exist_ok=True)
    df.to_csv(OUT_FEATS, index=False)
    print(f"saved {OUT_FEATS}: {len(df)} segments "
          f"(orig {int((df.orientation=='orig').sum())}, flip {int((df.orientation=='flip').sum())})")
    return df


def train_gbt(df):
    X = df[FEATS].values
    y = df['label'].values
    groups = df['episode_id'].values            # orig+flip of an episode share a group
    print(f"\nTrain matrix: {X.shape}  TP={int(y.sum())} FP={int((1-y).sum())}")
    for orient in ('orig', 'flip'):
        m = (df.orientation == orient).values
        print(f"  {orient}: {int(m.sum())} segs, TP={int(y[m].sum())} FP={int((1-y[m]).sum())}")

    # 5-fold episode-grouped CV (reporting)
    gkf = GroupKFold(n_splits=5)
    cv_preds = np.zeros(len(y))
    for fi, (tr, va) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                         learning_rate=0.1, subsample=0.8, random_state=SEED)
        clf.fit(Xtr, y[tr])
        cv_preds[va] = clf.predict_proba(Xva)[:, 1]
        yp = (cv_preds[va] >= 0.5).astype(int)
        print(f"  fold {fi}: F1={f1_score(y[va], yp):.3f} "
              f"P={precision_score(y[va], yp):.3f} R={recall_score(y[va], yp):.3f}")
    cvb = (cv_preds >= 0.5).astype(int)
    cv_f1 = f1_score(y, cvb); cv_auc = roc_auc_score(y, cv_preds)
    print(f"  Overall CV F1={cv_f1:.3f} AUROC={cv_auc:.3f}")

    # final fit on all train (both orientations)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    gbt = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.1, subsample=0.8, random_state=SEED)
    gbt.fit(Xs, y)
    imp = sorted(zip(FEATS, gbt.feature_importances_), key=lambda z: -z[1])
    print("\nFeature importances:")
    for f, v in imp:
        print(f"  {f:<25s} {v:.4f}")

    with open(OUT_PKL, 'wb') as fp:
        pickle.dump({'model': gbt, 'scaler': scaler, 'feature_names': FEATS,
                     'model_type': 'gradient_boosting', 'cv_f1': float(cv_f1),
                     'cv_auroc': float(cv_auc),
                     'note': 'flip-aug: orig+flipped training segments from variant3_flipaug.pt'}, fp)
    print(f"\nsaved {OUT_PKL}  (CV F1={cv_f1:.3f} AUROC={cv_auc:.3f})")
    json.dump({'cv_f1': float(cv_f1), 'cv_auroc': float(cv_auc),
               'n_train': int(len(y)), 'importances': dict(imp)},
              open(os.path.join('..', 'results', 'retrain', 'stage2_gbt_flipaug_summary.json'), 'w'),
              indent=2)


if __name__ == '__main__':
    df = build_features()
    train_gbt(df)
