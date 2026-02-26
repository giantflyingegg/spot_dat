#!/usr/bin/env python3
"""
Apply GBT filtering to V3 detections: (A) existing V2 GBT, (B) retrained V3 GBT.

Reads segment_features.csv, trains/applies GBT models, evaluates on held-out episodes,
saves filtered detection JSONs, comparison table, figures, and report.

Usage:
    python3 -u apply_gbt_filter.py
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report
)
from sklearn.model_selection import GroupKFold

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(os.path.expanduser('~/bsl_project/full_detection_v3/'))
GBT_DIR = BASE / 'gbt_filtering'
V3_DET_DIR = BASE / 'detections'
V2_GBT_PATH = Path(os.path.expanduser(
    '~/bsl_project/improved_detection/duration_aware/models/duration_filter.pkl'))
TEST_ANN_PATH = Path(os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/transpeller-test.csv'))
AUTO_ANN_PATH = Path(os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/'
    'fingerspelling-automatic-annotations.csv'))

# Previous detection dirs for reference comparison
V2_GBT_DET_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_detection_v2/detections_gbt/'))

FPS = 25
IOU_THRESHOLD = 0.3
SEED = 42

HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}

# The 15 features expected by GBT (sorted alphabetically, matching V2 GBT)
GBT_FEATURE_NAMES = [
    'conf_ramp', 'confidence_std', 'duration', 'hand_detection_rate',
    'isolation_ratio', 'l_finger_spread', 'l_fingertip_vel_mean',
    'max_confidence', 'mean_confidence', 'mean_finger_spread',
    'mean_mouth_opening', 'mean_wrist_height', 'mean_wrist_velocity',
    'r_finger_spread', 'r_fingertip_vel_mean',
]


# ══════════════════════════════════════════════════════════════════════════
# IoU MATCHING FOR EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0


def compute_overlap(ps, pe, gs, ge):
    """Overlap fraction = intersection / detection_duration."""
    inter = max(0, min(pe, ge) - max(ps, gs))
    det_dur = pe - ps
    return inter / det_dur if det_dur > 0 else 0


def eval_recall(detections_by_ep, ann_df, threshold=0.5, use_overlap=False):
    """Compute recall: fraction of GT annotations matched by detections.
    Only considers episodes present in both detections and annotations.
    use_overlap=True: score = overlap/det_dur (for test recall)
    use_overlap=False: score = IoU (for auto recall)
    """
    score_fn = compute_overlap if use_overlap else compute_iou
    total_gt = 0
    total_matched = 0
    common_eps = set(str(e) for e in ann_df['video_id'].unique()) & set(detections_by_ep.keys())

    for ep_id in common_eps:
        ep_ann = ann_df[ann_df['video_id'] == ep_id]
        gt_list = [(r['start'], r['end']) for _, r in ep_ann.iterrows()]
        dets = detections_by_ep.get(ep_id, [])

        total_gt += len(gt_list)

        # Build score matrix
        n_det = len(dets)
        n_gt = len(gt_list)
        if n_det == 0 or n_gt == 0:
            continue

        score_mat = np.zeros((n_det, n_gt))
        for i, d in enumerate(dets):
            for j, (gs, ge) in enumerate(gt_list):
                score_mat[i, j] = score_fn(d['start_time'], d['end_time'], gs, ge)

        # Greedy matching by highest score
        matched_d, matched_g = set(), set()
        while True:
            mask = np.ones_like(score_mat, dtype=bool)
            for i in matched_d: mask[i, :] = False
            for j in matched_g: mask[:, j] = False
            masked = score_mat * mask
            if masked.max() < threshold:
                break
            i, j = np.unravel_index(masked.argmax(), masked.shape)
            matched_d.add(i)
            matched_g.add(j)

        total_matched += len(matched_d)

    return total_matched, total_gt


def eval_precision_proxy(detections_by_ep, ann_df, iou_threshold=0.3):
    """Precision proxy: fraction of detections matching any GT (IoU >= threshold)."""
    total_det = 0
    total_matched = 0

    for ep_id, dets in detections_by_ep.items():
        if not dets:
            continue
        ep_ann = ann_df[ann_df['video_id'] == ep_id]
        gt_list = [(r['start'], r['end']) for _, r in ep_ann.iterrows()]
        total_det += len(dets)

        for d in dets:
            matched = False
            for gs, ge in gt_list:
                if compute_iou(d['start_time'], d['end_time'], gs, ge) >= iou_threshold:
                    matched = True
                    break
            if matched:
                total_matched += 1

    return total_matched, total_det


def count_new_discoveries(detections_by_ep, ann_df, max_iou=0.1):
    """Count detections that don't match any GT annotation."""
    count = 0
    for ep_id, dets in detections_by_ep.items():
        ep_ann = ann_df[ann_df['video_id'] == ep_id]
        gt_list = [(r['start'], r['end']) for _, r in ep_ann.iterrows()]
        for d in dets:
            if not gt_list:
                count += 1
                continue
            best_iou = max(compute_iou(d['start_time'], d['end_time'], gs, ge)
                          for gs, ge in gt_list)
            if best_iou < max_iou:
                count += 1
    return count


def compute_total_duration_hours(detections_by_ep):
    """Total duration of all episodes with detections."""
    # Use V3 detection JSONs for episode durations
    total_s = 0
    for ep_id in detections_by_ep:
        det_path = V3_DET_DIR / f'{ep_id}_detections.json'
        if det_path.exists():
            with open(det_path) as f:
                d = json.load(f)
            total_s += d['duration_sec']
    return total_s / 3600


# ══════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def evaluate_pipeline(name, detections_by_ep, test_ann, auto_ann, total_hours):
    """Full evaluation of a detection pipeline."""
    total_dets = sum(len(v) for v in detections_by_ep.values())

    test_matched, test_total = eval_recall(
        detections_by_ep, test_ann, threshold=0.5, use_overlap=True)
    auto_matched, auto_total = eval_recall(
        detections_by_ep, auto_ann, threshold=0.3, use_overlap=False)
    prec_matched, prec_total = eval_precision_proxy(detections_by_ep, auto_ann, iou_threshold=0.3)
    new_disc = count_new_discoveries(detections_by_ep, auto_ann)

    # Mean duration and confidence
    all_dets = [d for dets in detections_by_ep.values() for d in dets]
    mean_dur = np.mean([d['duration'] for d in all_dets]) if all_dets else 0
    mean_conf = np.mean([d.get('frame_confidence', d.get('mean_confidence', 0))
                         for d in all_dets]) if all_dets else 0

    result = {
        'name': name,
        'total_detections': total_dets,
        'det_per_hour': total_dets / total_hours if total_hours > 0 else 0,
        'test_recall': test_matched / test_total if test_total > 0 else 0,
        'test_matched': test_matched,
        'test_total': test_total,
        'auto_recall': auto_matched / auto_total if auto_total > 0 else 0,
        'auto_matched': auto_matched,
        'auto_total': auto_total,
        'precision_proxy': prec_matched / prec_total if prec_total > 0 else 0,
        'prec_matched': prec_matched,
        'new_discoveries': new_disc,
        'mean_duration': mean_dur,
        'mean_confidence': mean_conf,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def plot_comparison_bars(results_list, fig_dir):
    """Bar chart comparing all pipelines."""
    names = [r['name'] for r in results_list]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Total detections
    vals = [r['total_detections'] for r in results_list]
    colors = ['#4e79a7', '#e15759', '#59a14f', '#f28e2b']
    axes[0].bar(x, vals, color=colors[:len(names)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    axes[0].set_title('Total Detections')
    for i, v in enumerate(vals):
        axes[0].text(i, v + max(vals)*0.01, f'{v:,}', ha='center', fontsize=8)

    # Recall
    test_r = [r['test_recall'] for r in results_list]
    auto_r = [r['auto_recall'] for r in results_list]
    w = 0.35
    axes[1].bar(x - w/2, test_r, w, label='Test recall', color='#4e79a7')
    axes[1].bar(x + w/2, auto_r, w, label='Auto recall', color='#e15759', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    axes[1].set_title('Recall')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    for i in range(len(names)):
        axes[1].text(i - w/2, test_r[i] + 0.01, f'{test_r[i]:.1%}', ha='center', fontsize=7)
        axes[1].text(i + w/2, auto_r[i] + 0.01, f'{auto_r[i]:.1%}', ha='center', fontsize=7)

    # Precision proxy
    prec = [r['precision_proxy'] for r in results_list]
    axes[2].bar(x, prec, color=colors[:len(names)])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    axes[2].set_title('Precision Proxy')
    axes[2].set_ylim(0, 1.05)
    for i, v in enumerate(prec):
        axes[2].text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=8)

    # Detection density
    dens = [r['det_per_hour'] for r in results_list]
    axes[3].bar(x, dens, color=colors[:len(names)])
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    axes[3].set_title('Detections/hour')
    for i, v in enumerate(dens):
        axes[3].text(i, v + max(dens)*0.01, f'{v:.1f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / 'comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comparison_bars.png")


def plot_feature_importances(importances, feature_names, fig_dir):
    """Feature importance bar chart."""
    sorted_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_names)),
            importances[sorted_idx[::-1]],
            color='#59a14f')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title('V3 GBT Feature Importances')
    plt.tight_layout()
    plt.savefig(fig_dir / 'feature_importances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_importances.png")


def plot_confidence_distributions(df, kept_v2, kept_v3, fig_dir):
    """Confidence distribution of kept vs rejected for both GBTs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, kept, title in [
        (axes[0], kept_v2, 'V2 GBT Filter'),
        (axes[1], kept_v3, 'V3 GBT Filter'),
    ]:
        mask_kept = df.index.isin(kept)
        ax.hist(df.loc[mask_kept, 'mean_confidence'], bins=50, alpha=0.6,
                label=f'Kept ({mask_kept.sum():,})', color='#59a14f', density=True)
        ax.hist(df.loc[~mask_kept, 'mean_confidence'], bins=50, alpha=0.6,
                label=f'Rejected ({(~mask_kept).sum():,})', color='#e15759', density=True)
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'confidence_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved confidence_distributions.png")


def plot_recall_precision_tradeoff(df_train, gbt_model, scaler, feature_names, fig_dir):
    """Show recall vs precision at different GBT thresholds."""
    X = df_train[feature_names].values
    y = df_train['label'].values
    X_scaled = scaler.transform(X)
    probs = gbt_model.predict_proba(X_scaled)[:, 1]

    thresholds = np.arange(0.1, 0.95, 0.05)
    recalls, precisions = [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            recalls.append(0)
            precisions.append(1)
        else:
            recalls.append(recall_score(y, preds))
            precisions.append(precision_score(y, preds))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, recalls, 'o-', label='Recall', color='#4e79a7')
    ax.plot(thresholds, precisions, 's-', label='Precision', color='#e15759')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default threshold')
    ax.set_xlabel('GBT Threshold')
    ax.set_ylabel('Score')
    ax.set_title('V3 GBT: Recall vs Precision at Different Thresholds')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'recall_precision_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved recall_precision_tradeoff.png")


# ══════════════════════════════════════════════════════════════════════════
# SAVE FILTERED DETECTIONS
# ══════════════════════════════════════════════════════════════════════════

def save_filtered_detections(df_filtered, out_dir, variant_name):
    """Save filtered detections as episode JSONs (same format as V3 unfiltered)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for ep_id in df_filtered['episode_id'].unique():
        # Load original V3 detection JSON as template
        orig_path = V3_DET_DIR / f'{ep_id}_detections.json'
        with open(orig_path) as f:
            orig = json.load(f)

        # Get filtered detection indices
        ep_filtered = df_filtered[df_filtered['episode_id'] == ep_id]
        kept_indices = set(ep_filtered['detection_idx'].values)

        # Filter detections
        filtered_dets = []
        for idx, det in enumerate(orig['detections']):
            if idx in kept_indices:
                det_copy = dict(det)
                # Add GBT score from dataframe
                row = ep_filtered[ep_filtered['detection_idx'] == idx]
                if len(row) > 0 and 'gbt_score' in row.columns:
                    det_copy['gbt_score'] = round(float(row['gbt_score'].iloc[0]), 6)
                filtered_dets.append(det_copy)

        # Update output
        output = dict(orig)
        output['detections'] = filtered_dets
        output['n_detections'] = len(filtered_dets)
        output['variant'] = variant_name

        out_path = out_dir / f'{ep_id}_detections.json'
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=1)
        n_saved += 1

    print(f"  Saved {n_saved} episode JSONs to {out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# PER-EPISODE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════

def per_episode_breakdown(pipelines, test_ann, auto_ann):
    """Compute per-episode metrics for all pipelines."""
    # Collect all episode IDs
    all_eps = set()
    for name, dets_by_ep in pipelines.items():
        all_eps.update(dets_by_ep.keys())
    all_eps = sorted(all_eps)

    rows = []
    for ep_id in all_eps:
        row = {'episode_id': ep_id, 'is_held_out': ep_id in HELD_OUT}

        # Episode GT counts
        ep_test = test_ann[test_ann['video_id'] == ep_id]
        ep_auto = auto_ann[auto_ann['video_id'] == ep_id]
        row['test_gt'] = len(ep_test)
        row['auto_gt'] = len(ep_auto)

        for name, dets_by_ep in pipelines.items():
            dets = dets_by_ep.get(ep_id, [])
            row[f'{name}_n_dets'] = len(dets)

            # Test recall for this episode
            if len(ep_test) > 0:
                ep_dets_by_ep = {ep_id: dets}
                matched, total = eval_recall(
                    ep_dets_by_ep, ep_test, threshold=0.5, use_overlap=True)
                row[f'{name}_test_recall'] = matched / total if total > 0 else 0
            else:
                row[f'{name}_test_recall'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# LOAD DETECTIONS FROM JSON DIRS
# ══════════════════════════════════════════════════════════════════════════

def load_detections_from_dir(det_dir):
    """Load all detection JSONs from a directory into {ep_id: [det_list]}."""
    dets_by_ep = {}
    for f in sorted(det_dir.glob('*_detections.json')):
        ep_id = f.stem.replace('_detections', '')
        with open(f) as fh:
            data = json.load(fh)
        dets_by_ep[ep_id] = data['detections']
    return dets_by_ep


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    np.random.seed(SEED)

    print("=" * 70)
    print("GBT FILTERING FOR V3 DETECTIONS")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading segment features...")
    df = pd.read_csv(GBT_DIR / 'segment_features.csv')
    df['episode_id'] = df['episode_id'].astype(str)
    print(f"  {len(df)} segments, {df['label'].sum():.0f} TP, "
          f"{(1-df['label']).sum():.0f} FP")
    print(f"  Held-out: {df['is_held_out'].sum()} segments")

    print("\nLoading annotations...")
    test_ann = pd.read_csv(TEST_ANN_PATH)
    test_ann['video_id'] = test_ann['video_id'].astype(str)
    auto_ann = pd.read_csv(AUTO_ANN_PATH)
    auto_ann['video_id'] = auto_ann['video_id'].astype(str)
    print(f"  Test: {len(test_ann)}, Auto: {len(auto_ann)}")

    # ── Load V3 unfiltered detections ────────────────────────────────────
    print("\nLoading V3 unfiltered detections...")
    v3_dets = load_detections_from_dir(V3_DET_DIR)
    total_hours = compute_total_duration_hours(v3_dets)
    print(f"  {sum(len(v) for v in v3_dets.values())} detections, "
          f"{len(v3_dets)} episodes, {total_hours:.1f} hours")

    # ── Load V2+GBT reference detections ─────────────────────────────────
    print("\nLoading V2+GBT reference detections...")
    v2gbt_ref_dets = load_detections_from_dir(V2_GBT_DET_DIR)
    print(f"  {sum(len(v) for v in v2gbt_ref_dets.values())} detections")

    # ── Prepare features ─────────────────────────────────────────────────
    feature_cols = [c for c in GBT_FEATURE_NAMES if c in df.columns]
    missing_feats = set(GBT_FEATURE_NAMES) - set(feature_cols)
    if missing_feats:
        print(f"  WARNING: Missing features: {missing_feats}")

    X_all = df[feature_cols].values
    y_all = df['label'].values

    # Train/test split by held-out episodes
    train_mask = ~df['is_held_out'].values.astype(bool)
    test_mask = df['is_held_out'].values.astype(bool)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    groups_train = df.loc[train_mask, 'episode_id'].values

    print(f"\n  Train: {len(X_train)} ({y_train.sum():.0f} TP, "
          f"{(1-y_train).sum():.0f} FP)")
    print(f"  Test:  {len(X_test)} ({y_test.sum():.0f} TP, "
          f"{(1-y_test).sum():.0f} FP)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2A: Apply existing V2 GBT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STEP 2A: APPLY EXISTING V2 GBT TO V3 DETECTIONS")
    print(f"{'='*70}")

    with open(V2_GBT_PATH, 'rb') as f:
        v2_gbt_data = pickle.load(f)

    v2_clf = v2_gbt_data['model']
    v2_scaler = v2_gbt_data['scaler']
    v2_feat_names = v2_gbt_data['feature_names']
    print(f"  V2 GBT: {v2_gbt_data['model_type']}, CV F1={v2_gbt_data['cv_f1']:.3f}")
    print(f"  V2 features: {v2_feat_names}")

    # Score all segments with V2 GBT
    X_v2 = df[v2_feat_names].values
    X_v2_scaled = v2_scaler.transform(X_v2)
    v2_probs = v2_clf.predict_proba(X_v2_scaled)[:, 1]
    df['v2_gbt_score'] = v2_probs
    v2_kept_mask = v2_probs >= 0.5
    df['v2_gbt_kept'] = v2_kept_mask

    print(f"  V2 GBT kept: {v2_kept_mask.sum()} / {len(df)} "
          f"({v2_kept_mask.sum()/len(df):.1%})")
    print(f"  V2 GBT rejected: {(~v2_kept_mask).sum()}")

    # On held-out test set
    v2_test_preds = (v2_probs[test_mask] >= 0.5).astype(int)
    print(f"\n  Held-out performance (segment classification):")
    print(f"  F1:        {f1_score(y_test, v2_test_preds):.3f}")
    print(f"  Precision: {precision_score(y_test, v2_test_preds):.3f}")
    print(f"  Recall:    {recall_score(y_test, v2_test_preds):.3f}")
    if len(np.unique(y_test)) > 1:
        print(f"  AUROC:     {roc_auc_score(y_test, v2_probs[test_mask]):.3f}")

    # Build filtered detections
    v2gbt_filtered = df[df['v2_gbt_kept']].copy()
    v3_v2gbt_dets = defaultdict(list)
    for _, row in v2gbt_filtered.iterrows():
        ep_id = row['episode_id']
        det_idx = int(row['detection_idx'])
        orig_det = v3_dets[ep_id][det_idx]
        det_copy = dict(orig_det)
        det_copy['gbt_score'] = round(float(row['v2_gbt_score']), 6)
        v3_v2gbt_dets[ep_id].append(det_copy)
    # Ensure all episodes present (even with 0 detections)
    for ep_id in v3_dets:
        if ep_id not in v3_v2gbt_dets:
            v3_v2gbt_dets[ep_id] = []

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2B: Retrain fresh V3 GBT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STEP 2B: RETRAIN FRESH GBT ON V3 DETECTIONS")
    print(f"{'='*70}")

    # 5-fold episode-level CV on training data
    print("\n  5-fold episode-level CV...")
    gkf = GroupKFold(n_splits=5)
    cv_preds = np.zeros(len(y_train))

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(
            X_train, y_train, groups_train)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_idx])
        X_val = scaler.transform(X_train[val_idx])

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=SEED)
        clf.fit(X_tr, y_train[tr_idx])

        y_val_prob = clf.predict_proba(X_val)[:, 1]
        cv_preds[val_idx] = y_val_prob

        y_val_pred = (y_val_prob >= 0.5).astype(int)
        fold_f1 = f1_score(y_train[val_idx], y_val_pred)
        print(f"    Fold {fold_idx}: F1={fold_f1:.3f}, "
              f"P={precision_score(y_train[val_idx], y_val_pred):.3f}, "
              f"R={recall_score(y_train[val_idx], y_val_pred):.3f}")

    cv_pred_binary = (cv_preds >= 0.5).astype(int)
    cv_f1 = f1_score(y_train, cv_pred_binary)
    cv_auc = roc_auc_score(y_train, cv_preds) if len(np.unique(y_train)) > 1 else 0
    print(f"\n  Overall CV F1: {cv_f1:.3f}, AUROC: {cv_auc:.3f}")

    # Train final model on all training data
    print("\n  Training final V3 GBT on all training data...")
    v3_scaler = StandardScaler()
    X_train_scaled = v3_scaler.fit_transform(X_train)

    v3_gbt = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=SEED)
    v3_gbt.fit(X_train_scaled, y_train)

    # Evaluate on held-out
    X_test_scaled = v3_scaler.transform(X_test)
    v3_test_probs = v3_gbt.predict_proba(X_test_scaled)[:, 1]
    v3_test_preds = (v3_test_probs >= 0.5).astype(int)

    print(f"\n  Held-out performance (segment classification):")
    print(f"  F1:        {f1_score(y_test, v3_test_preds):.3f}")
    print(f"  Precision: {precision_score(y_test, v3_test_preds):.3f}")
    print(f"  Recall:    {recall_score(y_test, v3_test_preds):.3f}")
    if len(np.unique(y_test)) > 1:
        print(f"  AUROC:     {roc_auc_score(y_test, v3_test_probs):.3f}")

    # Feature importances
    importances = v3_gbt.feature_importances_
    imp_sorted = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print(f"\n  Feature importances:")
    for fname, imp in imp_sorted:
        print(f"    {fname:<25s} {imp:.4f}")

    # Save model
    v3_model_path = GBT_DIR / 'models' / 'gbt_v3.pkl'
    with open(v3_model_path, 'wb') as f:
        pickle.dump({
            'model': v3_gbt,
            'scaler': v3_scaler,
            'feature_names': feature_cols,
            'model_type': 'gradient_boosting',
            'cv_f1': float(cv_f1),
            'cv_auroc': float(cv_auc),
        }, f)
    print(f"\n  Saved V3 GBT model to {v3_model_path}")

    # Score all segments with V3 GBT
    X_all_scaled = v3_scaler.transform(X_all)
    v3_all_probs = v3_gbt.predict_proba(X_all_scaled)[:, 1]
    df['v3_gbt_score'] = v3_all_probs
    v3_kept_mask = v3_all_probs >= 0.5
    df['v3_gbt_kept'] = v3_kept_mask

    print(f"\n  V3 GBT kept: {v3_kept_mask.sum()} / {len(df)} "
          f"({v3_kept_mask.sum()/len(df):.1%})")

    # Build filtered detections
    v3gbt_filtered = df[df['v3_gbt_kept']].copy()
    v3_v3gbt_dets = defaultdict(list)
    for _, row in v3gbt_filtered.iterrows():
        ep_id = row['episode_id']
        det_idx = int(row['detection_idx'])
        orig_det = v3_dets[ep_id][det_idx]
        det_copy = dict(orig_det)
        det_copy['gbt_score'] = round(float(row['v3_gbt_score']), 6)
        v3_v3gbt_dets[ep_id].append(det_copy)
    for ep_id in v3_dets:
        if ep_id not in v3_v3gbt_dets:
            v3_v3gbt_dets[ep_id] = []

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: HEAD-TO-HEAD COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STEP 3: HEAD-TO-HEAD EVALUATION")
    print(f"{'='*70}")

    pipelines = {
        'V3 Unfiltered': v3_dets,
        'V3 + V2 GBT': dict(v3_v2gbt_dets),
        'V3 + V3 GBT': dict(v3_v3gbt_dets),
        'V2 + GBT (ref)': v2gbt_ref_dets,
    }

    results_list = []
    for name, dets_by_ep in pipelines.items():
        print(f"\n  Evaluating: {name}...")
        result = evaluate_pipeline(name, dets_by_ep, test_ann, auto_ann, total_hours)
        results_list.append(result)

        print(f"    Total:      {result['total_detections']:,}")
        print(f"    Det/hr:     {result['det_per_hour']:.1f}")
        print(f"    Test recall: {result['test_recall']:.1%} "
              f"({result['test_matched']}/{result['test_total']})")
        print(f"    Auto recall: {result['auto_recall']:.1%} "
              f"({result['auto_matched']}/{result['auto_total']})")
        print(f"    Prec proxy: {result['precision_proxy']:.1%}")
        print(f"    New disc:   {result['new_discoveries']:,}")

    # ── Save comparison table ─────────────────────────────────────────────
    comp_df = pd.DataFrame(results_list)
    comp_df.to_csv(GBT_DIR / 'comparison_table.csv', index=False)
    print(f"\n  Saved comparison_table.csv")

    # ── Save feature importances ──────────────────────────────────────────
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    imp_df.to_csv(GBT_DIR / 'feature_importances.csv', index=False)
    print("  Saved feature_importances.csv")

    # ── Per-episode breakdown ─────────────────────────────────────────────
    print("\n  Computing per-episode breakdown...")
    pipe_for_ep = {
        'v3': v3_dets,
        'v3_v2gbt': dict(v3_v2gbt_dets),
        'v3_v3gbt': dict(v3_v3gbt_dets),
        'v2_gbt_ref': v2gbt_ref_dets,
    }
    ep_df = per_episode_breakdown(pipe_for_ep, test_ann, auto_ann)
    ep_df.to_csv(GBT_DIR / 'per_episode_metrics.csv', index=False)
    print(f"  Saved per_episode_metrics.csv ({len(ep_df)} episodes)")

    # ── Save filtered detection JSONs ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 4: SAVE FILTERED DETECTIONS")
    print(f"{'='*70}")

    # Add gbt_score to df for saving
    save_filtered_detections(
        v2gbt_filtered, BASE / 'detections_v2gbt', 'v3_v2gbt')
    save_filtered_detections(
        v3gbt_filtered, BASE / 'detections_v3gbt', 'v3_v3gbt')

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    fig_dir = GBT_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    plot_comparison_bars(results_list, fig_dir)
    plot_feature_importances(importances, feature_cols, fig_dir)

    v2_kept_indices = df.index[df['v2_gbt_kept']]
    v3_kept_indices = df.index[df['v3_gbt_kept']]
    plot_confidence_distributions(df, v2_kept_indices, v3_kept_indices, fig_dir)

    # Use training data for threshold sweep
    df_train = df[~df['is_held_out'].astype(bool)].copy()
    plot_recall_precision_tradeoff(df_train, v3_gbt, v3_scaler, feature_cols, fig_dir)

    # ══════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("GENERATING REPORT")
    print(f"{'='*70}")

    # Notable episodes where GBT helps/hurts most
    held_out_ep = ep_df[ep_df['is_held_out']]
    if 'v3_test_recall' in held_out_ep.columns and 'v3_v3gbt_test_recall' in held_out_ep.columns:
        held_out_ep = held_out_ep.copy()
        held_out_ep['recall_change'] = (
            held_out_ep['v3_v3gbt_test_recall'] - held_out_ep['v3_test_recall'])

    report_lines = []
    report_lines.append("# GBT Filtering for V3 Detections\n")
    report_lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**V3 Model:** variant3_75_25.pt (75/25 hard/easy, 158d, 25f)")
    report_lines.append(f"**Segment Features:** {len(df)} segments, 15 features")
    report_lines.append(f"**V2 GBT:** {v2_gbt_data['model_type']}, "
                        f"CV F1={v2_gbt_data['cv_f1']:.3f}")
    report_lines.append(f"**V3 GBT:** gradient_boosting, CV F1={cv_f1:.3f}, "
                        f"AUROC={cv_auc:.3f}")
    report_lines.append("")
    report_lines.append("---\n")

    report_lines.append("## Head-to-Head Comparison\n")
    report_lines.append("| Metric | V3 Unfiltered | V3 + V2 GBT | V3 + V3 GBT "
                        "| V2 + GBT (ref) |")
    report_lines.append("|--------|--------------|-------------|-------------|"
                        "----------------|")

    r = {r['name']: r for r in results_list}
    names = ['V3 Unfiltered', 'V3 + V2 GBT', 'V3 + V3 GBT', 'V2 + GBT (ref)']
    metrics = [
        ('Total detections', 'total_detections', '{:,}'),
        ('Det/hour', 'det_per_hour', '{:.1f}'),
        ('Test recall', 'test_recall', '{:.1%}'),
        ('Auto recall', 'auto_recall', '{:.1%}'),
        ('Precision proxy', 'precision_proxy', '{:.1%}'),
        ('New discoveries', 'new_discoveries', '{:,}'),
        ('Mean duration', 'mean_duration', '{:.2f}s'),
    ]
    for label, key, fmt in metrics:
        vals = [fmt.format(r[n][key]) for n in names]
        report_lines.append(f"| {label} | {' | '.join(vals)} |")

    report_lines.append("")
    report_lines.append("---\n")

    # V3 GBT feature importances
    report_lines.append("## V3 GBT Feature Importances\n")
    report_lines.append("| Rank | Feature | Importance |")
    report_lines.append("|------|---------|------------|")
    for rank, (fname, imp) in enumerate(imp_sorted, 1):
        report_lines.append(f"| {rank} | {fname} | {imp:.4f} |")

    report_lines.append("")
    report_lines.append("---\n")

    # Held-out episode detail
    report_lines.append("## Held-Out Episode Detail\n")
    report_lines.append("| Episode | GT | V3 Dets | V3+V2GBT | V3+V3GBT | "
                        "V3 Recall | V3+V3GBT Recall |")
    report_lines.append("|---------|-----|---------|----------|----------|"
                        "-----------|-----------------|")

    for _, row in held_out_ep.sort_values('episode_id').iterrows():
        ep = row['episode_id']
        gt = row.get('test_gt', 0)
        v3n = row.get('v3_n_dets', 0)
        v2gn = row.get('v3_v2gbt_n_dets', 0)
        v3gn = row.get('v3_v3gbt_n_dets', 0)
        v3r = row.get('v3_test_recall', 0)
        v3gr = row.get('v3_v3gbt_test_recall', 0)
        report_lines.append(
            f"| {ep} | {gt:.0f} | {v3n:.0f} | {v2gn:.0f} | {v3gn:.0f} | "
            f"{v3r:.1%} | {v3gr:.1%} |")

    report_lines.append("")
    report_lines.append("---\n")

    # Segment classification performance
    report_lines.append("## Segment Classification (Held-Out)\n")
    report_lines.append("| Model | F1 | Precision | Recall | AUROC |")
    report_lines.append("|-------|----|-----------|--------|-------|")

    v2_f1 = f1_score(y_test, v2_test_preds)
    v2_p = precision_score(y_test, v2_test_preds)
    v2_r = recall_score(y_test, v2_test_preds)
    v2_auc = roc_auc_score(y_test, v2_probs[test_mask]) if len(np.unique(y_test)) > 1 else 0
    v3_f1 = f1_score(y_test, v3_test_preds)
    v3_p = precision_score(y_test, v3_test_preds)
    v3_r = recall_score(y_test, v3_test_preds)
    v3_auc = roc_auc_score(y_test, v3_test_probs) if len(np.unique(y_test)) > 1 else 0

    report_lines.append(f"| V2 GBT | {v2_f1:.3f} | {v2_p:.3f} | {v2_r:.3f} | {v2_auc:.3f} |")
    report_lines.append(f"| V3 GBT | {v3_f1:.3f} | {v3_p:.3f} | {v3_r:.3f} | {v3_auc:.3f} |")

    report_lines.append("")
    report_lines.append("---\n")

    report_lines.append("## Figures\n")
    report_lines.append("| Figure | Description |")
    report_lines.append("|--------|-------------|")
    report_lines.append("| `comparison_bars.png` | Bar charts: detections, recall, "
                        "precision, density |")
    report_lines.append("| `feature_importances.png` | V3 GBT feature importance ranking |")
    report_lines.append("| `confidence_distributions.png` | Kept vs rejected confidence "
                        "distributions |")
    report_lines.append("| `recall_precision_tradeoff.png` | Recall vs precision at "
                        "different GBT thresholds |")
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*Generated by apply_gbt_filter.py*\n")

    report_path = GBT_DIR / 'GBT_V3_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved {report_path}")

    # Save updated segment features with GBT scores
    df.to_csv(GBT_DIR / 'segment_features.csv', index=False)
    print("  Updated segment_features.csv with GBT scores")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"GBT FILTERING COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Print summary
    print("\nSUMMARY:")
    print(f"{'Pipeline':<20s} {'Total':>8s} {'TestRec':>8s} {'AutoRec':>8s} "
          f"{'PrecPxy':>8s} {'NewDisc':>8s}")
    for res in results_list:
        print(f"{res['name']:<20s} {res['total_detections']:>8,} "
              f"{res['test_recall']:>7.1%} {res['auto_recall']:>7.1%} "
              f"{res['precision_proxy']:>7.1%} {res['new_discoveries']:>8,}")


if __name__ == '__main__':
    main()
