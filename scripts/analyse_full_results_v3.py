#!/usr/bin/env python3
"""
Analyse full BOBSL V3 detection results: compare V1, V2 Opt, V2 GBT, V3 Hard-Neg.

Steps:
  1. Load all detection sets
  2. Detection statistics
  3. Recall against test + auto annotations
  4. New discoveries
  5. Precision proxy
  6. Per-episode breakdown
  7. Head-to-head comparison table
  8. Figures
  9. Report
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(os.path.expanduser('~/bsl_project/full_detection_v3/'))
V1_DET_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_episode_detection/detections/'))
V2_OPT_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_detection_v2/detections_optimised/'))
V2_GBT_DIR = Path(os.path.expanduser(
    '~/bsl_project/full_detection_v2/detections_gbt/'))
V3_DET_DIR = BASE_DIR / 'detections'

RESULTS_DIR = BASE_DIR / 'results'
FIGURE_DIR = RESULTS_DIR / 'figures'
EVAL_DIR = BASE_DIR / 'evaluation'

TEST_ANN_PATH = os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/transpeller-test.csv')
AUTO_ANN_PATH = os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/fingerspelling-automatic-annotations.csv')

FPS = 25
HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_detections(det_dir, label=''):
    """Load all detection JSONs from a directory."""
    ep_data = {}
    for f in sorted(det_dir.glob('*_detections.json')):
        ep_id = f.stem.replace('_detections', '')
        with open(f) as fh:
            d = json.load(fh)
        ep_data[ep_id] = d
    if label:
        print(f"  Loaded {len(ep_data)} episodes from {label}")
    return ep_data


def load_annotations():
    test = pd.read_csv(TEST_ANN_PATH)
    test['video_id'] = test['video_id'].astype(str)
    auto = pd.read_csv(AUTO_ANN_PATH)
    auto['video_id'] = auto['video_id'].astype(str)
    return test, auto


# ══════════════════════════════════════════════════════════════════════════
# MATCHING
# ══════════════════════════════════════════════════════════════════════════

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0


def compute_overlap(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    det_dur = pe - ps
    return inter / det_dur if det_dur > 0 else 0


def match_detections_to_gt(detections, gt_list, threshold=0.3, use_overlap=False):
    if not detections or not gt_list:
        return 0, set(), set(range(len(detections)))

    score_fn = compute_overlap if use_overlap else compute_iou
    score_mat = np.zeros((len(detections), len(gt_list)))
    for i, d in enumerate(detections):
        for j, g in enumerate(gt_list):
            score_mat[i, j] = score_fn(d['start_time'], d['end_time'],
                                       g['start'], g['end'])

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

    unmatched_d = set(range(len(detections))) - matched_d
    return len(matched_d), matched_g, unmatched_d


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_detection_stats(ep_data):
    total_dets = 0
    total_duration_h = 0
    durations, confidences = [], []

    for ep_id, d in ep_data.items():
        dets = d['detections']
        total_dets += len(dets)
        total_duration_h += d['duration_sec'] / 3600
        for det in dets:
            durations.append(det['duration'])
            confidences.append(det['frame_confidence'])

    durations = np.array(durations) if durations else np.array([0])
    confidences = np.array(confidences) if confidences else np.array([0])

    return {
        'total_detections': total_dets,
        'total_hours': total_duration_h,
        'det_per_hour': total_dets / total_duration_h if total_duration_h > 0 else 0,
        'mean_duration': float(np.mean(durations)),
        'median_duration': float(np.median(durations)),
        'std_duration': float(np.std(durations)),
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences)),
        'durations': durations,
        'confidences': confidences,
    }


def compute_recall(ep_data, ann_df, use_overlap=False, threshold=0.3):
    ep_ids = set(ep_data.keys())
    ann_in = ann_df[ann_df['video_id'].isin(ep_ids)]

    total_matched = 0
    total_gt = 0
    per_ep = {}

    for ep_id in ep_ids:
        ep_ann = ann_in[ann_in['video_id'] == ep_id]
        gt_list = [{'start': r['start'], 'end': r['end']} for _, r in ep_ann.iterrows()]
        total_gt += len(gt_list)

        dets = ep_data[ep_id]['detections']
        n_matched, matched_gt, _ = match_detections_to_gt(
            dets, gt_list, threshold=threshold, use_overlap=use_overlap)
        total_matched += n_matched

        ep_recall = n_matched / len(gt_list) if gt_list else 0
        per_ep[ep_id] = {
            'n_gt': len(gt_list),
            'n_matched': n_matched,
            'recall': ep_recall,
            'n_detections': len(dets),
        }

    overall_recall = total_matched / total_gt if total_gt > 0 else 0
    return {
        'total_gt': total_gt,
        'total_matched': total_matched,
        'overall_recall': overall_recall,
        'per_episode': per_ep,
    }


def compute_new_discoveries(ep_data, test_df, auto_df, threshold=0.1):
    ep_ids = set(ep_data.keys())
    total_new = 0
    total_checked = 0
    per_ep_new = {}

    for ep_id in ep_ids:
        test_gt = [{'start': r['start'], 'end': r['end']}
                   for _, r in test_df[test_df['video_id'] == ep_id].iterrows()]
        auto_gt = [{'start': r['start'], 'end': r['end']}
                   for _, r in auto_df[auto_df['video_id'] == ep_id].iterrows()]
        all_gt = test_gt + auto_gt

        dets = ep_data[ep_id]['detections']
        total_checked += len(dets)

        if not all_gt:
            total_new += len(dets)
            per_ep_new[ep_id] = len(dets)
            continue

        _, _, unmatched = match_detections_to_gt(
            dets, all_gt, threshold=threshold, use_overlap=False)
        total_new += len(unmatched)
        per_ep_new[ep_id] = len(unmatched)

    return total_new, total_checked, per_ep_new


def compute_precision_proxy(ep_data, test_df, auto_df, threshold=0.3):
    ep_ids = set(ep_data.keys())
    total_matched = 0
    total_dets = 0

    for ep_id in ep_ids:
        test_gt = [{'start': r['start'], 'end': r['end']}
                   for _, r in test_df[test_df['video_id'] == ep_id].iterrows()]
        auto_gt = [{'start': r['start'], 'end': r['end']}
                   for _, r in auto_df[auto_df['video_id'] == ep_id].iterrows()]
        all_gt = test_gt + auto_gt

        dets = ep_data[ep_id]['detections']
        total_dets += len(dets)

        if all_gt:
            n_matched, _, _ = match_detections_to_gt(
                dets, all_gt, threshold=threshold, use_overlap=False)
            total_matched += n_matched

    proxy = total_matched / total_dets if total_dets > 0 else 0
    return {'matched': total_matched, 'total': total_dets, 'precision_proxy': proxy}


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════

COLORS = {
    'V1': '#4e79a7',
    'V2 Opt': '#e15759',
    'V2 GBT': '#59a14f',
    'V3': '#f28e2b',
}


def fig_comparison_bars(all_stats, all_recall):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = list(all_stats.keys())
    colors = [COLORS.get(l, '#999') for l in labels]

    # Detection count
    ax = axes[0]
    vals = [all_stats[l]['total_detections'] for l in labels]
    bars = ax.bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 200, f'{v:,}',
                ha='center', fontsize=9)
    ax.set_ylabel('Total Detections')
    ax.set_title('Total Detections')

    # Recall
    ax = axes[1]
    test_vals = [all_recall[l]['test']['overall_recall'] for l in labels]
    auto_vals = [all_recall[l]['auto']['overall_recall'] for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, test_vals, w, label='Test recall', color='#4e79a7', alpha=0.8)
    ax.bar(x + w/2, auto_vals, w, label='Auto recall', color='#e15759', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Annotations')
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, (tv, av) in enumerate(zip(test_vals, auto_vals)):
        ax.text(i - w/2, tv + 0.02, f'{tv:.1%}', ha='center', fontsize=8)
        ax.text(i + w/2, av + 0.02, f'{av:.1%}', ha='center', fontsize=8)

    # Density
    ax = axes[2]
    dens = [all_stats[l]['det_per_hour'] for l in labels]
    bars = ax.bar(labels, dens, color=colors)
    for bar, v in zip(bars, dens):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}',
                ha='center', fontsize=9)
    ax.set_ylabel('Detections/hour')
    ax.set_title('Detection Density')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comparison_bars.png")


def fig_per_episode_recall_scatter(all_recall):
    """Scatter: per-episode test recall V1 vs V3 and V2 vs V3."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [('V1', 'V3'), ('V2 Opt', 'V3'), ('V2 GBT', 'V3')]
    for ax, (old_label, new_label) in zip(axes, pairs):
        old_per_ep = all_recall[old_label]['test']['per_episode']
        new_per_ep = all_recall[new_label]['test']['per_episode']

        old_r, new_r = [], []
        for ep_id in old_per_ep:
            if ep_id in new_per_ep and old_per_ep[ep_id]['n_gt'] > 0:
                old_r.append(old_per_ep[ep_id]['recall'])
                new_r.append(new_per_ep[ep_id]['recall'])

        ax.scatter(old_r, new_r, alpha=0.5, s=15, color=COLORS.get(new_label, '#f28e2b'))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel(f'{old_label} Recall')
        ax.set_ylabel(f'{new_label} Recall')
        ax.set_title(f'{old_label} vs {new_label}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

        # Count improvements
        improved = sum(1 for o, n in zip(old_r, new_r) if n > o + 0.01)
        degraded = sum(1 for o, n in zip(old_r, new_r) if n < o - 0.01)
        ax.text(0.05, 0.95, f'Improved: {improved}\nDegraded: {degraded}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'per_episode_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved per_episode_scatter.png")


def fig_duration_distributions(all_stats):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 6, 70)
    for label in all_stats:
        ax.hist(all_stats[label]['durations'], bins=bins, alpha=0.4,
                label=f"{label} (n={all_stats[label]['total_detections']:,})",
                color=COLORS.get(label, '#999'), density=True)
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Density')
    ax.set_title('Detection Duration Distributions')
    ax.legend()
    ax.set_xlim(0, 6)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'duration_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved duration_distributions.png")


def fig_confidence_distributions(all_stats):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)
    for label in all_stats:
        ax.hist(all_stats[label]['confidences'], bins=bins, alpha=0.4,
                label=f"{label} (mean={all_stats[label]['mean_confidence']:.3f})",
                color=COLORS.get(label, '#999'), density=True)
    ax.set_xlabel('Frame Confidence (mean prob)')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distributions')
    ax.legend()
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'confidence_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved confidence_distributions.png")


def fig_recall_sorted_bars(all_recall, test_df):
    """Per-episode test recall sorted by V1 recall, showing all 4 pipelines."""
    v1_per_ep = all_recall['V1']['test']['per_episode']
    ep_ids = sorted(v1_per_ep.keys(),
                    key=lambda e: v1_per_ep[e]['recall'])
    ep_ids = [e for e in ep_ids if v1_per_ep[e]['n_gt'] > 0]

    fig, ax = plt.subplots(figsize=(18, 5))
    x = np.arange(len(ep_ids))
    n_labels = len(all_recall)
    w = 0.8 / n_labels

    for i, (label, recall_data) in enumerate(all_recall.items()):
        per_ep = recall_data['test']['per_episode']
        recalls = [per_ep.get(e, {}).get('recall', 0) for e in ep_ids]
        offset = (i - n_labels/2 + 0.5) * w
        ax.bar(x + offset, recalls, w, label=label,
               color=COLORS.get(label, '#999'), alpha=0.7)

    ax.set_ylabel('Test Recall')
    ax.set_title(f'Per-Episode Test Recall ({len(ep_ids)} episodes, sorted by V1 recall)')
    ax.legend(loc='upper left')
    ax.set_xlim(-1, len(ep_ids))
    ax.set_xticks([])
    ax.set_xlabel('Episodes (sorted by V1 recall)')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'episode_recall_sorted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved episode_recall_sorted.png")


def fig_recall_improvement_heatmap(all_recall):
    """Show which episodes gained/lost most from V2 Opt -> V3."""
    v2_per_ep = all_recall['V2 Opt']['test']['per_episode']
    v3_per_ep = all_recall['V3']['test']['per_episode']

    ep_ids = []
    deltas = []
    for ep_id in v2_per_ep:
        if ep_id in v3_per_ep and v2_per_ep[ep_id]['n_gt'] > 0:
            d = v3_per_ep[ep_id]['recall'] - v2_per_ep[ep_id]['recall']
            ep_ids.append(ep_id)
            deltas.append(d)

    # Sort by delta
    order = np.argsort(deltas)
    ep_ids = [ep_ids[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(16, 5))
    colors_bar = ['#e15759' if d < -0.01 else '#59a14f' if d > 0.01 else '#999'
                  for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors_bar, width=1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Episodes (sorted by recall change)')
    ax.set_ylabel('Recall Change (V3 - V2 Opt)')
    ax.set_title('Per-Episode Recall Change: V3 vs V2 Optimised')
    ax.set_xticks([])

    # Annotate extremes
    n_improved = sum(1 for d in deltas if d > 0.01)
    n_degraded = sum(1 for d in deltas if d < -0.01)
    n_same = len(deltas) - n_improved - n_degraded
    ax.text(0.02, 0.95, f'Improved: {n_improved} | Same: {n_same} | Degraded: {n_degraded}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'recall_improvement_v2_vs_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved recall_improvement_v2_vs_v3.png")


def fig_detection_density_histogram(all_data):
    """Per-episode detection density histogram for each pipeline."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 300, 60)

    for label, ep_data in all_data.items():
        densities = []
        for ep_id, d in ep_data.items():
            dur_h = d['duration_sec'] / 3600
            if dur_h > 0:
                densities.append(len(d['detections']) / dur_h)
        ax.hist(densities, bins=bins, alpha=0.4,
                label=f"{label} (mean={np.mean(densities):.1f}/hr)",
                color=COLORS.get(label, '#999'), density=True)

    ax.set_xlabel('Detections per hour')
    ax.set_ylabel('Density')
    ax.set_title('Per-Episode Detection Density')
    ax.legend()
    ax.set_xlim(0, 300)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'detection_density_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved detection_density_histogram.png")


# ══════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════

def generate_report(all_stats, all_recall, all_new_disc, all_prec_proxy,
                    per_ep_df, low_recall_analysis):

    # Build comparison table rows
    rows = []
    for label in ['V1', 'V2 Opt', 'V2 GBT', 'V3']:
        s = all_stats[label]
        r = all_recall[label]
        rows.append(
            f"| {label} | {s['total_detections']:,} | {s['det_per_hour']:.1f} | "
            f"{r['test']['overall_recall']:.1%} | {r['auto']['overall_recall']:.1%} | "
            f"{all_new_disc[label]:,} | {s['mean_duration']:.2f}s | "
            f"{s['mean_confidence']:.3f} | {all_prec_proxy[label]['precision_proxy']:.1%} |"
        )
    table = "\n".join(rows)

    report = f"""# Full BOBSL Detection V3 — Hard-Negative CNN

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Model:** V3 Hard-Negative CNN (75/25 hard/easy, 158d, 25f window, unified source)
**Episodes:** {len(per_ep_df)} ({all_stats['V3']['total_hours']:.1f} hours)
**Post-processing:** k=11, t=0.30, gap=0.5s, min_dur=0.5s, conf_filter=0.60, ext=0.2s

---

## Head-to-Head Comparison

| Pipeline | Total Det | Det/hr | Test Recall | Auto Recall | New Disc | Mean Dur | Mean Conf | Prec Proxy |
|----------|-----------|--------|-------------|-------------|----------|----------|-----------|------------|
{table}

**V1:** 6-joint CNN, old post-processing (median k=13, t=0.50)
**V2 Opt:** Rich CNN (158d), optimised post-processing (k=11, t=0.30, gap=0.5s, cf=0.60)
**V2 GBT:** Rich CNN + GBT second-stage filter (replaces confidence filter)
**V3:** Hard-negative CNN (75/25 hard/easy, trained on unified full-episode data)

---

## Key Findings

### V3 vs V2 Optimised (primary comparison)

| Metric | V2 Opt | V3 | Change |
|--------|--------|-----|--------|
| Total detections | {all_stats['V2 Opt']['total_detections']:,} | {all_stats['V3']['total_detections']:,} | {all_stats['V3']['total_detections'] - all_stats['V2 Opt']['total_detections']:+,} |
| Det/hour | {all_stats['V2 Opt']['det_per_hour']:.1f} | {all_stats['V3']['det_per_hour']:.1f} | {all_stats['V3']['det_per_hour'] - all_stats['V2 Opt']['det_per_hour']:+.1f} |
| Test recall | {all_recall['V2 Opt']['test']['overall_recall']:.1%} | {all_recall['V3']['test']['overall_recall']:.1%} | {(all_recall['V3']['test']['overall_recall'] - all_recall['V2 Opt']['test']['overall_recall'])*100:+.1f}pp |
| Auto recall | {all_recall['V2 Opt']['auto']['overall_recall']:.1%} | {all_recall['V3']['auto']['overall_recall']:.1%} | {(all_recall['V3']['auto']['overall_recall'] - all_recall['V2 Opt']['auto']['overall_recall'])*100:+.1f}pp |
| Precision proxy | {all_prec_proxy['V2 Opt']['precision_proxy']:.1%} | {all_prec_proxy['V3']['precision_proxy']:.1%} | {(all_prec_proxy['V3']['precision_proxy'] - all_prec_proxy['V2 Opt']['precision_proxy'])*100:+.1f}pp |
| Mean confidence | {all_stats['V2 Opt']['mean_confidence']:.3f} | {all_stats['V3']['mean_confidence']:.3f} | {all_stats['V3']['mean_confidence'] - all_stats['V2 Opt']['mean_confidence']:+.3f} |

### Low-Recall Episode Analysis

{low_recall_analysis}

---

## Detection Statistics

| Pipeline | Total | Det/hr | Mean Dur | Median Dur | Std Dur | Mean Conf | Median Conf |
|----------|-------|--------|----------|------------|---------|-----------|-------------|
| V1 | {all_stats['V1']['total_detections']:,} | {all_stats['V1']['det_per_hour']:.1f} | {all_stats['V1']['mean_duration']:.2f}s | {all_stats['V1']['median_duration']:.2f}s | {all_stats['V1']['std_duration']:.2f}s | {all_stats['V1']['mean_confidence']:.3f} | {all_stats['V1']['median_confidence']:.3f} |
| V2 Opt | {all_stats['V2 Opt']['total_detections']:,} | {all_stats['V2 Opt']['det_per_hour']:.1f} | {all_stats['V2 Opt']['mean_duration']:.2f}s | {all_stats['V2 Opt']['median_duration']:.2f}s | {all_stats['V2 Opt']['std_duration']:.2f}s | {all_stats['V2 Opt']['mean_confidence']:.3f} | {all_stats['V2 Opt']['median_confidence']:.3f} |
| V2 GBT | {all_stats['V2 GBT']['total_detections']:,} | {all_stats['V2 GBT']['det_per_hour']:.1f} | {all_stats['V2 GBT']['mean_duration']:.2f}s | {all_stats['V2 GBT']['median_duration']:.2f}s | {all_stats['V2 GBT']['std_duration']:.2f}s | {all_stats['V2 GBT']['mean_confidence']:.3f} | {all_stats['V2 GBT']['median_confidence']:.3f} |
| V3 | {all_stats['V3']['total_detections']:,} | {all_stats['V3']['det_per_hour']:.1f} | {all_stats['V3']['mean_duration']:.2f}s | {all_stats['V3']['median_duration']:.2f}s | {all_stats['V3']['std_duration']:.2f}s | {all_stats['V3']['mean_confidence']:.3f} | {all_stats['V3']['median_confidence']:.3f} |

---

## Recall Analysis

### Test Annotations ({all_recall['V3']['test']['total_gt']:,} events)

| Pipeline | Matched | Total | Recall |
|----------|---------|-------|--------|
| V1 | {all_recall['V1']['test']['total_matched']:,} | {all_recall['V1']['test']['total_gt']:,} | {all_recall['V1']['test']['overall_recall']:.1%} |
| V2 Opt | {all_recall['V2 Opt']['test']['total_matched']:,} | {all_recall['V2 Opt']['test']['total_gt']:,} | {all_recall['V2 Opt']['test']['overall_recall']:.1%} |
| V2 GBT | {all_recall['V2 GBT']['test']['total_matched']:,} | {all_recall['V2 GBT']['test']['total_gt']:,} | {all_recall['V2 GBT']['test']['overall_recall']:.1%} |
| V3 | {all_recall['V3']['test']['total_matched']:,} | {all_recall['V3']['test']['total_gt']:,} | {all_recall['V3']['test']['overall_recall']:.1%} |

### Auto Annotations ({all_recall['V3']['auto']['total_gt']:,} events)

| Pipeline | Matched | Total | Recall |
|----------|---------|-------|--------|
| V1 | {all_recall['V1']['auto']['total_matched']:,} | {all_recall['V1']['auto']['total_gt']:,} | {all_recall['V1']['auto']['overall_recall']:.1%} |
| V2 Opt | {all_recall['V2 Opt']['auto']['total_matched']:,} | {all_recall['V2 Opt']['auto']['total_gt']:,} | {all_recall['V2 Opt']['auto']['overall_recall']:.1%} |
| V2 GBT | {all_recall['V2 GBT']['auto']['total_matched']:,} | {all_recall['V2 GBT']['auto']['total_gt']:,} | {all_recall['V2 GBT']['auto']['overall_recall']:.1%} |
| V3 | {all_recall['V3']['auto']['total_matched']:,} | {all_recall['V3']['auto']['total_gt']:,} | {all_recall['V3']['auto']['overall_recall']:.1%} |

---

## Precision Proxy

Fraction of detections matching any known annotation (test + auto, IoU >= 0.3):

| Pipeline | Matched | Total | Proxy |
|----------|---------|-------|-------|
| V1 | {all_prec_proxy['V1']['matched']:,} | {all_prec_proxy['V1']['total']:,} | {all_prec_proxy['V1']['precision_proxy']:.1%} |
| V2 Opt | {all_prec_proxy['V2 Opt']['matched']:,} | {all_prec_proxy['V2 Opt']['total']:,} | {all_prec_proxy['V2 Opt']['precision_proxy']:.1%} |
| V2 GBT | {all_prec_proxy['V2 GBT']['matched']:,} | {all_prec_proxy['V2 GBT']['total']:,} | {all_prec_proxy['V2 GBT']['precision_proxy']:.1%} |
| V3 | {all_prec_proxy['V3']['matched']:,} | {all_prec_proxy['V3']['total']:,} | {all_prec_proxy['V3']['precision_proxy']:.1%} |

---

## Figures

| Figure | Description |
|--------|-------------|
| `comparison_bars.png` | Bar charts: detection count, recall, density |
| `per_episode_scatter.png` | Per-episode recall scatter: V1/V2/GBT vs V3 |
| `duration_distributions.png` | Detection duration distributions |
| `confidence_distributions.png` | Confidence distributions |
| `episode_recall_sorted.png` | Per-episode test recall sorted by V1 |
| `recall_improvement_v2_vs_v3.png` | Per-episode recall change V2 Opt -> V3 |
| `detection_density_histogram.png` | Per-episode detection density distributions |

---

*Generated by analyse_full_results_v3.py*
"""

    with open(RESULTS_DIR / 'FULL_DETECTION_V3_REPORT.md', 'w') as f:
        f.write(report)
    print("  Saved FULL_DETECTION_V3_REPORT.md")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("=" * 70)
    print("FULL DETECTION ANALYSIS — V3 Hard-Negative CNN")
    print("=" * 70)

    # Load detections
    print("\nLoading detections...")
    all_data = {
        'V1': load_detections(V1_DET_DIR, 'V1 (6-joint)'),
        'V2 Opt': load_detections(V2_OPT_DIR, 'V2 Optimised'),
        'V2 GBT': load_detections(V2_GBT_DIR, 'V2 GBT'),
        'V3': load_detections(V3_DET_DIR, 'V3 Hard-Neg'),
    }

    # Load annotations
    print("\nLoading annotations...")
    test_df, auto_df = load_annotations()
    print(f"  Test: {len(test_df)} annotations")
    print(f"  Auto: {len(auto_df)} annotations")

    # Common episodes
    common_eps = set.intersection(*[set(d.keys()) for d in all_data.values()])
    print(f"\n  Common episodes: {len(common_eps)}")

    # ── Detection stats ──
    print("\n" + "=" * 70)
    print("DETECTION STATISTICS")
    print("=" * 70)
    all_stats = {}
    for label, ep_data in all_data.items():
        stats = compute_detection_stats(ep_data)
        all_stats[label] = stats
        print(f"\n  {label}:")
        print(f"    Total: {stats['total_detections']:,} ({stats['total_hours']:.1f}h)")
        print(f"    Density: {stats['det_per_hour']:.1f}/hr")
        print(f"    Duration: mean={stats['mean_duration']:.2f}s, median={stats['median_duration']:.2f}s")
        print(f"    Confidence: mean={stats['mean_confidence']:.3f}")

    # ── Recall ──
    print("\n" + "=" * 70)
    print("RECALL ANALYSIS")
    print("=" * 70)
    all_recall = {}
    for label, ep_data in all_data.items():
        # Test recall with overlap >= 0.5 (same as V2 analysis)
        test_r = compute_recall(ep_data, test_df, use_overlap=True, threshold=0.5)
        auto_r = compute_recall(ep_data, auto_df, use_overlap=False, threshold=0.3)
        all_recall[label] = {'test': test_r, 'auto': auto_r}

        print(f"\n  {label}:")
        print(f"    Test:  {test_r['total_matched']}/{test_r['total_gt']} = {test_r['overall_recall']:.1%}")
        print(f"    Auto:  {auto_r['total_matched']}/{auto_r['total_gt']} = {auto_r['overall_recall']:.1%}")

    # Save recall JSONs
    for label in ['V3']:
        recall_data = all_recall[label]
        test_out = {
            'total_gt': recall_data['test']['total_gt'],
            'total_matched': recall_data['test']['total_matched'],
            'overall_recall': recall_data['test']['overall_recall'],
            'per_episode': {k: v for k, v in recall_data['test']['per_episode'].items()},
        }
        with open(EVAL_DIR / 'test_recall.json', 'w') as f:
            json.dump(test_out, f, indent=2)

        auto_out = {
            'total_gt': recall_data['auto']['total_gt'],
            'total_matched': recall_data['auto']['total_matched'],
            'overall_recall': recall_data['auto']['overall_recall'],
            'per_episode': {k: v for k, v in recall_data['auto']['per_episode'].items()},
        }
        with open(EVAL_DIR / 'auto_recall.json', 'w') as f:
            json.dump(auto_out, f, indent=2)

    # ── New discoveries ──
    print("\n" + "=" * 70)
    print("NEW DISCOVERIES")
    print("=" * 70)
    all_new_disc = {}
    for label, ep_data in all_data.items():
        n_new, n_total, per_ep_new = compute_new_discoveries(ep_data, test_df, auto_df)
        all_new_disc[label] = n_new
        print(f"  {label}: {n_new:,} new (of {n_total:,} total)")

        if label == 'V3':
            # Save new discoveries CSV
            rows = []
            for ep_id in sorted(per_ep_new.keys()):
                if per_ep_new[ep_id] > 0:
                    rows.append({'episode_id': ep_id, 'n_new_discoveries': per_ep_new[ep_id]})
            pd.DataFrame(rows).to_csv(EVAL_DIR / 'new_discoveries.csv', index=False)

    # ── Precision proxy ──
    print("\n" + "=" * 70)
    print("PRECISION PROXY")
    print("=" * 70)
    all_prec_proxy = {}
    for label, ep_data in all_data.items():
        prec = compute_precision_proxy(ep_data, test_df, auto_df)
        all_prec_proxy[label] = prec
        print(f"  {label}: {prec['matched']}/{prec['total']} = {prec['precision_proxy']:.1%}")

    # ── Per-episode breakdown ──
    print("\n" + "=" * 70)
    print("PER-EPISODE BREAKDOWN")
    print("=" * 70)
    per_ep_rows = []
    for ep_id in sorted(common_eps):
        row = {
            'episode_id': ep_id,
            'is_held_out': ep_id in HELD_OUT,
            'n_frames': all_data['V1'][ep_id]['n_frames'],
            'duration_min': all_data['V1'][ep_id]['duration_sec'] / 60,
        }
        for label in ['V1', 'V2 Opt', 'V2 GBT', 'V3']:
            recall_data = all_recall[label]
            ep_r = recall_data['test']['per_episode'].get(ep_id, {})
            row[f'{label.lower().replace(" ", "_")}_test_recall'] = ep_r.get('recall', 0)
            row[f'{label.lower().replace(" ", "_")}_n_detections'] = ep_r.get('n_detections', 0)
            row[f'{label.lower().replace(" ", "_")}_test_gt'] = ep_r.get('n_gt', 0)
        per_ep_rows.append(row)

    per_ep_df = pd.DataFrame(per_ep_rows)
    per_ep_df.to_csv(RESULTS_DIR / 'per_episode_metrics.csv', index=False)
    print(f"  Saved per_episode_metrics.csv ({len(per_ep_df)} rows)")

    # ── Low-recall episode analysis ──
    print("\n  Low-recall episode analysis...")
    v2_per_ep = all_recall['V2 Opt']['test']['per_episode']
    v3_per_ep = all_recall['V3']['test']['per_episode']

    # Episodes with <20% V2 recall that have GT
    low_recall_eps = [e for e in v2_per_ep
                      if v2_per_ep[e]['n_gt'] > 0 and v2_per_ep[e]['recall'] < 0.20]
    if low_recall_eps:
        lr_lines = [f"Episodes with V2 test recall < 20% (n={len(low_recall_eps)}):\n"]
        lr_lines.append("| Episode | GT | V1 Recall | V2 Recall | V3 Recall | Change |")
        lr_lines.append("|---------|-----|-----------|-----------|-----------|--------|")
        for ep_id in sorted(low_recall_eps, key=lambda e: v2_per_ep[e]['recall']):
            v1_r = all_recall['V1']['test']['per_episode'].get(ep_id, {}).get('recall', 0)
            v2_r = v2_per_ep[ep_id]['recall']
            v3_r = v3_per_ep.get(ep_id, {}).get('recall', 0)
            n_gt = v2_per_ep[ep_id]['n_gt']
            lr_lines.append(f"| {ep_id} | {n_gt} | {v1_r:.1%} | {v2_r:.1%} | {v3_r:.1%} | {(v3_r-v2_r)*100:+.1f}pp |")
        low_recall_analysis = "\n".join(lr_lines)
    else:
        low_recall_analysis = "No episodes with V2 test recall < 20%."

    # ── Comparison table CSV ──
    comp_rows = []
    for label in ['V1', 'V2 Opt', 'V2 GBT', 'V3']:
        s = all_stats[label]
        r = all_recall[label]
        comp_rows.append({
            'pipeline': label,
            'total_detections': s['total_detections'],
            'det_per_hour': s['det_per_hour'],
            'test_recall': r['test']['overall_recall'],
            'auto_recall': r['auto']['overall_recall'],
            'new_discoveries': all_new_disc[label],
            'mean_duration': s['mean_duration'],
            'mean_confidence': s['mean_confidence'],
            'precision_proxy': all_prec_proxy[label]['precision_proxy'],
        })
    pd.DataFrame(comp_rows).to_csv(RESULTS_DIR / 'comparison_table.csv', index=False)
    print("  Saved comparison_table.csv")

    # ── Figures ──
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    fig_comparison_bars(all_stats, all_recall)
    fig_per_episode_recall_scatter(all_recall)
    fig_duration_distributions(all_stats)
    fig_confidence_distributions(all_stats)
    fig_recall_sorted_bars(all_recall, test_df)
    fig_recall_improvement_heatmap(all_recall)
    fig_detection_density_histogram(all_data)

    # ── Report ──
    print("\n  Generating report...")
    generate_report(all_stats, all_recall, all_new_disc, all_prec_proxy,
                    per_ep_df, low_recall_analysis)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
