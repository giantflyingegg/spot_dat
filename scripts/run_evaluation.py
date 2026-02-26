#!/usr/bin/env python3
"""V3+GBT Detection Evaluation Deep-Dive + Subtitle Cross-Validation.

Steps:
  1. Ground truth coverage analysis (test + auto GT)
  2. Miss pattern analysis (duration, word, episode, temporal, near-miss)
  3. Detection validation via subtitle cross-referencing (BOBSL inverted pipeline)
  4. NHM detection subtitle cross-referencing
  5. Generate comprehensive report + figures
"""

import csv
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════

BASE = Path(os.path.expanduser('~/bsl_project'))
DET_V3GBT_DIR = BASE / 'full_detection_v3' / 'detections_v3gbt'
SEG_FEATURES = BASE / 'full_detection_v3' / 'gbt_filtering' / 'segment_features.csv'

TEST_ANN_PATH = Path(os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/transpeller-test.csv'))
AUTO_ANN_PATH = Path(os.path.expanduser(
    '~/transpeller/data/fingerspelling-data-bmvc2022/fingerspelling-automatic-annotations.csv'))

SUBTITLES_DIR = BASE / 'data' / 'subtitles'
SIGN_VOCAB_PATH = BASE / 'data' / 'sign_vocabulary_dict.csv'

NHM_DET_DIR = BASE / 'youtube_mining' / 'output_v3_gbt'
NHM_OLD_OUTPUT = BASE / 'youtube_mining' / 'output'

OUT_DIR = BASE / 'full_detection_v3' / 'evaluation_deep_dive'
FIG_DIR = OUT_DIR / 'figures'

HELD_OUT = {
    '5940843002271361558', '5938229944168473479', '5540715259863557164',
    '6784596629031752949', '5943440598491986019',
}
NHM_VIDEOS = ['38tq2ze5BJE', 'Ch8BkMpJ-Fg', 'EjgVYWh2SAA', 'ml1N7kX2DMQ']

IOU_THRESHOLD = 0.3
SUBTITLE_WINDOW = 10.0  # ±10s for subtitle cross-referencing


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
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


def load_detections():
    """Load all V3+GBT detection JSONs → dict of ep_id → list of dets."""
    all_dets = {}
    for f in sorted(DET_V3GBT_DIR.glob('*_detections.json')):
        ep_id = f.stem.replace('_detections', '')
        with open(f) as fh:
            d = json.load(fh)
        all_dets[ep_id] = d
    print(f"  Loaded {len(all_dets)} episodes, "
          f"{sum(d['n_detections'] for d in all_dets.values())} total detections")
    return all_dets


def load_gbt_scores():
    """Load GBT scores from segment_features.csv.

    Returns dict keyed by (ep_id, start_time, end_time) since the detection_idx
    in segment_features refers to V3 UNFILTERED indices, not the V3+GBT JSON indices.
    """
    df = pd.read_csv(SEG_FEATURES)
    df['episode_id'] = df['episode_id'].astype(str)
    scores = {}
    for _, row in df.iterrows():
        if row.get('v3_gbt_kept', False):
            # Key by time boundaries for reliable matching
            key = (row['episode_id'], round(row['start_time'], 2),
                   round(row['end_time'], 2))
            scores[key] = {
                'gbt_score': row['v3_gbt_score'],
                'max_confidence': row['max_confidence'],
                'mean_confidence': row['mean_confidence'],
                'label': int(row['label']),
            }
    return scores


def load_gt():
    """Load test and auto ground truth annotations."""
    # Test GT
    test = pd.read_csv(TEST_ANN_PATH)
    test['video_id'] = test['video_id'].astype(str)
    test['word'] = test['corresponding_word'].str.lower().str.strip()
    test['source'] = 'test'
    test = test.rename(columns={'video_id': 'episode_id'})

    # Auto GT
    auto = pd.read_csv(AUTO_ANN_PATH)
    auto['video_id'] = auto['video_id'].astype(str)
    auto['word'] = auto['word'].str.lower().str.strip()
    auto['source'] = 'auto'
    auto = auto.rename(columns={'video_id': 'episode_id'})

    # Filter to 249 episodes with detections
    ep_ids = set(f.stem.replace('_detections', '')
                 for f in DET_V3GBT_DIR.glob('*_detections.json'))
    test = test[test['episode_id'].isin(ep_ids)].reset_index(drop=True)
    auto = auto[auto['episode_id'].isin(ep_ids)].reset_index(drop=True)

    print(f"  Test GT: {len(test)} annotations across {test['episode_id'].nunique()} episodes")
    print(f"  Auto GT: {len(auto)} annotations across {auto['episode_id'].nunique()} episodes")
    return test, auto


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: GROUND TRUTH COVERAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def match_gt_to_detections(gt_df, all_dets, gbt_scores, use_overlap=False, threshold=0.3):
    """Match GT annotations to V3+GBT detections using greedy matching.

    Args:
        use_overlap: If True, use overlap (inter/det_dur) metric (for test GT).
                     If False, use IoU metric (for auto GT).
        threshold: Score threshold for matching.
    """
    score_fn = compute_overlap if use_overlap else compute_iou

    # Process episode by episode for greedy matching
    results = []
    for ep_id, ep_gt in gt_df.groupby('episode_id'):
        if ep_id not in all_dets:
            # No detections for this episode — all GT are missed
            for idx, gt_row in ep_gt.iterrows():
                results.append({
                    'annotation_id': idx,
                    'episode_id': ep_id,
                    'gt_start': gt_row['start'],
                    'gt_end': gt_row['end'],
                    'gt_word': gt_row['word'],
                    'gt_duration': gt_row['end'] - gt_row['start'],
                    'gt_source': gt_row['source'],
                    'matched': False,
                    'iou_score': 0,
                    'matched_det_start': np.nan,
                    'matched_det_end': np.nan,
                    'matched_gbt_score': np.nan,
                    'matched_max_prob': np.nan,
                })
            continue

        dets = all_dets[ep_id]['detections']
        gt_list = list(ep_gt.iterrows())  # list of (idx, row)
        n_det = len(dets)
        n_gt = len(gt_list)

        if n_det == 0:
            for idx, gt_row in gt_list:
                results.append({
                    'annotation_id': idx,
                    'episode_id': ep_id,
                    'gt_start': gt_row['start'],
                    'gt_end': gt_row['end'],
                    'gt_word': gt_row['word'],
                    'gt_duration': gt_row['end'] - gt_row['start'],
                    'gt_source': gt_row['source'],
                    'matched': False,
                    'iou_score': 0,
                    'matched_det_start': np.nan,
                    'matched_det_end': np.nan,
                    'matched_gbt_score': np.nan,
                    'matched_max_prob': np.nan,
                })
            continue

        # Build score matrix: (n_det, n_gt)
        score_mat = np.zeros((n_det, n_gt))
        for i, det in enumerate(dets):
            for j, (_, gt_row) in enumerate(gt_list):
                score_mat[i, j] = score_fn(
                    det['start_time'], det['end_time'],
                    gt_row['start'], gt_row['end'])

        # Greedy matching: pick highest score, mark both as matched, repeat
        matched_d, matched_g = set(), set()
        # Also store which detection matched which GT
        gt_to_det = {}  # gt_index → det_index
        while True:
            mask = np.ones_like(score_mat, dtype=bool)
            for i in matched_d:
                mask[i, :] = False
            for j in matched_g:
                mask[:, j] = False
            masked = score_mat * mask
            if masked.max() < threshold:
                break
            i, j = np.unravel_index(masked.argmax(), masked.shape)
            matched_d.add(i)
            matched_g.add(j)
            gt_to_det[j] = i

        # Build result rows
        for j, (idx, gt_row) in enumerate(gt_list):
            gs, ge = gt_row['start'], gt_row['end']
            is_matched = j in matched_g

            row = {
                'annotation_id': idx,
                'episode_id': ep_id,
                'gt_start': gs,
                'gt_end': ge,
                'gt_word': gt_row['word'],
                'gt_duration': ge - gs,
                'gt_source': gt_row['source'],
                'matched': is_matched,
                'iou_score': round(float(score_mat[gt_to_det[j], j]) if is_matched else
                                   float(score_mat[:, j].max()) if n_det > 0 else 0, 4),
            }
            if is_matched:
                di = gt_to_det[j]
                det = dets[di]
                gbt_key = (ep_id, round(det['start_time'], 2),
                           round(det['end_time'], 2))
                gbt_info = gbt_scores.get(gbt_key, {})
                row['matched_det_start'] = det['start_time']
                row['matched_det_end'] = det['end_time']
                row['matched_gbt_score'] = gbt_info.get('gbt_score', np.nan)
                row['matched_max_prob'] = det.get('max_confidence',
                                                   det.get('max_prob', np.nan))
            else:
                row['matched_det_start'] = np.nan
                row['matched_det_end'] = np.nan
                row['matched_gbt_score'] = np.nan
                row['matched_max_prob'] = np.nan

            results.append(row)

    return pd.DataFrame(results)


def step1_coverage(test, auto, all_dets, gbt_scores):
    """Step 1: Ground truth coverage analysis."""
    print("\n" + "="*70)
    print("STEP 1: GROUND TRUTH COVERAGE ANALYSIS")
    print("="*70)

    # Test GT uses overlap >= 0.5, Auto GT uses IoU >= 0.3
    # (consistent with existing evaluation in apply_gbt_filter.py)
    test_cov = match_gt_to_detections(test, all_dets, gbt_scores,
                                       use_overlap=True, threshold=0.5)
    auto_cov = match_gt_to_detections(auto, all_dets, gbt_scores,
                                       use_overlap=False, threshold=0.3)

    # Combine
    auto_cov['annotation_id'] = auto_cov['annotation_id'] + len(test_cov)
    coverage = pd.concat([test_cov, auto_cov], ignore_index=True)

    # Save
    coverage.to_csv(OUT_DIR / 'gt_coverage.csv', index=False)
    print(f"  Saved gt_coverage.csv ({len(coverage)} rows)")

    # Headlines
    test_recall = test_cov['matched'].mean()
    auto_recall = auto_cov['matched'].mean()
    combined_recall = coverage['matched'].mean()

    print(f"\n  Test recall (overlap≥0.5):  {test_recall:.4f} "
          f"({test_cov['matched'].sum()}/{len(test_cov)})")
    print(f"  Auto recall (IoU≥{IOU_THRESHOLD}):     {auto_recall:.4f} "
          f"({auto_cov['matched'].sum()}/{len(auto_cov)})")
    print(f"  Combined recall:           {combined_recall:.4f} "
          f"({coverage['matched'].sum()}/{len(coverage)})")

    return coverage, test_cov, auto_cov


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: MISS PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def step2_miss_patterns(coverage, all_dets):
    """Step 2: Analyse patterns in missed GT annotations."""
    print("\n" + "="*70)
    print("STEP 2: MISS PATTERN ANALYSIS")
    print("="*70)

    hits = coverage[coverage['matched']]
    misses = coverage[~coverage['matched']]
    print(f"  Hits: {len(hits)}, Misses: {len(misses)}")

    report = {}

    # ── 2a: Duration analysis ──
    print("\n  2a. Duration analysis")
    hit_dur = hits['gt_duration']
    miss_dur = misses['gt_duration']
    report['duration'] = {
        'hit_mean': round(hit_dur.mean(), 3),
        'hit_median': round(hit_dur.median(), 3),
        'miss_mean': round(miss_dur.mean(), 3),
        'miss_median': round(miss_dur.median(), 3),
    }
    print(f"    Hit duration:  mean={hit_dur.mean():.3f}s, median={hit_dur.median():.3f}s")
    print(f"    Miss duration: mean={miss_dur.mean():.3f}s, median={miss_dur.median():.3f}s")

    # Figure: hit vs miss duration histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 12, 0.5)
    ax.hist(hit_dur, bins=bins, alpha=0.6, label=f'Hits (n={len(hits)})', color='#2196F3')
    ax.hist(miss_dur, bins=bins, alpha=0.6, label=f'Misses (n={len(misses)})', color='#F44336')
    ax.set_xlabel('GT Annotation Duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Duration Distribution: Hit vs Missed GT Annotations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'hit_miss_duration_histogram.png', dpi=150)
    plt.close(fig)
    print("    → Saved hit_miss_duration_histogram.png")

    # ── 2b: Word-level analysis ──
    print("\n  2b. Word-level analysis")
    miss_words = misses['gt_word'].value_counts()
    top20_missed = miss_words.head(20)
    report['top20_missed_words'] = top20_missed.to_dict()
    print(f"    Top 20 missed words:")
    for word, count in top20_missed.items():
        total = len(coverage[coverage['gt_word'] == word])
        rate = (total - count) / total if total > 0 else 0
        print(f"      {word}: {count} misses / {total} total (recall={rate:.2f})")

    # Miss rate by word length
    coverage_wl = coverage.copy()
    coverage_wl['word_len'] = coverage_wl['gt_word'].str.len()
    bins_wl = [0, 1, 2, 4, 7, 100]
    labels_wl = ['1', '2', '3-4', '5-7', '8+']
    coverage_wl['len_bin'] = pd.cut(coverage_wl['word_len'], bins=bins_wl, labels=labels_wl)
    wl_stats = coverage_wl.groupby('len_bin', observed=False).agg(
        total=('matched', 'count'),
        hits=('matched', 'sum'),
    )
    wl_stats['miss_rate'] = 1 - wl_stats['hits'] / wl_stats['total']
    wl_stats['recall'] = wl_stats['hits'] / wl_stats['total']
    report['miss_rate_by_word_length'] = {
        str(k): {'total': int(v['total']), 'hits': int(v['hits']),
                  'miss_rate': round(float(v['miss_rate']), 4)}
        for k, v in wl_stats.iterrows()
    }
    print(f"\n    Miss rate by word length:")
    for idx, row in wl_stats.iterrows():
        print(f"      {idx} chars: {row['miss_rate']:.3f} miss rate "
              f"({int(row['total']-row['hits'])}/{int(row['total'])})")

    # Figure: miss rate by word length
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(wl_stats)), wl_stats['miss_rate'] * 100,
                  color='#FF7043', edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(wl_stats)))
    ax.set_xticklabels(wl_stats.index)
    ax.set_xlabel('Word Length (characters)')
    ax.set_ylabel('Miss Rate (%)')
    ax.set_title('Miss Rate by Word Length')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, (_, row) in zip(bars, wl_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={int(row["total"])}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'miss_rate_by_word_length.png', dpi=150)
    plt.close(fig)
    print("    → Saved miss_rate_by_word_length.png")

    # ── 2c: Episode/signer analysis ──
    print("\n  2c. Episode-level analysis")
    ep_stats = coverage.groupby('episode_id').agg(
        total=('matched', 'count'),
        hits=('matched', 'sum'),
    )
    ep_stats['recall'] = ep_stats['hits'] / ep_stats['total']
    ep_stats['miss_count'] = ep_stats['total'] - ep_stats['hits']
    ep_stats = ep_stats.sort_values('recall')

    worst10 = ep_stats.head(10)
    report['worst_10_episodes'] = {
        ep: {'recall': round(float(r['recall']), 3), 'total': int(r['total']),
             'misses': int(r['miss_count'])}
        for ep, r in worst10.iterrows()
    }
    print(f"    Worst 10 episodes by recall:")
    for ep, row in worst10.iterrows():
        held = " [HELD-OUT]" if ep in HELD_OUT else ""
        print(f"      {ep}: recall={row['recall']:.3f} "
              f"({int(row['hits'])}/{int(row['total'])}){held}")

    # Consecutive miss analysis
    print("\n    Consecutive miss clusters:")
    miss_clusters = []
    for ep_id, group in misses.groupby('episode_id'):
        times = sorted(group['gt_start'].values)
        if len(times) < 2:
            continue
        cluster = [times[0]]
        for t in times[1:]:
            if t - cluster[-1] < 3.0:  # within 3s
                cluster.append(t)
            else:
                if len(cluster) >= 2:
                    miss_clusters.append({
                        'episode_id': ep_id,
                        'cluster_size': len(cluster),
                        'start': cluster[0],
                        'end': cluster[-1],
                    })
                cluster = [t]
        if len(cluster) >= 2:
            miss_clusters.append({
                'episode_id': ep_id,
                'cluster_size': len(cluster),
                'start': cluster[0],
                'end': cluster[-1],
            })
    report['miss_clusters'] = len(miss_clusters)
    print(f"    Found {len(miss_clusters)} miss clusters (≥2 misses within 3s)")

    # Figure: per-episode recall distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ep_stats['recall'], bins=20, color='#4CAF50', edgecolor='black', linewidth=0.5)
    ax.axvline(ep_stats['recall'].mean(), color='red', linestyle='--',
               label=f'Mean={ep_stats["recall"].mean():.3f}')
    ax.axvline(ep_stats['recall'].median(), color='blue', linestyle='--',
               label=f'Median={ep_stats["recall"].median():.3f}')
    ax.set_xlabel('Per-Episode Recall (IoU≥0.3)')
    ax.set_ylabel('Number of Episodes')
    ax.set_title('Distribution of Per-Episode Recall (V3+GBT, 249 Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'per_episode_recall_distribution.png', dpi=150)
    plt.close(fig)
    print("    → Saved per_episode_recall_distribution.png")

    # ── 2d: Temporal analysis ──
    print("\n  2d. Temporal analysis")
    # Normalised position within episode
    for _, row in coverage.iterrows():
        ep_id = row['episode_id']
        if ep_id in all_dets:
            ep_dur = all_dets[ep_id].get('duration_sec', 1)
            coverage.loc[_, 'norm_position'] = row['gt_start'] / ep_dur if ep_dur > 0 else 0

    if 'norm_position' in coverage.columns:
        hit_pos = coverage[coverage['matched']]['norm_position'].dropna()
        miss_pos = coverage[~coverage['matched']]['norm_position'].dropna()
        report['temporal'] = {
            'hit_mean_pos': round(float(hit_pos.mean()), 3),
            'miss_mean_pos': round(float(miss_pos.mean()), 3),
        }
        print(f"    Hit mean normalised position:  {hit_pos.mean():.3f}")
        print(f"    Miss mean normalised position: {miss_pos.mean():.3f}")

    # ── 2e: Near-miss analysis ──
    print("\n  2e. Near-miss analysis")
    near_miss_count = 0
    complete_miss_count = 0
    near_miss_details = []
    for _, row in misses.iterrows():
        ep_id = row['episode_id']
        gs, ge = row['gt_start'], row['gt_end']
        gt_mid = (gs + ge) / 2

        nearest_dist = float('inf')
        nearest_iou = 0
        nearest_det = None
        if ep_id in all_dets:
            for det in all_dets[ep_id]['detections']:
                ds, de = det['start_time'], det['end_time']
                det_mid = (ds + de) / 2
                dist = abs(det_mid - gt_mid)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_iou = compute_iou(ds, de, gs, ge)
                    nearest_det = det

        if nearest_dist <= 2.0:
            near_miss_count += 1
            near_miss_details.append({
                'episode_id': ep_id,
                'gt_word': row['gt_word'],
                'gt_start': gs,
                'gt_end': ge,
                'nearest_dist': round(nearest_dist, 2),
                'nearest_iou': round(nearest_iou, 4),
                'det_start': nearest_det['start_time'] if nearest_det else np.nan,
                'det_end': nearest_det['end_time'] if nearest_det else np.nan,
            })
        else:
            complete_miss_count += 1

    total_misses = len(misses)
    report['near_miss'] = {
        'near_misses': near_miss_count,
        'complete_misses': complete_miss_count,
        'total_misses': total_misses,
        'near_miss_pct': round(near_miss_count / total_misses * 100, 1) if total_misses > 0 else 0,
        'complete_miss_pct': round(complete_miss_count / total_misses * 100, 1) if total_misses > 0 else 0,
    }
    print(f"    Near-misses (det within ±2s): {near_miss_count} ({near_miss_count/total_misses*100:.1f}%)")
    print(f"    Complete misses (no det nearby): {complete_miss_count} ({complete_miss_count/total_misses*100:.1f}%)")

    if near_miss_details:
        pd.DataFrame(near_miss_details).to_csv(OUT_DIR / 'near_miss_details.csv', index=False)

    # Figure: near miss vs complete miss
    fig, ax = plt.subplots(figsize=(7, 7))
    sizes = [near_miss_count, complete_miss_count]
    labels_pie = [
        f'Near-miss\n(det ≤2s away)\n{near_miss_count} ({near_miss_count/total_misses*100:.1f}%)',
        f'Complete miss\n(no det nearby)\n{complete_miss_count} ({complete_miss_count/total_misses*100:.1f}%)',
    ]
    colors = ['#FFB74D', '#E57373']
    ax.pie(sizes, labels=labels_pie, colors=colors, autopct='', startangle=90,
           textprops={'fontsize': 11})
    ax.set_title(f'Miss Classification (n={total_misses})')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'near_miss_vs_complete_miss.png', dpi=150)
    plt.close(fig)
    print("    → Saved near_miss_vs_complete_miss.png")

    return report


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: SUBTITLE CROSS-REFERENCING (BOBSL)
# ══════════════════════════════════════════════════════════════════════════

def parse_vtt(path):
    """Parse a WebVTT file → list of (start_sec, end_sec, text)."""
    blocks = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            parts = line.split('-->')
            start = _parse_ts(parts[0].strip())
            end_part = parts[1].strip().split()[0]  # strip position metadata
            end = _parse_ts(end_part)
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1
            text = ' '.join(text_lines)
            if text:
                blocks.append((start, end, text))
        else:
            i += 1
    return blocks


def _parse_ts(ts):
    ts = ts.strip()
    if ts.count(':') == 2:
        h, m, rest = ts.split(':')
        s = rest.replace(',', '.')
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif ts.count(':') == 1:
        m, rest = ts.split(':')
        s = rest.replace(',', '.')
        return int(m) * 60 + float(s)
    return float(ts)


def run_inverted_pipeline_on_all(episode_ids):
    """Run inverted subtitle mining pipeline on all episodes.
    Returns list of candidate dicts: {episode_id, word, sub_start, sub_end, confidence}.
    """
    import spacy
    nlp = spacy.load('en_core_web_sm')

    # Load sign vocabulary
    sign_vocab = set()
    with open(SIGN_VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            sign_vocab.add(row['word'].lower().strip())

    # Build subtitle frequency from all BOBSL subtitles
    print("    Building subtitle frequency index...")
    freq = Counter()
    for fn in os.listdir(SUBTITLES_DIR):
        if not fn.endswith('.vtt'):
            continue
        with open(SUBTITLES_DIR / fn) as f:
            for line in f:
                line = line.strip()
                if '-->' in line or line.startswith('WEBVTT') or not line:
                    continue
                for w in re.findall(r'[a-zA-Z]+', line):
                    freq[w.lower()] += 1

    N = 3000
    top_n_words = set(w for w, _ in freq.most_common(N))
    print(f"    Sign vocab: {len(sign_vocab)} words, Top-{N}: {len(top_n_words)} words")

    NER_STOPWORDS = {'the', 'a', 'an', 'of', 'in', 'to', 'for', 'and',
                     'but', 'or', 'is', 'it', 'its', 'he', 'she', 'we',
                     'they', 'this', 'that', 'with', 'on', 'at', 'by',
                     'as', 'be', 'was', 'are', 'were', 'been', 'has',
                     'had', 'have', 'do', 'did', 'does', 'so', 'no',
                     'not', 'if', 'up', 'out', 'all', 'my', 'our'}

    all_candidates = []
    no_vtt = 0

    for i, ep_id in enumerate(episode_ids):
        if (i + 1) % 50 == 0:
            print(f"    Processing episode {i+1}/{len(episode_ids)}...")

        vtt_path = SUBTITLES_DIR / f'{ep_id}.vtt'
        if not vtt_path.exists():
            no_vtt += 1
            continue

        blocks = parse_vtt(vtt_path)
        for start, end, text in blocks:
            clean = re.sub(r'<[^>]+>', '', text)
            clean = re.sub(r'\[.*?\]', '', clean)

            raw_tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clean)
            words = []
            for tok in raw_tokens:
                base = re.sub(r"'[a-z]+$", '', tok.lower())
                if base and len(base) >= 2:
                    words.append(base)

            if not words:
                continue

            # NER
            doc = nlp(clean)
            ner_spans = {}
            for ent in doc.ents:
                if ent.label_ in ('PERSON', 'GPE', 'ORG', 'FAC', 'NORP', 'LOC',
                                  'EVENT', 'WORK_OF_ART', 'LAW', 'PRODUCT'):
                    for token in ent:
                        w = token.text.lower().strip()
                        if w and re.match(r'^[a-z]+$', w) and len(w) >= 2 and w not in NER_STOPWORDS:
                            ner_spans[w] = ent.label_

            for word in words:
                if word in sign_vocab:
                    continue
                if word in ner_spans:
                    all_candidates.append({
                        'episode_id': ep_id,
                        'word': word,
                        'sub_start': start,
                        'sub_end': end,
                        'confidence': 'HIGH',
                        'reason': 'ner',
                        'ner_type': ner_spans[word],
                    })
                    continue
                if word in top_n_words:
                    continue
                all_candidates.append({
                    'episode_id': ep_id,
                    'word': word,
                    'sub_start': start,
                    'sub_end': end,
                    'confidence': 'MEDIUM',
                    'reason': 'rare',
                    'ner_type': '',
                })

    print(f"    Episodes without VTT: {no_vtt}")
    print(f"    Total inverted candidates: {len(all_candidates)}")
    return all_candidates


def cross_reference_detections(all_dets, coverage_df, sub_candidates):
    """Cross-reference each V3+GBT detection with GT and subtitle candidates."""
    # Build subtitle lookup: ep_id → list of (sub_mid, word)
    sub_by_ep = defaultdict(list)
    for c in sub_candidates:
        mid = (c['sub_start'] + c['sub_end']) / 2
        ep_str = str(c['episode_id']).split('.')[0]  # handle int/float from CSV
        sub_by_ep[ep_str].append({
            'mid': mid,
            'word': c['word'],
            'confidence': c['confidence'],
        })

    # Build GT match set from coverage (matched detections)
    gt_matched_dets = set()  # (ep_id, det_start, det_end)
    matched_gt_words = {}    # (ep_id, det_start, det_end) → gt_word
    for _, row in coverage_df[coverage_df['matched']].iterrows():
        if not np.isnan(row['matched_det_start']):
            key = (row['episode_id'], row['matched_det_start'], row['matched_det_end'])
            gt_matched_dets.add(key)
            matched_gt_words[key] = row['gt_word']

    results = []
    for ep_id, ep_data in all_dets.items():
        for di, det in enumerate(ep_data['detections']):
            ds, de = det['start_time'], det['end_time']
            det_mid = (ds + de) / 2
            det_key = (ep_id, ds, de)

            # Check GT match
            is_gt_matched = det_key in gt_matched_dets
            gt_word = matched_gt_words.get(det_key, '')

            # Check subtitle corroboration
            sub_word = ''
            sub_corroborated = False
            if ep_id in sub_by_ep:
                for sc in sub_by_ep[ep_id]:
                    if abs(sc['mid'] - det_mid) <= SUBTITLE_WINDOW:
                        sub_corroborated = True
                        sub_word = sc['word']
                        break

            if is_gt_matched:
                category = 'gt_matched'
            elif sub_corroborated:
                category = 'sub_corroborated'
            else:
                category = 'uncorroborated'

            results.append({
                'episode_id': ep_id,
                'det_start': ds,
                'det_end': de,
                'gbt_score': det.get('gbt_score', np.nan),
                'max_prob': det.get('max_confidence', det.get('max_prob', np.nan)),
                'frame_confidence': det.get('frame_confidence',
                                            det.get('mean_prob', np.nan)),
                'category': category,
                'matched_gt_word': gt_word,
                'matched_sub_word': sub_word,
            })

    return pd.DataFrame(results)


def step3_subtitle_crossref(all_dets, coverage, gbt_scores):
    """Step 3: Detection validation via subtitle cross-referencing."""
    print("\n" + "="*70)
    print("STEP 3: DETECTION VALIDATION VIA SUBTITLE CROSS-REFERENCING")
    print("="*70)

    episode_ids = sorted(all_dets.keys())

    # Run inverted pipeline (or load cached results)
    cached_path = OUT_DIR / 'bobsl_inverted_candidates.csv'
    if cached_path.exists():
        print("  Loading cached inverted candidates...")
        sub_df = pd.read_csv(cached_path)
        sub_candidates = sub_df.to_dict('records')
        print(f"  Loaded {len(sub_candidates)} cached candidates")
    else:
        print("  Running inverted pipeline on 249 BOBSL episodes (N=3000)...")
        t0 = time.time()
        sub_candidates = run_inverted_pipeline_on_all(episode_ids)
        t1 = time.time()
        print(f"  Pipeline completed in {t1-t0:.1f}s")
        pd.DataFrame(sub_candidates).to_csv(cached_path, index=False)
        print(f"  Saved bobsl_inverted_candidates.csv ({len(sub_candidates)} candidates)")

    # Cross-reference
    print("  Cross-referencing detections with GT + subtitles...")
    validation = cross_reference_detections(all_dets, coverage, sub_candidates)

    # Enrich with GBT scores from segment_features (keyed by time boundaries)
    for ep_id in all_dets:
        for det in all_dets[ep_id]['detections']:
            gbt_key = (ep_id, round(det['start_time'], 2),
                       round(det['end_time'], 2))
            if gbt_key in gbt_scores:
                mask = ((validation['episode_id'] == ep_id) &
                        (validation['det_start'] == det['start_time']) &
                        (validation['det_end'] == det['end_time']))
                validation.loc[mask, 'gbt_score'] = gbt_scores[gbt_key]['gbt_score']

    validation.to_csv(OUT_DIR / 'detection_validation.csv', index=False)
    print(f"  Saved detection_validation.csv ({len(validation)} detections)")

    # Summary
    cats = validation['category'].value_counts()
    total = len(validation)
    report = {}
    for cat in ['gt_matched', 'sub_corroborated', 'uncorroborated']:
        n = cats.get(cat, 0)
        pct = n / total * 100
        report[cat] = {'count': int(n), 'pct': round(pct, 1)}
        print(f"    {cat}: {n} ({pct:.1f}%)")

    # GT word vs subtitle word agreement
    gt_matched_with_sub = validation[
        (validation['category'] == 'gt_matched') &
        (validation['matched_sub_word'].fillna('') != '') &
        (validation['matched_gt_word'].fillna('') != '')
    ]
    if len(gt_matched_with_sub) > 0:
        agrees = sum(
            str(row['matched_sub_word']).lower() in str(row['matched_gt_word']).lower() or
            str(row['matched_gt_word']).lower() in str(row['matched_sub_word']).lower()
            for _, row in gt_matched_with_sub.iterrows()
        )
        report['word_agreement'] = {
            'gt_with_sub': len(gt_matched_with_sub),
            'agrees': agrees,
            'agreement_rate': round(agrees / len(gt_matched_with_sub), 3),
        }
        print(f"\n    GT↔Subtitle word agreement: {agrees}/{len(gt_matched_with_sub)} "
              f"({agrees/len(gt_matched_with_sub)*100:.1f}%)")

    # Top sub_corroborated words
    sub_corr = validation[validation['category'] == 'sub_corroborated']
    if len(sub_corr) > 0:
        top_sub_words = sub_corr['matched_sub_word'].value_counts().head(20)
        report['top_sub_corroborated_words'] = top_sub_words.to_dict()
        print(f"\n    Top 20 sub-corroborated words:")
        for word, count in top_sub_words.items():
            print(f"      {word}: {count}")

    # Figure: detection validation pie
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [cats.get('gt_matched', 0), cats.get('sub_corroborated', 0),
             cats.get('uncorroborated', 0)]
    labels_pie = [
        f'GT-matched\n{sizes[0]} ({sizes[0]/total*100:.1f}%)',
        f'Sub-corroborated\n{sizes[1]} ({sizes[1]/total*100:.1f}%)',
        f'Uncorroborated\n{sizes[2]} ({sizes[2]/total*100:.1f}%)',
    ]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    ax.pie(sizes, labels=labels_pie, colors=colors, autopct='', startangle=90,
           textprops={'fontsize': 12})
    ax.set_title(f'V3+GBT Detection Validation (n={total})')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'detection_validation_pie.png', dpi=150)
    plt.close(fig)
    print("    → Saved detection_validation_pie.png")

    # Figure: validation by GBT score
    if validation['gbt_score'].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for cat, color in [('gt_matched', '#4CAF50'), ('sub_corroborated', '#2196F3'),
                           ('uncorroborated', '#FF9800')]:
            subset = validation[(validation['category'] == cat) &
                               (validation['gbt_score'].notna())]
            if len(subset) > 0:
                ax.hist(subset['gbt_score'], bins=30, alpha=0.5, label=f'{cat} (n={len(subset)})',
                        color=color, density=True)
        ax.set_xlabel('GBT Score')
        ax.set_ylabel('Density')
        ax.set_title('GBT Score Distribution by Validation Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'validation_by_gbt_score.png', dpi=150)
        plt.close(fig)
        print("    → Saved validation_by_gbt_score.png")

    return report, validation, sub_candidates


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: NHM DETECTION SUBTITLE CROSS-REFERENCING
# ══════════════════════════════════════════════════════════════════════════

def step4_nhm(all_dets_validation=None):
    """Step 4: Cross-reference NHM V3+GBT detections with old subtitle candidates."""
    print("\n" + "="*70)
    print("STEP 4: NHM DETECTION SUBTITLE CROSS-REFERENCING")
    print("="*70)

    results = []
    nhm_summary = {}

    for vid in NHM_VIDEOS:
        # Load V3+GBT detections
        det_path = NHM_DET_DIR / vid / 'detections_v3_gbt.json'
        if not det_path.exists():
            print(f"  {vid}: No detection file found, skipping")
            continue

        with open(det_path) as f:
            det_data = json.load(f)
        dets = det_data['detections']

        # Load old subtitle candidates
        cand_path = NHM_OLD_OUTPUT / vid / 'scored_candidates.csv'
        sub_candidates = []
        if cand_path.exists():
            with open(cand_path) as f:
                for row in csv.DictReader(f):
                    sub_candidates.append({
                        'sub_start': float(row['sub_start']),
                        'sub_end': float(row['sub_end']),
                        'word': row['word'].lower(),
                        'tier': row.get('tier', ''),
                    })

        print(f"  {vid}: {len(dets)} detections, {len(sub_candidates)} subtitle candidates")

        sub_corr = 0
        uncorr = 0
        for det in dets:
            ds, de = det['start_time'], det['end_time']
            det_mid = (ds + de) / 2

            matched_word = ''
            is_corr = False
            for sc in sub_candidates:
                sc_mid = (sc['sub_start'] + sc['sub_end']) / 2
                if abs(sc_mid - det_mid) <= SUBTITLE_WINDOW:
                    is_corr = True
                    matched_word = sc['word']
                    break

            cat = 'sub_corroborated' if is_corr else 'uncorroborated'
            if is_corr:
                sub_corr += 1
            else:
                uncorr += 1

            results.append({
                'video_id': vid,
                'det_start': ds,
                'det_end': de,
                'gbt_score': det.get('gbt_score', np.nan),
                'max_prob': det.get('max_prob', np.nan),
                'category': cat,
                'matched_sub_word': matched_word,
            })

        nhm_summary[vid] = {
            'n_dets': len(dets),
            'sub_corroborated': sub_corr,
            'uncorroborated': uncorr,
            'corr_pct': round(sub_corr / len(dets) * 100, 1) if dets else 0,
        }
        print(f"    Sub-corroborated: {sub_corr} ({sub_corr/len(dets)*100:.1f}%)")
        print(f"    Uncorroborated: {uncorr} ({uncorr/len(dets)*100:.1f}%)")

    nhm_df = pd.DataFrame(results)
    nhm_df.to_csv(OUT_DIR / 'nhm_detection_validation.csv', index=False)
    print(f"  Saved nhm_detection_validation.csv ({len(nhm_df)} rows)")

    if len(nhm_df) > 0:
        total = len(nhm_df)
        cats = nhm_df['category'].value_counts()
        print(f"\n  NHM Overall:")
        for cat in ['sub_corroborated', 'uncorroborated']:
            n = cats.get(cat, 0)
            print(f"    {cat}: {n} ({n/total*100:.1f}%)")

        # Top corroborated words
        corr = nhm_df[nhm_df['category'] == 'sub_corroborated']
        if len(corr) > 0:
            top_words = corr['matched_sub_word'].value_counts().head(20)
            print(f"\n  Top NHM sub-corroborated words:")
            for word, count in top_words.items():
                print(f"    {word}: {count}")

    return nhm_summary, nhm_df


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════

def step5_report(test_cov, auto_cov, coverage, miss_report, sub_report,
                 validation_df, nhm_summary, nhm_df):
    """Step 5: Generate comprehensive markdown report."""
    print("\n" + "="*70)
    print("STEP 5: GENERATING REPORT")
    print("="*70)

    test_recall = test_cov['matched'].mean()
    auto_recall = auto_cov['matched'].mean()
    total_dets = len(validation_df) if validation_df is not None else 0

    report_lines = []
    def w(line=''):
        report_lines.append(line)

    w("# V3+GBT Detection Evaluation Deep-Dive")
    w()
    w(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}")
    w(f"**Detector:** V3 (variant3_75_25.pt) + V3 GBT (gbt_v3.pkl)")
    w(f"**Dataset:** 249 BOBSL episodes")
    w(f"**IoU threshold:** {IOU_THRESHOLD}")
    w()

    # ── Section 1: Detection Coverage ──
    w("## Section 1: Detection Coverage")
    w()
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Total V3+GBT detections | {total_dets:,} |")
    w(f"| Test GT annotations | {len(test_cov):,} |")
    w(f"| Auto GT annotations | {len(auto_cov):,} |")
    w(f"| **Test recall (overlap≥0.5)** | **{test_recall:.4f}** ({test_cov['matched'].sum()}/{len(test_cov)}) |")
    w(f"| **Auto recall (IoU≥0.3)** | **{auto_recall:.4f}** ({auto_cov['matched'].sum()}/{len(auto_cov)}) |")
    w(f"| Combined recall (mixed) | {coverage['matched'].mean():.4f} ({coverage['matched'].sum()}/{len(coverage)}) |")
    w()

    # Held-out vs non-held-out
    held_out_test = test_cov[test_cov['episode_id'].isin(HELD_OUT)]
    non_held_test = test_cov[~test_cov['episode_id'].isin(HELD_OUT)]
    if len(held_out_test) > 0:
        w(f"| Held-out test recall | {held_out_test['matched'].mean():.4f} ({held_out_test['matched'].sum()}/{len(held_out_test)}) |")
    if len(non_held_test) > 0:
        w(f"| Non-held-out test recall | {non_held_test['matched'].mean():.4f} ({non_held_test['matched'].sum()}/{len(non_held_test)}) |")
    w()

    # ── Section 2: Miss Patterns ──
    w("## Section 2: Miss Patterns")
    w()

    # Duration
    w("### 2.1 Duration Patterns")
    w()
    if 'duration' in miss_report:
        d = miss_report['duration']
        w(f"| | Mean | Median |")
        w(f"|--|------|--------|")
        w(f"| Hit duration | {d['hit_mean']}s | {d['hit_median']}s |")
        w(f"| Miss duration | {d['miss_mean']}s | {d['miss_median']}s |")
        w()
        w("![Duration histogram](figures/hit_miss_duration_histogram.png)")
        w()

    # Top 20 missed words
    w("### 2.2 Top 20 Most-Missed Words")
    w()
    if 'top20_missed_words' in miss_report:
        w("| Word | Misses | Total | Recall |")
        w("|------|--------|-------|--------|")
        for word, count in miss_report['top20_missed_words'].items():
            total_w = len(coverage[coverage['gt_word'] == word])
            recall_w = (total_w - count) / total_w if total_w > 0 else 0
            w(f"| {word} | {count} | {total_w} | {recall_w:.3f} |")
        w()

    # Word length
    w("### 2.3 Miss Rate by Word Length")
    w()
    if 'miss_rate_by_word_length' in miss_report:
        w("| Length | Total | Misses | Miss Rate |")
        w("|--------|-------|--------|-----------|")
        for length, stats in miss_report['miss_rate_by_word_length'].items():
            misses_n = stats['total'] - stats['hits']
            w(f"| {length} chars | {stats['total']} | {misses_n} | {stats['miss_rate']:.4f} |")
        w()
        w("![Miss rate by word length](figures/miss_rate_by_word_length.png)")
        w()

    # Episode analysis
    w("### 2.4 Per-Episode Recall")
    w()
    if 'worst_10_episodes' in miss_report:
        w("**Worst 10 episodes:**")
        w()
        w("| Episode ID | Recall | Total | Misses | Held-out? |")
        w("|-----------|--------|-------|--------|-----------|")
        for ep, stats in miss_report['worst_10_episodes'].items():
            held = "Yes" if ep in HELD_OUT else "No"
            w(f"| {ep} | {stats['recall']:.3f} | {stats['total']} | {stats['misses']} | {held} |")
        w()
    w("![Per-episode recall](figures/per_episode_recall_distribution.png)")
    w()

    if 'miss_clusters' in miss_report:
        w(f"**Consecutive miss clusters** (≥2 misses within 3s): {miss_report['miss_clusters']}")
        w()

    # Near-miss
    w("### 2.5 Near-Miss vs Complete Miss")
    w()
    if 'near_miss' in miss_report:
        nm = miss_report['near_miss']
        w(f"| Category | Count | % of Misses |")
        w(f"|----------|-------|-------------|")
        w(f"| Near-miss (det ≤2s) | {nm['near_misses']} | {nm['near_miss_pct']}% |")
        w(f"| Complete miss | {nm['complete_misses']} | {nm['complete_miss_pct']}% |")
        w(f"| **Total misses** | **{nm['total_misses']}** | **100%** |")
        w()
    w("![Near miss breakdown](figures/near_miss_vs_complete_miss.png)")
    w()

    # Temporal
    w("### 2.6 Temporal Patterns")
    w()
    if 'temporal' in miss_report:
        t = miss_report['temporal']
        w(f"- Hit mean normalised position: {t['hit_mean_pos']}")
        w(f"- Miss mean normalised position: {t['miss_mean_pos']}")
        w()

    # ── Section 3: Detection Validation ──
    w("## Section 3: Detection Validation")
    w()

    if validation_df is not None and len(validation_df) > 0:
        cats = validation_df['category'].value_counts()
        total = len(validation_df)

        w("### 3.1 BOBSL Detection Categories")
        w()
        w("| Category | Count | % |")
        w("|----------|-------|---|")
        for cat in ['gt_matched', 'sub_corroborated', 'uncorroborated']:
            n = cats.get(cat, 0)
            w(f"| {cat} | {n:,} | {n/total*100:.1f}% |")
        w(f"| **Total** | **{total:,}** | **100%** |")
        w()
        w("![Detection validation](figures/detection_validation_pie.png)")
        w()

        if 'word_agreement' in sub_report:
            wa = sub_report['word_agreement']
            w(f"**GT↔Subtitle word agreement:** {wa['agrees']}/{wa['gt_with_sub']} "
              f"({wa['agreement_rate']*100:.1f}%) of GT-matched detections with subtitle overlap "
              f"have matching words.")
            w()

        if 'top_sub_corroborated_words' in sub_report:
            w("### 3.2 Top Sub-Corroborated Words (potential new FS vocabulary)")
            w()
            w("| Word | Count |")
            w("|------|-------|")
            for word, count in list(sub_report['top_sub_corroborated_words'].items())[:20]:
                w(f"| {word} | {count} |")
            w()

        w("![Validation by GBT score](figures/validation_by_gbt_score.png)")
        w()

    # NHM
    w("### 3.3 NHM Detection Validation")
    w()
    if nhm_summary:
        w("| Video | Detections | Sub-corr | Uncorr | Corr % |")
        w("|-------|-----------|----------|--------|--------|")
        total_nhm = 0
        total_corr = 0
        for vid, stats in nhm_summary.items():
            w(f"| {vid} | {stats['n_dets']} | {stats['sub_corroborated']} | "
              f"{stats['uncorroborated']} | {stats['corr_pct']}% |")
            total_nhm += stats['n_dets']
            total_corr += stats['sub_corroborated']
        if total_nhm > 0:
            w(f"| **Total** | **{total_nhm}** | **{total_corr}** | "
              f"**{total_nhm-total_corr}** | **{total_corr/total_nhm*100:.1f}%** |")
        w()

        if nhm_df is not None and len(nhm_df) > 0:
            corr = nhm_df[nhm_df['category'] == 'sub_corroborated']
            if len(corr) > 0:
                top_words = corr['matched_sub_word'].value_counts().head(15)
                w("**Top NHM sub-corroborated words:**")
                w()
                w("| Word | Count |")
                w("|------|-------|")
                for word, count in top_words.items():
                    w(f"| {word} | {count} |")
                w()

    # ── Section 4: Implications ──
    w("## Section 4: Implications")
    w()

    if validation_df is not None and len(validation_df) > 0:
        cats = validation_df['category'].value_counts()
        total = len(validation_df)
        gt_n = cats.get('gt_matched', 0)
        sub_n = cats.get('sub_corroborated', 0)
        uncorr_n = cats.get('uncorroborated', 0)

        est_tp = gt_n + sub_n
        w(f"### Estimated True Positive Rate")
        w()
        w(f"- GT-confirmed TPs: {gt_n:,} ({gt_n/total*100:.1f}%)")
        w(f"- Subtitle-corroborated (likely TPs): {sub_n:,} ({sub_n/total*100:.1f}%)")
        w(f"- **Lower-bound TP estimate:** {est_tp:,} ({est_tp/total*100:.1f}%)")
        w(f"- Uncorroborated (unknown): {uncorr_n:,} ({uncorr_n/total*100:.1f}%)")
        w()
        w("Note: Uncorroborated detections include (a) true fingerspelling during "
          "non-subtitled moments, (b) fingerspelling of common words filtered by the "
          "inverted pipeline, and (c) false positives. The actual TP rate is likely "
          "between the lower bound above and 100%.")
        w()

    w("### Fixable vs Inherent Miss Patterns")
    w()
    if 'near_miss' in miss_report:
        nm = miss_report['near_miss']
        w(f"1. **Near-misses ({nm['near_misses']} = {nm['near_miss_pct']}% of misses):** "
          "Detector fires but boundaries are off. Could be improved by:")
        w("   - Boundary extension tuning")
        w("   - Adaptive gap bridging")
        w("   - Post-hoc boundary refinement using frame probabilities")
        w()
        w(f"2. **Complete misses ({nm['complete_misses']} = {nm['complete_miss_pct']}% of misses):** "
          "Detector doesn't fire at all. Causes include:")
        w("   - Very short events (below minimum duration threshold)")
        w("   - Low-confidence frame probabilities (below threshold)")
        w("   - Domain gap (unusual signing style, context)")
        w()

    w("### Recommendations for Clip Extraction")
    w()
    w("1. **Extract all GT-matched detections** — confirmed fingerspelling")
    w("2. **Extract sub-corroborated detections** — high probability of being correct, "
      "and subtitle provides candidate word label")
    w("3. **Apply GBT score threshold to uncorroborated** — extract only high-confidence "
      "uncorroborated detections (gbt_score ≥ 0.95)")
    w("4. **Skip very short uncorroborated detections** (< 0.8s) — higher FP rate")
    w()

    report_text = '\n'.join(report_lines)
    report_path = BASE / 'full_detection_v3' / 'EVALUATION_DEEP_DIVE.md'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  Saved {report_path}")

    return report_text


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("V3+GBT Detection Evaluation Deep-Dive")
    print("="*70)

    # Load data
    print("\nLoading data...")
    all_dets = load_detections()
    gbt_scores = load_gbt_scores()
    test, auto = load_gt()

    # Step 1: GT coverage
    coverage, test_cov, auto_cov = step1_coverage(test, auto, all_dets, gbt_scores)

    # Step 2: Miss patterns
    miss_report = step2_miss_patterns(coverage, all_dets)

    # Step 3: Subtitle cross-referencing
    sub_report, validation_df, sub_candidates = step3_subtitle_crossref(
        all_dets, coverage, gbt_scores)

    # Step 4: NHM
    nhm_summary, nhm_df = step4_nhm()

    # Step 5: Report
    step5_report(test_cov, auto_cov, coverage, miss_report, sub_report,
                 validation_df, nhm_summary, nhm_df)

    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
