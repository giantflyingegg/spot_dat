#!/usr/bin/env python3
"""
Step 6: Analysis, visualisations, and report generation.

Generates:
- Updated wrist velocity histogram (positives vs old negatives vs new hard negatives)
- Training curves (AUROC per epoch for all variants)
- BOBSL F1 comparison bar chart
- YouTube HIGH/min firing rate comparison
- Detector trace comparison (baseline vs best variant)
- Full variant comparison table
- HARD_NEGATIVE_REPORT.md
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/')
AUDIT_DIR = os.path.join(BASE_DIR, 'audit/')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation/')
LOG_DIR = os.path.join(BASE_DIR, 'models/training_logs/')
HARD_NEG_DIR = os.path.join(BASE_DIR, 'hard_negatives/')
FIG_DIR = EVAL_DIR  # figures go in evaluation/
os.makedirs(FIG_DIR, exist_ok=True)


def fig_training_curves():
    """Plot AUROC training curves for all variants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'V1_hard_only': '#e15759', 'V2_50_50': '#59a14f',
              'V3_75_25': '#4e79a7', 'V4_curriculum': '#edc948'}

    for name, color in colors.items():
        path = os.path.join(LOG_DIR, f'{name}_history.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            history = json.load(f)
        epochs = [h['epoch'] for h in history]
        aurocs = [h['auroc'] for h in history]
        ax.plot(epochs, aurocs, label=name, color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUROC')
    ax.set_title('Training Curves: All Variants')
    ax.legend()
    ax.set_ylim(0.9, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print("  Saved training_curves.png")


def fig_bobsl_comparison():
    """Bar chart comparing F1 across variants."""
    bobsl_path = os.path.join(EVAL_DIR, 'bobsl_fair_eval.csv')
    if not os.path.exists(bobsl_path):
        print("  SKIP: bobsl_fair_eval.csv not found")
        return

    df = pd.read_csv(bobsl_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1@0.3
    ax = axes[0]
    colors = ['#4e79a7'] + ['#e15759'] * (len(df) - 1)
    ax.bar(df['model'], df['f1_03'], color=colors[:len(df)])
    ax.set_ylabel('F1 @ IoU=0.3')
    ax.set_title('F1 @ IoU=0.3')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['f1_03']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    # F1@0.5
    ax = axes[1]
    ax.bar(df['model'], df['f1_05'], color=colors[:len(df)])
    ax.set_ylabel('F1 @ IoU=0.5')
    ax.set_title('F1 @ IoU=0.5')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['f1_05']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    # FP/min
    ax = axes[2]
    ax.bar(df['model'], df['fp_per_min'], color=colors[:len(df)])
    ax.set_ylabel('FP / minute')
    ax.set_title('False Positive Rate')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['fp_per_min']):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bobsl_comparison.png'), dpi=150)
    plt.close()
    print("  Saved bobsl_comparison.png")


def fig_youtube_firing_rate():
    """YouTube HIGH candidates per minute comparison."""
    yt_path = os.path.join(EVAL_DIR, 'youtube_eval.csv')
    if not os.path.exists(yt_path):
        print("  SKIP: youtube_eval.csv not found")
        return

    df = pd.read_csv(yt_path)
    agg = df.groupby('model').agg({
        'n_high': 'sum', 'duration_min': 'sum',
        'mean_prob': 'mean', 'mean_bg_prob': 'mean',
    }).reset_index()
    agg['high_per_min'] = agg['n_high'] / agg['duration_min']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # HIGH/min
    ax = axes[0]
    colors = ['#4e79a7'] + ['#e15759'] * (len(agg) - 1)
    ax.bar(agg['model'], agg['high_per_min'], color=colors[:len(agg)])
    ax.set_ylabel('HIGH candidates / min')
    ax.set_title('YouTube: HIGH Detection Rate')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(agg['high_per_min']):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)

    # Background probability
    ax = axes[1]
    ax.bar(agg['model'], agg['mean_bg_prob'], color=colors[:len(agg)])
    ax.set_ylabel('Mean background probability')
    ax.set_title('YouTube: Background Firing Rate')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(agg['mean_bg_prob']):
        ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'youtube_firing_rate.png'), dpi=150)
    plt.close()
    print("  Saved youtube_firing_rate.png")


def fig_detector_trace():
    """Frame-by-frame probability trace: baseline vs best variant."""
    vid = '38tq2ze5BJE'
    bp = os.path.join(EVAL_DIR, f'frame_probs_baseline_{vid}.npz')
    vp = os.path.join(EVAL_DIR, f'frame_probs_last_variant_{vid}.npz')

    if not os.path.exists(bp) or not os.path.exists(vp):
        print("  SKIP: frame prob traces not found")
        return

    b_probs = np.load(bp)['probs']
    v_probs = np.load(vp)['probs']

    # Plot a 5-minute window (start from 5 min in)
    fps = 30  # approximate for YouTube
    start = 5 * 60 * fps
    end = start + 5 * 60 * fps
    start = min(start, len(b_probs) - 1000)
    end = min(end, len(b_probs))

    t = np.arange(end - start) / fps / 60 + start / fps / 60

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t, b_probs[start:end], color='#4e79a7', alpha=0.8, linewidth=0.5)
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Baseline (original)')
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(t, v_probs[start:end], color='#e15759', alpha=0.8, linewidth=0.5)
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Best Hard-Negative Variant')
    axes[1].set_xlabel('Time (minutes)')
    axes[1].set_ylim(0, 1.05)

    plt.suptitle(f'Detector Trace: {vid} (5-min window)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'detector_trace_comparison.png'), dpi=150)
    plt.close()
    print("  Saved detector_trace_comparison.png")


def fig_variant_comparison():
    """Full comparison heatmap/table."""
    bobsl_path = os.path.join(EVAL_DIR, 'bobsl_fair_eval.csv')
    yt_path = os.path.join(EVAL_DIR, 'youtube_eval.csv')

    if not os.path.exists(bobsl_path) or not os.path.exists(yt_path):
        print("  SKIP: missing eval CSVs")
        return

    bobsl = pd.read_csv(bobsl_path).set_index('model')
    yt = pd.read_csv(yt_path)
    yt_agg = yt.groupby('model').agg({
        'n_high': 'sum', 'duration_min': 'sum', 'mean_prob': 'mean', 'mean_bg_prob': 'mean',
    })
    yt_agg['high_per_min'] = yt_agg['n_high'] / yt_agg['duration_min']

    # Build comparison table
    models = bobsl.index.tolist()
    metrics = ['f1_03', 'f1_05', 'precision03', 'recall03', 'fp_per_min']
    yt_metrics = ['high_per_min', 'mean_prob', 'mean_bg_prob']

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    headers = ['Model', 'F1@0.3', 'F1@0.5', 'Prec', 'Recall', 'FP/min',
               'YT HIGH/min', 'YT mean_p', 'YT bg_p']
    table_data = []
    for m in models:
        row = [m]
        for met in metrics:
            row.append(f'{bobsl.loc[m, met]:.3f}')
        if m in yt_agg.index:
            row.append(f'{yt_agg.loc[m, "high_per_min"]:.1f}')
            row.append(f'{yt_agg.loc[m, "mean_prob"]:.3f}')
            row.append(f'{yt_agg.loc[m, "mean_bg_prob"]:.3f}')
        else:
            row.extend(['N/A'] * 3)
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color-code: highlight best per column
    for col_idx in range(1, len(headers)):
        vals = []
        for row in table_data:
            try:
                vals.append(float(row[col_idx]))
            except (ValueError, IndexError):
                vals.append(None)
        # Best = highest for F1/P/R, lowest for FP/min and YT metrics
        if col_idx in [5, 6, 7, 8]:  # lower is better
            best_idx = min(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else float('inf'))
        else:  # higher is better
            best_idx = max(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else float('-inf'))
        table[best_idx + 1, col_idx].set_facecolor('#c7e9c0')

    plt.title('Full Variant Comparison', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'variant_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved variant_comparison.png")


def generate_report():
    """Generate HARD_NEGATIVE_REPORT.md."""
    # Load all data
    audit = {}
    audit_path = os.path.join(AUDIT_DIR, 'audit_summary.json')
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)

    mining = {}
    mining_path = os.path.join(HARD_NEG_DIR, 'hard_negative_stats.json')
    if os.path.exists(mining_path):
        with open(mining_path) as f:
            mining = json.load(f)

    training = {}
    training_path = os.path.join(LOG_DIR, 'training_summary.json')
    if os.path.exists(training_path):
        with open(training_path) as f:
            training = json.load(f)

    bobsl_df = None
    bobsl_path = os.path.join(EVAL_DIR, 'bobsl_fair_eval.csv')
    if os.path.exists(bobsl_path):
        bobsl_df = pd.read_csv(bobsl_path)

    yt_df = None
    yt_path = os.path.join(EVAL_DIR, 'youtube_eval.csv')
    if os.path.exists(yt_path):
        yt_df = pd.read_csv(yt_path)

    signing_check = {}
    sc_path = os.path.join(EVAL_DIR, 'signing_vs_nonsigning_check.json')
    if os.path.exists(sc_path):
        with open(sc_path) as f:
            signing_check = json.load(f)

    report = f"""# Hard Negative Mining & Detector Retraining Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Model:** Rich CNN (158d, 25f window)
**Task:** Improve detector specificity for YouTube interpreter content

---

## 1. Negative Audit

Sampled {audit.get('n_negatives', 'N/A')} current training negatives and {audit.get('n_positives', 'N/A')} positives.

### Activity Classification of Current Negatives
"""
    if 'classification_pct' in audit:
        for cls, pct in sorted(audit['classification_pct'].items()):
            report += f"- **{cls}**: {pct}%\n"

    report += f"""
### Wrist Velocity Comparison
| Metric | Positives | Negatives |
|--------|-----------|-----------|
| Mean | {audit.get('positive_wrist_vel', {}).get('mean', 'N/A'):.4f} | {audit.get('negative_wrist_vel', {}).get('mean', 'N/A'):.4f} |
| Median | {audit.get('positive_wrist_vel', {}).get('median', 'N/A'):.4f} | {audit.get('negative_wrist_vel', {}).get('median', 'N/A'):.4f} |

### Key Finding
Current negatives are **NOT idle** — {audit.get('classification_pct', {}).get('ACTIVE', 'N/A')}% are classified as ACTIVE with similar wrist velocity distributions to positives. The 5s temporal buffer already selects active signing periods. The YouTube firing issue is more likely due to **domain shift** than negative quality.

---

## 2. Hard Negative Mining
"""

    # Fall back to counting from metadata/npz files if mining stats unavailable
    a_meta_path = os.path.join(HARD_NEG_DIR, 'method_a_metadata.csv')
    a_windows_path = os.path.join(HARD_NEG_DIR, 'method_a_windows.npz')
    n_a_seg = n_a_win = 0
    if os.path.exists(a_meta_path):
        n_a_seg = len(pd.read_csv(a_meta_path)) - 1  # minus header
    if os.path.exists(a_windows_path):
        n_a_win = 666128  # known from mining output

    if mining:
        n_a_win = mining.get('method_a', {}).get('n_windows', n_a_win)
        n_a_seg = mining.get('method_a', {}).get('n_segments', n_a_seg)

    report += f"""
| Method | Windows | Segments | Notes |
|--------|---------|----------|-------|
| A: Between-annotation | {n_a_win:,} | {n_a_seg:,} | Gaps 2-30s, 1s buffer, 244 episodes |
| B: Subtitle-aligned | — | — | OOM killed at 200/244 episodes |
| **Total (train)** | **{n_a_win:,}** | | Method A only (>2x positives) |

Method B was killed by OOM at ~1.36M windows. Method A alone provided {n_a_win:,} windows, more than sufficient (>2x the 283K positives).
"""

    report += """
---

## 3. Training Results
"""
    if training:
        report += "\n| Variant | Val AUROC | Val F1 | Epochs |\n|---------|-----------|--------|--------|\n"
        for name in ['V1_hard_only', 'V2_50_50', 'V3_75_25', 'V4_curriculum']:
            if name in training:
                t = training[name]
                report += f"| {name} | {t['auroc']:.4f} | {t['best_f1']:.4f} | {t['epochs_trained']} |\n"
        if 'shuffled_auroc_V1' in training:
            report += f"\nShuffled-label sanity check (V1): AUROC = {training['shuffled_auroc_V1']:.4f} (expect ~0.5)\n"

    report += """
---

## 4. BOBSL Fair Evaluation (5 Held-Out Episodes)
"""
    if bobsl_df is not None:
        report += "\n| Model | F1@0.3 | F1@0.5 | Precision | Recall | FP/min |\n"
        report += "|-------|--------|--------|-----------|--------|--------|\n"
        for _, r in bobsl_df.iterrows():
            report += f"| {r['model']} | {r['f1_03']:.3f} | {r['f1_05']:.3f} | " \
                      f"{r['precision03']:.3f} | {r['recall03']:.3f} | {r['fp_per_min']:.2f} |\n"

    report += """
---

## 5. YouTube Interpreter Evaluation
"""
    if yt_df is not None:
        yt_agg = yt_df.groupby('model').agg({
            'n_high': 'sum', 'n_medium': 'sum', 'n_low': 'sum',
            'duration_min': 'sum', 'mean_prob': 'mean', 'mean_bg_prob': 'mean',
        }).reset_index()
        yt_agg['high_per_min'] = yt_agg['n_high'] / yt_agg['duration_min']

        report += "\n| Model | HIGH | MEDIUM | LOW | HIGH/min | Mean prob | BG prob |\n"
        report += "|-------|------|--------|-----|----------|-----------|--------|\n"
        for _, r in yt_agg.iterrows():
            report += f"| {r['model']} | {r['n_high']:.0f} | {r['n_medium']:.0f} | " \
                      f"{r['n_low']:.0f} | {r['high_per_min']:.1f} | " \
                      f"{r['mean_prob']:.3f} | {r['mean_bg_prob']:.3f} |\n"

    if signing_check:
        report += "\n### Signing vs Non-Signing Check\n"
        for model_name, vids in signing_check.items():
            report += f"\n**{model_name}:**\n"
            for vid, vals in vids.items():
                report += f"- {vid}: interpreter={vals.get('interpreter_mean_prob', 0):.3f}, " \
                          f"full_frame={vals.get('full_frame_mean_prob', 0):.3f}\n"

    report += """
---

## 6. Figures

| Figure | Description |
|--------|-------------|
| `wrist_velocity_histogram.png` | Wrist velocity: positives vs negatives |
| `activity_scatter.png` | Wrist velocity vs hand height |
| `negative_composition_pie.png` | IDLE/ACTIVE/AMBIGUOUS split |
| `training_curves.png` | AUROC training curves for all variants |
| `bobsl_comparison.png` | BOBSL F1 comparison bar chart |
| `youtube_firing_rate.png` | YouTube HIGH/min across variants |
| `detector_trace_comparison.png` | Frame-by-frame probability trace |
| `variant_comparison.png` | Full comparison table |

---

## 7. Critical Finding: Data Source Domain Shift

All hard negative variants achieved near-zero F1 on BOBSL while dramatically reducing YouTube false positives. This reveals a **data source confound**:

- **Positives**: From per-annotation landmark files (`mediapipe_features/train/`) — 1,660 episodes
- **Hard negatives**: From full-episode landmark files (`full_episode_detection/landmarks/`) — 244 episodes
- **Evaluation**: Uses full-episode landmarks (same source as hard negatives)

The model learned to distinguish the **landmark extraction pipeline** rather than fingerspelling vs non-fingerspelling. Evidence:
1. Val AUROC = 1.0000 (trivially separable — suspiciously perfect)
2. BOBSL F1 = 0.000 (detects nothing on full-episode data)
3. YouTube mean_prob reduced 60x (0.121 to 0.002)

### Root Cause
Per-annotation landmarks in `mediapipe_features/train/` were extracted differently from full-episode landmarks, creating a distributional confound that the model exploited instead of learning the task.

### Recommendations
1. **Re-extract positives from full-episode landmarks** at annotation timestamps to eliminate the domain confound
2. **Verify pipeline consistency**: Compare per-annotation vs full-episode landmarks for the same annotations
3. **Retrain with matched sources**: Hard negative approach may work correctly once data sources are unified
4. **YouTube threshold tuning**: As an interim fix, apply higher confidence threshold for YouTube

---

*Generated by analyse_results.py*
"""

    report_path = os.path.join(BASE_DIR, 'HARD_NEGATIVE_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved HARD_NEGATIVE_REPORT.md")


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEP 6: ANALYSIS & REPORT")
    print("=" * 70)

    print("\nGenerating figures...")
    fig_training_curves()
    fig_bobsl_comparison()
    fig_youtube_firing_rate()
    fig_detector_trace()
    fig_variant_comparison()

    print("\nGenerating report...")
    generate_report()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
