#!/usr/bin/env python3
"""
Step 5 (v2): Generate UNIFIED_SOURCE_REPORT.md with figures.

Compares:
- Original confounded results (from evaluation/)
- Unified-source results (from evaluation_v2/)
- Domain shift verification (from unified_data/)
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
UNIFIED_DIR = os.path.join(BASE_DIR, 'unified_data/')
EVAL_V2_DIR = os.path.join(BASE_DIR, 'evaluation_v2/')
EVAL_V1_DIR = os.path.join(BASE_DIR, 'evaluation/')
LOG_V2_DIR = os.path.join(BASE_DIR, 'models_v2/training_logs/')
FIG_DIR = os.path.join(EVAL_V2_DIR, 'figures/')
os.makedirs(FIG_DIR, exist_ok=True)


def fig_training_curves():
    """Plot AUROC training curves for all v2 variants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'V1_hard_only': '#e15759', 'V2_50_50': '#59a14f',
              'V3_75_25': '#4e79a7', 'V4_curriculum': '#edc948'}

    any_plotted = False
    for name, color in colors.items():
        path = os.path.join(LOG_V2_DIR, f'{name}_history.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            history = json.load(f)
        epochs = [h['epoch'] for h in history]
        aurocs = [h['auroc'] for h in history]
        ax.plot(epochs, aurocs, label=name, color=color, linewidth=2)
        any_plotted = True

    if not any_plotted:
        print("  SKIP: No training history files found")
        plt.close()
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUROC')
    ax.set_title('Training Curves: Unified-Source Variants (v2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'training_curves_v2.png'), dpi=150)
    plt.close()
    print("  Saved training_curves_v2.png")


def fig_bobsl_comparison():
    """Bar chart comparing BOBSL F1: baseline vs confounded vs unified."""
    v2_path = os.path.join(EVAL_V2_DIR, 'bobsl_fair_eval.csv')
    v1_path = os.path.join(EVAL_V1_DIR, 'bobsl_fair_eval.csv')

    if not os.path.exists(v2_path):
        print("  SKIP: bobsl_fair_eval.csv (v2) not found")
        return

    v2 = pd.read_csv(v2_path)

    # Also load v1 (confounded) for comparison if available
    v1 = pd.read_csv(v1_path) if os.path.exists(v1_path) else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # F1@0.3
    ax = axes[0]
    models = v2['model'].tolist()
    f1_03 = v2['f1_03'].tolist()

    # Add confounded v1 results for comparison
    if v1 is not None:
        conf_models = [m for m in v1['model'] if m != 'Baseline']
        for cm in conf_models:
            label = f'{cm}_conf'
            models.append(label)
            f1_03.append(float(v1[v1['model'] == cm]['f1_03'].iloc[0]))

    colors = []
    for m in models:
        if m == 'Baseline':
            colors.append('#4e79a7')
        elif '_conf' in m:
            colors.append('#bab0ac')  # grey for confounded
        else:
            colors.append('#e15759')

    ax.bar(range(len(models)), f1_03, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('F1 @ IoU=0.3')
    ax.set_title('F1 @ IoU=0.3')
    for i, v in enumerate(f1_03):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    # F1@0.5 (v2 only)
    ax = axes[1]
    v2_models = v2['model'].tolist()
    v2_colors = ['#4e79a7'] + ['#e15759'] * (len(v2_models) - 1)
    ax.bar(v2_models, v2['f1_05'], color=v2_colors)
    ax.set_ylabel('F1 @ IoU=0.5')
    ax.set_title('F1 @ IoU=0.5')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(v2['f1_05']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    # FP/min
    ax = axes[2]
    ax.bar(v2_models, v2['fp_per_min'], color=v2_colors)
    ax.set_ylabel('FP / minute')
    ax.set_title('False Positive Rate')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(v2['fp_per_min']):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bobsl_comparison_v2.png'), dpi=150)
    plt.close()
    print("  Saved bobsl_comparison_v2.png")


def fig_youtube_comparison():
    """YouTube HIGH/min comparison: baseline vs confounded vs unified."""
    v2_path = os.path.join(EVAL_V2_DIR, 'youtube_eval.csv')
    v1_path = os.path.join(EVAL_V1_DIR, 'youtube_eval.csv')

    if not os.path.exists(v2_path):
        print("  SKIP: youtube_eval.csv (v2) not found")
        return

    v2 = pd.read_csv(v2_path)
    v2_agg = v2.groupby('model').agg({
        'n_high': 'sum', 'duration_min': 'sum',
        'mean_prob': 'mean', 'mean_bg_prob': 'mean',
    }).reset_index()
    v2_agg['high_per_min'] = v2_agg['n_high'] / v2_agg['duration_min']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # HIGH/min
    ax = axes[0]
    models = v2_agg['model'].tolist()
    hpm = v2_agg['high_per_min'].tolist()

    # Add confounded for context
    if os.path.exists(v1_path):
        v1 = pd.read_csv(v1_path)
        v1_agg = v1.groupby('model').agg({
            'n_high': 'sum', 'duration_min': 'sum',
        }).reset_index()
        v1_agg['high_per_min'] = v1_agg['n_high'] / v1_agg['duration_min']
        for _, r in v1_agg.iterrows():
            if r['model'] != 'Baseline':
                models.append(f"{r['model']}_conf")
                hpm.append(r['high_per_min'])

    colors = []
    for m in models:
        if m == 'Baseline':
            colors.append('#4e79a7')
        elif '_conf' in m:
            colors.append('#bab0ac')
        else:
            colors.append('#e15759')

    ax.bar(range(len(models)), hpm, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('HIGH candidates / min')
    ax.set_title('YouTube: HIGH Detection Rate')
    for i, v in enumerate(hpm):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=8)

    # Background probability (v2 only)
    ax = axes[1]
    v2_colors = ['#4e79a7'] + ['#e15759'] * (len(v2_agg) - 1)
    ax.bar(v2_agg['model'], v2_agg['mean_bg_prob'], color=v2_colors)
    ax.set_ylabel('Mean background probability')
    ax.set_title('YouTube: Background Firing Rate')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(v2_agg['mean_bg_prob']):
        ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'youtube_comparison_v2.png'), dpi=150)
    plt.close()
    print("  Saved youtube_comparison_v2.png")


def fig_variant_table():
    """Full comparison table as figure."""
    v2_bobsl_path = os.path.join(EVAL_V2_DIR, 'bobsl_fair_eval.csv')
    v2_yt_path = os.path.join(EVAL_V2_DIR, 'youtube_eval.csv')

    if not os.path.exists(v2_bobsl_path) or not os.path.exists(v2_yt_path):
        print("  SKIP: missing v2 eval CSVs")
        return

    bobsl = pd.read_csv(v2_bobsl_path).set_index('model')
    yt = pd.read_csv(v2_yt_path)
    yt_agg = yt.groupby('model').agg({
        'n_high': 'sum', 'duration_min': 'sum', 'mean_prob': 'mean', 'mean_bg_prob': 'mean',
    })
    yt_agg['high_per_min'] = yt_agg['n_high'] / yt_agg['duration_min']

    models = bobsl.index.tolist()
    headers = ['Model', 'F1@0.3', 'F1@0.5', 'Prec', 'Recall', 'FP/min',
               'YT HIGH/min', 'YT mean_p', 'YT bg_p']
    table_data = []
    for m in models:
        row = [m]
        for met in ['f1_03', 'f1_05', 'precision03', 'recall03', 'fp_per_min']:
            row.append(f'{bobsl.loc[m, met]:.3f}')
        if m in yt_agg.index:
            row.append(f'{yt_agg.loc[m, "high_per_min"]:.1f}')
            row.append(f'{yt_agg.loc[m, "mean_prob"]:.3f}')
            row.append(f'{yt_agg.loc[m, "mean_bg_prob"]:.3f}')
        else:
            row.extend(['N/A'] * 3)
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for col_idx in range(1, len(headers)):
        vals = []
        for row in table_data:
            try:
                vals.append(float(row[col_idx]))
            except (ValueError, IndexError):
                vals.append(None)
        if col_idx in [5, 6, 7, 8]:  # lower is better
            best_idx = min(range(len(vals)),
                          key=lambda i: vals[i] if vals[i] is not None else float('inf'))
        else:
            best_idx = max(range(len(vals)),
                          key=lambda i: vals[i] if vals[i] is not None else float('-inf'))
        table[best_idx + 1, col_idx].set_facecolor('#c7e9c0')

    plt.title('Unified-Source Variant Comparison (v2)', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'variant_comparison_v2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved variant_comparison_v2.png")


def generate_report():
    """Generate UNIFIED_SOURCE_REPORT.md."""

    # ── Load all data ──
    # Domain shift verification
    ds_path = os.path.join(UNIFIED_DIR, 'domain_shift_verification.json')
    ds = {}
    if os.path.exists(ds_path):
        with open(ds_path) as f:
            ds = json.load(f)

    # Dataset stats
    ds_stats_path = os.path.join(UNIFIED_DIR, 'dataset_stats.json')
    ds_stats = {}
    if os.path.exists(ds_stats_path):
        with open(ds_stats_path) as f:
            ds_stats = json.load(f)

    # Training logs
    training = {}
    training_path = os.path.join(LOG_V2_DIR, 'training_summary.json')
    if os.path.exists(training_path):
        with open(training_path) as f:
            training = json.load(f)

    # BOBSL eval
    bobsl_v2_df = None
    bobsl_v2_path = os.path.join(EVAL_V2_DIR, 'bobsl_fair_eval.csv')
    if os.path.exists(bobsl_v2_path):
        bobsl_v2_df = pd.read_csv(bobsl_v2_path)

    # YouTube eval
    yt_v2_df = None
    yt_v2_path = os.path.join(EVAL_V2_DIR, 'youtube_eval.csv')
    if os.path.exists(yt_v2_path):
        yt_v2_df = pd.read_csv(yt_v2_path)

    # Signing check
    signing_check = {}
    sc_path = os.path.join(EVAL_V2_DIR, 'signing_vs_nonsigning_check.json')
    if os.path.exists(sc_path):
        with open(sc_path) as f:
            signing_check = json.load(f)

    # Previous (confounded) BOBSL results for comparison
    bobsl_v1_df = None
    bobsl_v1_path = os.path.join(EVAL_V1_DIR, 'bobsl_fair_eval.csv')
    if os.path.exists(bobsl_v1_path):
        bobsl_v1_df = pd.read_csv(bobsl_v1_path)

    # ── Build report ──
    report = f"""# Unified Source Hard Negative Retraining Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Model:** Rich CNN (158d, 25f window)
**Purpose:** Fix data source domain shift found in first hard negative experiment

---

## 0. Background: Data Source Confound

The first hard negative experiment discovered a critical data pipeline confound:
- **Positives** came from per-annotation landmarks (`mediapipe_features/train/`, 1,660 episodes)
- **Hard negatives** came from full-episode landmarks (`full_episode_detection/landmarks/`, 244 episodes)

The model learned to distinguish the **extraction pipeline** (Val AUROC=1.0000), not fingerspelling.
Result: F1=0.000 on BOBSL (which uses full-episode landmarks).

**Fix:** Re-extract ALL training data from full-episode landmark files.

---

## 1. Unified Dataset Extraction

All positives, easy negatives, and hard negatives now come from the same source:
`full_episode_detection/landmarks/` (244 non-held-out episodes).

"""

    # Dataset sizes from the npz files
    for split_name, fname in [('positives_train', 'positives_train.npz'),
                               ('positives_val', 'positives_val.npz'),
                               ('easy_neg_train', 'easy_neg_train.npz'),
                               ('easy_neg_val', 'easy_neg_val.npz'),
                               ('hard_neg_train', 'hard_neg_train.npz'),
                               ('hard_neg_val', 'hard_neg_val.npz')]:
        fpath = os.path.join(UNIFIED_DIR, fname)
        if os.path.exists(fpath):
            data = np.load(fpath)
            shape = data['windows'].shape
            report += f"- **{split_name}**: {shape[0]:,} windows ({shape})\n"
            del data

    report += f"""
Episode split: {ds_stats.get('n_train_episodes', '?')} train, {ds_stats.get('n_val_episodes', '?')} val
Annotations processed: {ds_stats.get('n_pos_ann_processed', '?')}
Annotations skipped (short): {ds_stats.get('n_pos_ann_skipped_short', '?')}

---

## 2. Domain Shift Verification (GATE CHECK)

"""
    if ds:
        test_a = ds.get('test_a_old_vs_hard', {})
        test_b = ds.get('test_b_new_vs_hard', {})

        report += f"""| Test | Description | AUROC | Accuracy |
|------|-------------|-------|----------|
| A: Old pos vs hard neg | Confirms original confound | {test_a.get('auroc', 0):.4f} | {test_a.get('accuracy', 0):.4f} |
| B: New pos vs hard neg | **Should be low if confound eliminated** | {test_b.get('auroc', 0):.4f} | {test_b.get('accuracy', 0):.4f} |

**Gate result:** {'PASSED' if ds.get('gate_passed', False) else 'FAILED'} (Test B AUROC = {test_b.get('auroc', 0):.4f} {'<' if ds.get('gate_passed', False) else '>='} 0.85)

"""
        if 'group_means' in ds:
            report += "### Feature Group Mean Comparison\n\n"
            report += "| Group | Old positives | New positives | Hard negatives |\n"
            report += "|-------|--------------|---------------|----------------|\n"
            for gname, vals in ds['group_means'].items():
                report += f"| {gname} | {vals['old_pos_mean']:.4f} | {vals['new_pos_mean']:.4f} | {vals['hard_neg_mean']:.4f} |\n"
            report += "\n"
    else:
        report += "*Domain shift verification not run or results not found.*\n\n"

    report += """---

## 3. Training Results

"""
    if training:
        report += "| Variant | Val AUROC | Val F1 | Epochs |\n|---------|-----------|--------|--------|\n"
        for name in ['V1_hard_only', 'V2_50_50', 'V3_75_25', 'V4_curriculum']:
            if name in training:
                t = training[name]
                report += f"| {name} | {t['auroc']:.4f} | {t['best_f1']:.4f} | {t['epochs_trained']} |\n"

        if 'shuffled_auroc_V1' in training:
            report += f"\nShuffled-label sanity check (V1): AUROC = {training['shuffled_auroc_V1']:.4f} (expect ~0.5)\n"

        report += "\n**Key check:** Val AUROC should be significantly below 1.0000 (the confounded experiment achieved 1.0000 trivially).\n"
    else:
        report += "*Training results not found.*\n"

    report += """
---

## 4. BOBSL Fair Evaluation (5 Held-Out Episodes)

### Unified-Source Results (v2)

"""
    if bobsl_v2_df is not None:
        report += "| Model | F1@0.3 | F1@0.5 | Precision | Recall | FP/min |\n"
        report += "|-------|--------|--------|-----------|--------|--------|\n"
        for _, r in bobsl_v2_df.iterrows():
            report += f"| {r['model']} | {r['f1_03']:.3f} | {r['f1_05']:.3f} | " \
                      f"{r['precision03']:.3f} | {r['recall03']:.3f} | {r['fp_per_min']:.2f} |\n"
    else:
        report += "*BOBSL v2 results not found.*\n"

    if bobsl_v1_df is not None:
        report += "\n### Confounded Results (v1, for reference)\n\n"
        report += "| Model | F1@0.3 | F1@0.5 | Precision | Recall | FP/min |\n"
        report += "|-------|--------|--------|-----------|--------|--------|\n"
        for _, r in bobsl_v1_df.iterrows():
            report += f"| {r['model']} | {r['f1_03']:.3f} | {r['f1_05']:.3f} | " \
                      f"{r['precision03']:.3f} | {r['recall03']:.3f} | {r['fp_per_min']:.2f} |\n"

    report += """
---

## 5. YouTube Interpreter Evaluation

"""
    if yt_v2_df is not None:
        yt_agg = yt_v2_df.groupby('model').agg({
            'n_high': 'sum', 'n_medium': 'sum', 'n_low': 'sum',
            'duration_min': 'sum', 'mean_prob': 'mean', 'mean_bg_prob': 'mean',
        }).reset_index()
        yt_agg['high_per_min'] = yt_agg['n_high'] / yt_agg['duration_min']

        report += "| Model | HIGH | MEDIUM | LOW | HIGH/min | Mean prob | BG prob |\n"
        report += "|-------|------|--------|-----|----------|-----------|--------|\n"
        for _, r in yt_agg.iterrows():
            report += f"| {r['model']} | {r['n_high']:.0f} | {r['n_medium']:.0f} | " \
                      f"{r['n_low']:.0f} | {r['high_per_min']:.1f} | " \
                      f"{r['mean_prob']:.3f} | {r['mean_bg_prob']:.3f} |\n"
    else:
        report += "*YouTube v2 results not found.*\n"

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
| `training_curves_v2.png` | AUROC training curves for unified-source variants |
| `bobsl_comparison_v2.png` | BOBSL F1: baseline vs confounded vs unified |
| `youtube_comparison_v2.png` | YouTube HIGH/min: baseline vs confounded vs unified |
| `variant_comparison_v2.png` | Full comparison table |

---

## 7. Conclusions

"""

    # Auto-generate conclusions based on results
    if bobsl_v2_df is not None:
        best_v2 = bobsl_v2_df[bobsl_v2_df['model'] != 'Baseline']
        if len(best_v2) > 0:
            best_row = best_v2.loc[best_v2['f1_03'].idxmax()]
            baseline_row = bobsl_v2_df[bobsl_v2_df['model'] == 'Baseline']
            baseline_f1 = float(baseline_row['f1_03'].iloc[0]) if len(baseline_row) > 0 else 0

            report += f"**Best unified-source variant:** {best_row['model']} "
            report += f"(F1@0.3={best_row['f1_03']:.3f})\n"
            report += f"**Baseline:** F1@0.3={baseline_f1:.3f}\n\n"

            if best_row['f1_03'] > 0:
                report += "The domain shift confound has been eliminated: unified-source variants "
                report += "now produce non-zero BOBSL F1 scores, confirming they learned the actual task.\n\n"
            else:
                report += "WARNING: Unified-source variants still show F1=0.000 on BOBSL. "
                report += "The issue may not be purely data source related.\n\n"

            delta = best_row['f1_03'] - baseline_f1
            if delta > 0:
                report += f"The best variant improves over baseline by +{delta:.3f} F1@0.3.\n"
            elif delta < 0:
                report += f"The best variant is {abs(delta):.3f} below baseline. "
                report += "Hard negatives alone do not improve detection on BOBSL, "
                report += "but may still improve YouTube specificity.\n"

    report += """
---

*Generated by generate_report_v2.py*
"""

    report_path = os.path.join(BASE_DIR, 'UNIFIED_SOURCE_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved UNIFIED_SOURCE_REPORT.md")


def main():
    t0 = time.time()
    print("=" * 70)
    print("GENERATING UNIFIED SOURCE REPORT")
    print("=" * 70)

    print("\nGenerating figures...")
    fig_training_curves()
    fig_bobsl_comparison()
    fig_youtube_comparison()
    fig_variant_table()

    print("\nGenerating report...")
    generate_report()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"REPORT GENERATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
