#!/usr/bin/env python3
"""
Domain shift verification: confirm the data source confound is eliminated.

Test A: Old per-annotation positives vs hard negatives → expect AUROC ~1.0 (confound present)
Test B: New full-episode positives vs hard negatives → expect AUROC 0.55-0.75 (confound gone)

If Test B AUROC > 0.85: STOP — confound persists, do not proceed to training.
"""
import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────
UNIFIED_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/unified_data/')
OLD_DATA_PATH = os.path.expanduser('~/bsl_project/improved_detection/data/rich_25f.npz')

N_SAMPLE = 5000
SEED = 42


def compute_summary_stats(windows):
    """Compute per-window summary: mean, std, min, max per feature dim → 632-dim."""
    # windows shape: (N, 25, 158)
    mean = windows.mean(axis=1)   # (N, 158)
    std = windows.std(axis=1)     # (N, 158)
    wmin = windows.min(axis=1)    # (N, 158)
    wmax = windows.max(axis=1)    # (N, 158)
    return np.concatenate([mean, std, wmin, wmax], axis=1)  # (N, 632)


def run_test(pos_stats, neg_stats, name):
    """Train logistic regression and report AUROC."""
    X = np.vstack([pos_stats, neg_stats])
    y = np.concatenate([np.ones(len(pos_stats)), np.zeros(len(neg_stats))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Use a small max_iter to keep it fast
    clf = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, probs)
    acc = clf.score(X, y)

    print(f"  {name}: AUROC={auroc:.4f}, Accuracy={acc:.4f}")
    return {'auroc': float(auroc), 'accuracy': float(acc)}


def main():
    print("=" * 70)
    print("DOMAIN SHIFT VERIFICATION")
    print("=" * 70)

    rng = np.random.RandomState(SEED)

    # Load new positives (from full-episode landmarks)
    print("\nLoading new positives (full-episode source)...")
    new_pos = np.load(os.path.join(UNIFIED_DIR, 'positives_train.npz'))['windows']
    print(f"  Shape: {new_pos.shape}")
    idx = rng.choice(len(new_pos), size=min(N_SAMPLE, len(new_pos)), replace=False)
    new_pos_sample = new_pos[idx]
    del new_pos

    # Load hard negatives (also from full-episode landmarks)
    print("Loading hard negatives (full-episode source)...")
    hard_neg = np.load(os.path.join(UNIFIED_DIR, 'hard_neg_train.npz'))['windows']
    print(f"  Shape: {hard_neg.shape}")
    idx = rng.choice(len(hard_neg), size=min(N_SAMPLE, len(hard_neg)), replace=False)
    hard_neg_sample = hard_neg[idx]
    del hard_neg

    # Load old positives (from per-annotation landmarks — the confounded source)
    print("Loading old positives (per-annotation source)...")
    old_data = np.load(OLD_DATA_PATH)
    old_pos = old_data['train_X'][old_data['train_y'] == 1]
    print(f"  Shape: {old_pos.shape}")
    idx = rng.choice(len(old_pos), size=min(N_SAMPLE, len(old_pos)), replace=False)
    old_pos_sample = old_pos[idx]
    del old_pos, old_data

    # Compute summary stats
    print("\nComputing summary statistics...")
    new_pos_stats = compute_summary_stats(new_pos_sample)
    hard_neg_stats = compute_summary_stats(hard_neg_sample)
    old_pos_stats = compute_summary_stats(old_pos_sample)

    print(f"  Summary dims: {new_pos_stats.shape[1]}")

    # ── Test A: Old positives vs hard negatives ──
    print("\n--- Test A: Old (per-annotation) positives vs hard negatives ---")
    print("  (Expect AUROC ~1.0 — confirms original confound)")
    result_a = run_test(old_pos_stats, hard_neg_stats, "Test A")

    # ── Test B: New positives vs hard negatives ──
    print("\n--- Test B: New (full-episode) positives vs hard negatives ---")
    print("  (Expect AUROC 0.55-0.75 — confound eliminated)")
    result_b = run_test(new_pos_stats, hard_neg_stats, "Test B")

    # ── Per-group mean comparison ──
    print("\n--- Feature Group Mean Comparison ---")
    # Group A: dims 0-23, B: 24-131, C: 132-137, D: 138-157
    groups = {'A_kinematics': (0, 24), 'B_hand_shape': (24, 132),
              'C_mouth': (132, 138), 'D_temporal': (138, 158)}
    group_stats = {}
    for gname, (start, end) in groups.items():
        new_mean = new_pos_sample[:, :, start:end].mean()
        hard_mean = hard_neg_sample[:, :, start:end].mean()
        old_mean = old_pos_sample[:, :, start:end].mean()
        print(f"  {gname}: old_pos={old_mean:.4f}, new_pos={new_mean:.4f}, hard_neg={hard_mean:.4f}")
        group_stats[gname] = {
            'old_pos_mean': float(old_mean),
            'new_pos_mean': float(new_mean),
            'hard_neg_mean': float(hard_mean),
        }

    # ── Save results ──
    results = {
        'test_a_old_vs_hard': result_a,
        'test_b_new_vs_hard': result_b,
        'group_means': group_stats,
        'n_sample': N_SAMPLE,
        'gate_passed': result_b['auroc'] < 0.85,
    }

    out_path = os.path.join(UNIFIED_DIR, 'domain_shift_verification.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    if result_b['auroc'] < 0.85:
        print(f"GATE PASSED: Test B AUROC = {result_b['auroc']:.4f} < 0.85")
        print("Domain confound eliminated. Proceed to training.")
    else:
        print(f"GATE FAILED: Test B AUROC = {result_b['auroc']:.4f} >= 0.85")
        print("Domain confound may persist. Investigate before training!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
