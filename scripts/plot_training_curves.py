#!/usr/bin/env python3
"""Generate training curve plots for the V2 model audit."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/models_v2/training_logs/')
OUT_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/')

def load_history(variant):
    path = os.path.join(LOG_DIR, f'{variant}_history.json')
    with open(path) as f:
        return json.load(f)

def plot_all_variants():
    variants = {
        'V1_hard_only': 'V1: Hard Only',
        'V2_50_50': 'V2: 50/50 Mix',
        'V3_75_25': 'V3: 75/25 Mix',
        'V4_curriculum': 'V4: Curriculum',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Audit: Loss and Validation Metrics per Epoch', fontsize=14, fontweight='bold')

    for idx, (key, label) in enumerate(variants.items()):
        ax = axes[idx // 2][idx % 2]
        hist = load_history(key)

        epochs = [h['epoch'] for h in hist]
        train_loss = [h['train_loss'] for h in hist]
        auroc = [h['auroc'] for h in hist]

        # Find best epoch
        best_idx = np.argmax(auroc)
        best_epoch = epochs[best_idx]

        ax_loss = ax
        ax_auroc = ax.twinx()

        l1, = ax_loss.plot(epochs, train_loss, 'b-o', markersize=4, label='Train Loss', alpha=0.8)
        l2, = ax_auroc.plot(epochs, auroc, 'r-s', markersize=4, label='Val AUROC', alpha=0.8)
        ax_auroc.axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best epoch ({best_epoch})')

        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Train Loss', color='blue')
        ax_auroc.set_ylabel('Val AUROC', color='red')
        ax_loss.set_title(label)

        # Phase annotation for V4
        if key == 'V4_curriculum':
            ax_loss.axvline(10.5, color='gray', linestyle=':', alpha=0.5)
            ax_loss.text(5.5, max(train_loss) * 0.95, 'Phase 1\n(Easy)', ha='center',
                        fontsize=8, color='gray')
            ax_loss.text(14.5, max(train_loss) * 0.95, 'Phase 2\n(Hard)', ha='center',
                        fontsize=8, color='gray')

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax_loss.legend(lines, labels, loc='center right', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'training_curves_audit.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_v3_detail():
    """Detailed V3 training curve with overfitting analysis."""
    hist = load_history('V3_75_25')

    epochs = [h['epoch'] for h in hist]
    train_loss = [h['train_loss'] for h in hist]
    auroc = [h['auroc'] for h in hist]
    f1 = [h['best_f1'] for h in hist]
    lr = [h['lr'] for h in hist]

    best_idx = np.argmax(auroc)
    best_epoch = epochs[best_idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('V3 (75/25) Detailed Training Analysis', fontsize=14, fontweight='bold')

    # Panel 1: Train loss
    axes[0].plot(epochs, train_loss, 'b-o', markersize=5)
    axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best AUROC epoch ({best_epoch})')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss (monotonically decreasing)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Val AUROC + Val F1
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    l1, = ax2a.plot(epochs, auroc, 'r-s', markersize=5, label='Val AUROC')
    l2, = ax2b.plot(epochs, f1, 'g-^', markersize=5, label='Val F1', alpha=0.8)
    ax2a.axvline(best_epoch, color='green', linestyle='--', alpha=0.6)
    ax2a.set_ylabel('AUROC', color='red')
    ax2b.set_ylabel('Best F1', color='green')
    ax2a.set_title('Validation Metrics (plateau, no degradation)')
    lines = [l1, l2]
    ax2a.legend(lines, [l.get_label() for l in lines], loc='lower left')
    ax2a.grid(True, alpha=0.3)

    # Panel 3: Learning rate
    axes[2].step(epochs, lr, 'k-', where='mid')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_title('Learning Rate Schedule (ReduceLROnPlateau)')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'v3_training_detail_audit.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    plot_all_variants()
    plot_v3_detail()
    print("Done.")
