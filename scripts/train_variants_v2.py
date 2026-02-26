#!/usr/bin/env python3
"""
Train 4 detector variants using unified-source data (all from full-episode landmarks).

Variants:
  V1: Hard only — 100% hard negatives
  V2: 50/50 — 50% hard, 50% easy negatives
  V3: 75/25 — 75% hard, 25% easy
  V4: Curriculum — easy first (10 epochs), then hard (40 epochs)
"""
import os
import sys
import gc
import json
import time
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

# ── Paths ────────────────────────────────────────────────────────────────
UNIFIED_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/unified_data/')
MODEL_DIR = os.path.expanduser('~/bsl_project/hard_negative_detection/models_v2/')
LOG_DIR = os.path.join(MODEL_DIR, 'training_logs/')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
INPUT_DIM = 158
WINDOW_SIZE = 25


class CNN1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def train_model(model, train_X, train_y, val_X, val_y,
                epochs=50, batch_size=128, lr=1e-3, patience=7):
    """Train with early stopping on AUROC. Scales in-place to save memory."""
    model = model.to(DEVICE)

    feat_dim = train_X.shape[2]
    scaler = StandardScaler()
    n_fit = min(50000, len(train_X))
    fit_idx = np.random.choice(len(train_X), size=n_fit, replace=False)
    scaler.fit(train_X[fit_idx].reshape(-1, feat_dim))
    del fit_idx

    # Scale in-place
    train_X.reshape(-1, feat_dim).__isub__(scaler.mean_).__itruediv__(scaler.scale_ + 1e-8)
    val_X.reshape(-1, feat_dim).__isub__(scaler.mean_).__itruediv__(scaler.scale_ + 1e-8)

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y.astype(np.float32)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='max')

    best_auroc = 0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                probs = torch.sigmoid(model(X_batch.to(DEVICE))).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        auroc = roc_auc_score(all_labels, all_probs)

        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_f1_idx = np.argmax(f1s)
        best_f1 = f1s[best_f1_idx]
        best_thresh = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

        scheduler.step(auroc)
        history.append({
            'epoch': epoch + 1, 'train_loss': float(train_loss),
            'auroc': float(auroc), 'best_f1': float(best_f1),
            'best_thresh': float(best_thresh),
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = deepcopy(model.state_dict())
            best_f1_final = best_f1
            best_thresh_final = best_thresh
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or no_improve == 0:
            print(f"    Epoch {epoch+1:3d}: loss={train_loss:.4f} AUROC={auroc:.4f} "
                  f"F1={best_f1:.4f}@{best_thresh:.3f} {'*' if no_improve==0 else ''}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, {
        'best_auroc': float(best_auroc), 'best_f1': float(best_f1_final),
        'best_threshold': float(best_thresh_final),
        'epochs_trained': len(history), 'history': history,
    }


def train_curriculum(pos_train, easy_neg_train, hard_neg_train,
                     pos_val, neg_val, phase1_epochs=10, phase2_epochs=40):
    """Curriculum training: easy negatives first, then hard negatives."""
    model = CNN1D(input_dim=INPUT_DIM).to(DEVICE)
    feat_dim = INPUT_DIM

    # Phase 1: Easy negatives
    print("    Phase 1: Easy negatives...")
    n_pos = len(pos_train)
    easy_neg = easy_neg_train[:n_pos] if len(easy_neg_train) >= n_pos else easy_neg_train

    train_X1 = np.vstack([pos_train, easy_neg])
    train_y1 = np.concatenate([np.ones(n_pos), np.zeros(len(easy_neg))]).astype(np.float32)

    scaler1 = StandardScaler()
    n_fit = min(50000, len(train_X1))
    scaler1.fit(train_X1[np.random.choice(len(train_X1), n_fit, replace=False)].reshape(-1, feat_dim))

    train_X1.reshape(-1, feat_dim).__isub__(scaler1.mean_).__itruediv__(scaler1.scale_ + 1e-8)

    val_X = np.vstack([pos_val, neg_val])
    val_y = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))]).astype(np.float32)
    val_X.reshape(-1, feat_dim).__isub__(scaler1.mean_).__itruediv__(scaler1.scale_ + 1e-8)

    train_ds = TensorDataset(torch.from_numpy(train_X1), torch.from_numpy(train_y1))
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    for epoch in range(phase1_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        all_probs, all_labels_v = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                probs = torch.sigmoid(model(X_batch.to(DEVICE))).cpu().numpy()
                all_probs.extend(probs)
                all_labels_v.extend(y_batch.numpy())
        auroc = roc_auc_score(np.array(all_labels_v), np.array(all_probs))
        history.append({'epoch': epoch+1, 'phase': 1, 'train_loss': float(train_loss),
                        'auroc': float(auroc)})
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    P1 Epoch {epoch+1:3d}: loss={train_loss:.4f} AUROC={auroc:.4f}")

    del train_ds, val_ds, train_loader, val_loader, train_X1
    gc.collect()

    # Phase 2: Hard negatives
    print("    Phase 2: Hard negatives...")
    hard_neg = hard_neg_train[:n_pos] if len(hard_neg_train) >= n_pos else hard_neg_train
    train_X2 = np.vstack([pos_train, hard_neg])
    train_y2 = np.concatenate([np.ones(n_pos), np.zeros(len(hard_neg))]).astype(np.float32)

    scaler2 = StandardScaler()
    n_fit2 = min(50000, len(train_X2))
    scaler2.fit(train_X2[np.random.choice(len(train_X2), n_fit2, replace=False)].reshape(-1, feat_dim))

    train_X2.reshape(-1, feat_dim).__isub__(scaler2.mean_).__itruediv__(scaler2.scale_ + 1e-8)

    val_X2 = np.vstack([pos_val, neg_val])
    val_y2 = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))]).astype(np.float32)
    val_X2.reshape(-1, feat_dim).__isub__(scaler2.mean_).__itruediv__(scaler2.scale_ + 1e-8)

    train_ds2 = TensorDataset(torch.from_numpy(train_X2), torch.from_numpy(train_y2))
    val_ds2 = TensorDataset(torch.from_numpy(val_X2), torch.from_numpy(val_y2))
    train_loader2 = DataLoader(train_ds2, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader2 = DataLoader(val_ds2, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    optimizer2 = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, patience=3, factor=0.5, mode='max')

    best_auroc = 0
    best_state = None
    no_improve = 0

    for epoch in range(phase2_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader2:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer2.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer2.step()
            train_loss += loss.item()
        train_loss /= len(train_loader2)

        model.eval()
        all_probs, all_labels_v = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader2:
                probs = torch.sigmoid(model(X_batch.to(DEVICE))).cpu().numpy()
                all_probs.extend(probs)
                all_labels_v.extend(y_batch.numpy())
        all_probs = np.array(all_probs)
        all_labels_v = np.array(all_labels_v)
        auroc = roc_auc_score(all_labels_v, all_probs)

        precisions, recalls, thresholds = precision_recall_curve(all_labels_v, all_probs)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_f1_idx = np.argmax(f1s)
        best_f1 = f1s[best_f1_idx]
        best_thresh = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

        scheduler2.step(auroc)
        history.append({'epoch': phase1_epochs + epoch + 1, 'phase': 2,
                        'train_loss': float(train_loss), 'auroc': float(auroc),
                        'best_f1': float(best_f1)})

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = deepcopy(model.state_dict())
            best_f1_final = best_f1
            best_thresh_final = best_thresh
            no_improve = 0
        else:
            no_improve += 1

        if (epoch+1) % 5 == 0 or epoch == 0 or no_improve == 0:
            print(f"    P2 Epoch {epoch+1:3d}: loss={train_loss:.4f} AUROC={auroc:.4f} "
                  f"F1={best_f1:.4f} {'*' if no_improve==0 else ''}")

        if no_improve >= 7:
            print(f"    Early stopping at P2 epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler2, {
        'best_auroc': float(best_auroc), 'best_f1': float(best_f1_final),
        'best_threshold': float(best_thresh_final),
        'epochs_trained': len(history), 'history': history,
    }


def shuffled_label_check(train_X, train_y, val_X, val_y):
    """Train with shuffled labels to verify no data leakage."""
    print("\n  Shuffled label sanity check...")
    rng = np.random.RandomState(SEED)
    y_shuf = rng.permutation(train_y)
    model = CNN1D(input_dim=INPUT_DIM)
    # Use copies to avoid in-place scaling corruption
    tX = train_X.copy()
    vX = val_X.copy()
    _, _, results = train_model(model, tX, y_shuf, vX, val_y.copy(),
                                epochs=15, patience=5)
    print(f"  Shuffled AUROC: {results['best_auroc']:.4f} (should be ~0.5)")
    return results['best_auroc']


def save_model(model, scaler, results, variant_name, model_path):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
        'input_dim': INPUT_DIM, 'window_size': WINDOW_SIZE,
        'model_type': 'CNN1D',
        'auroc': results['best_auroc'], 'best_f1': results['best_f1'],
        'best_threshold': results['best_threshold'],
        'variant': variant_name,
        'data_source': 'full_episode_matched',
    }, model_path)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 70)
    print("TRAIN DETECTOR VARIANTS V2 (UNIFIED SOURCE)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    rng = np.random.RandomState(SEED)

    # Load unified data
    print("\nLoading unified dataset...")
    pos_train = np.load(os.path.join(UNIFIED_DIR, 'positives_train.npz'))['windows']
    pos_val = np.load(os.path.join(UNIFIED_DIR, 'positives_val.npz'))['windows']
    easy_neg_train = np.load(os.path.join(UNIFIED_DIR, 'easy_neg_train.npz'))['windows']
    easy_neg_val = np.load(os.path.join(UNIFIED_DIR, 'easy_neg_val.npz'))['windows']
    hard_neg_train = np.load(os.path.join(UNIFIED_DIR, 'hard_neg_train.npz'))['windows']
    hard_neg_val = np.load(os.path.join(UNIFIED_DIR, 'hard_neg_val.npz'))['windows']

    n_pos_train = len(pos_train)
    n_pos_val = len(pos_val)
    print(f"  Positives: train={n_pos_train:,}, val={n_pos_val:,}")
    print(f"  Hard negatives: train={len(hard_neg_train):,}, val={len(hard_neg_val):,}")
    print(f"  Easy negatives: train={len(easy_neg_train):,}, val={len(easy_neg_val):,}")

    all_results = {}

    # ══════════════════════════════════════════════════════════════════
    # VARIANT 1: Hard only
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("V1: HARD ONLY (100% hard negatives)")
    print(f"{'='*60}")

    n_neg = min(n_pos_train, len(hard_neg_train))
    idx = rng.choice(len(hard_neg_train), size=n_neg, replace=False)
    v1_train_X = np.vstack([pos_train, hard_neg_train[idx]])
    v1_train_y = np.concatenate([np.ones(n_pos_train), np.zeros(n_neg)])

    n_neg_v = min(n_pos_val, len(hard_neg_val))
    idx_v = rng.choice(len(hard_neg_val), size=n_neg_v, replace=False)
    v1_val_X = np.vstack([pos_val, hard_neg_val[idx_v]])
    v1_val_y = np.concatenate([np.ones(n_pos_val), np.zeros(n_neg_v)])

    print(f"  Train: {len(v1_train_X):,} ({n_pos_train:,} pos + {n_neg:,} neg)")

    model1 = CNN1D(input_dim=INPUT_DIM)
    model1, scaler1, res1 = train_model(model1, v1_train_X, v1_train_y, v1_val_X, v1_val_y)
    print(f"\n  RESULT: AUROC={res1['best_auroc']:.4f}, F1={res1['best_f1']:.4f}")
    save_model(model1, scaler1, res1, 'hard_only',
               os.path.join(MODEL_DIR, 'variant1_hard_only.pt'))
    all_results['V1_hard_only'] = res1

    # Shuffled label check
    shuf_auroc = shuffled_label_check(v1_train_X, v1_train_y, v1_val_X, v1_val_y)
    all_results['V1_shuffled_auroc'] = float(shuf_auroc)
    del v1_train_X, v1_train_y, v1_val_X, v1_val_y, model1, scaler1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # VARIANT 2: 50/50
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("V2: 50/50 MIX (50% hard + 50% easy)")
    print(f"{'='*60}")

    n_each = n_pos_train // 2
    idx_h = rng.choice(len(hard_neg_train), size=min(n_each, len(hard_neg_train)), replace=False)
    idx_e = rng.choice(len(easy_neg_train), size=min(n_each, len(easy_neg_train)), replace=False)
    v2_neg = np.vstack([hard_neg_train[idx_h], easy_neg_train[idx_e]])
    v2_train_X = np.vstack([pos_train, v2_neg])
    v2_train_y = np.concatenate([np.ones(n_pos_train), np.zeros(len(v2_neg))])
    del v2_neg

    n_each_v = n_pos_val // 2
    idx_hv = rng.choice(len(hard_neg_val), size=min(n_each_v, len(hard_neg_val)), replace=False)
    idx_ev = rng.choice(len(easy_neg_val), size=min(n_each_v, len(easy_neg_val)), replace=False)
    v2_val_neg = np.vstack([hard_neg_val[idx_hv], easy_neg_val[idx_ev]])
    v2_val_X = np.vstack([pos_val, v2_val_neg])
    v2_val_y = np.concatenate([np.ones(n_pos_val), np.zeros(len(v2_val_neg))])
    del v2_val_neg

    print(f"  Train: {len(v2_train_X):,} ({n_pos_train:,} pos + {n_each*2:,} neg)")

    model2 = CNN1D(input_dim=INPUT_DIM)
    model2, scaler2, res2 = train_model(model2, v2_train_X, v2_train_y, v2_val_X, v2_val_y)
    print(f"\n  RESULT: AUROC={res2['best_auroc']:.4f}, F1={res2['best_f1']:.4f}")
    save_model(model2, scaler2, res2, '50_50',
               os.path.join(MODEL_DIR, 'variant2_50_50.pt'))
    all_results['V2_50_50'] = res2
    del v2_train_X, v2_train_y, v2_val_X, v2_val_y, model2, scaler2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # VARIANT 3: 75/25
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("V3: 75/25 MIX (75% hard + 25% easy)")
    print(f"{'='*60}")

    n_hard3 = int(n_pos_train * 0.75)
    n_easy3 = n_pos_train - n_hard3
    idx_h3 = rng.choice(len(hard_neg_train), size=min(n_hard3, len(hard_neg_train)), replace=False)
    idx_e3 = rng.choice(len(easy_neg_train), size=min(n_easy3, len(easy_neg_train)), replace=False)
    v3_neg = np.vstack([hard_neg_train[idx_h3], easy_neg_train[idx_e3]])
    v3_train_X = np.vstack([pos_train, v3_neg])
    v3_train_y = np.concatenate([np.ones(n_pos_train), np.zeros(len(v3_neg))])
    del v3_neg

    n_hard3v = int(n_pos_val * 0.75)
    n_easy3v = n_pos_val - n_hard3v
    idx_hv3 = rng.choice(len(hard_neg_val), size=min(n_hard3v, len(hard_neg_val)), replace=False)
    idx_ev3 = rng.choice(len(easy_neg_val), size=min(n_easy3v, len(easy_neg_val)), replace=False)
    v3_val_neg = np.vstack([hard_neg_val[idx_hv3], easy_neg_val[idx_ev3]])
    v3_val_X = np.vstack([pos_val, v3_val_neg])
    v3_val_y = np.concatenate([np.ones(n_pos_val), np.zeros(len(v3_val_neg))])
    del v3_val_neg

    print(f"  Train: {len(v3_train_X):,} ({n_pos_train:,} pos + {n_hard3+n_easy3:,} neg)")

    model3 = CNN1D(input_dim=INPUT_DIM)
    model3, scaler3, res3 = train_model(model3, v3_train_X, v3_train_y, v3_val_X, v3_val_y)
    print(f"\n  RESULT: AUROC={res3['best_auroc']:.4f}, F1={res3['best_f1']:.4f}")
    save_model(model3, scaler3, res3, '75_25',
               os.path.join(MODEL_DIR, 'variant3_75_25.pt'))
    all_results['V3_75_25'] = res3
    del v3_train_X, v3_train_y, v3_val_X, v3_val_y, model3, scaler3
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # VARIANT 4: Curriculum
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("V4: CURRICULUM (easy first, then hard)")
    print(f"{'='*60}")

    v4_val_neg = hard_neg_val[:n_pos_val] if len(hard_neg_val) >= n_pos_val else hard_neg_val
    model4, scaler4, res4 = train_curriculum(
        pos_train.copy(), easy_neg_train, hard_neg_train,
        pos_val.copy(), v4_val_neg,
        phase1_epochs=10, phase2_epochs=40)
    print(f"\n  RESULT: AUROC={res4['best_auroc']:.4f}, F1={res4['best_f1']:.4f}")
    save_model(model4, scaler4, res4, 'curriculum',
               os.path.join(MODEL_DIR, 'variant4_curriculum.pt'))
    all_results['V4_curriculum'] = res4

    # Save training logs
    for name, res in [('V1_hard_only', res1), ('V2_50_50', res2),
                      ('V3_75_25', res3), ('V4_curriculum', res4)]:
        with open(os.path.join(LOG_DIR, f'{name}_history.json'), 'w') as f:
            json.dump(res['history'], f, indent=2)

    summary = {}
    for name in ['V1_hard_only', 'V2_50_50', 'V3_75_25', 'V4_curriculum']:
        r = all_results[name]
        summary[name] = {
            'auroc': r['best_auroc'], 'best_f1': r['best_f1'],
            'best_threshold': r['best_threshold'], 'epochs_trained': r['epochs_trained'],
        }
    summary['shuffled_auroc_V1'] = all_results.get('V1_shuffled_auroc')

    with open(os.path.join(LOG_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    for name in ['V1_hard_only', 'V2_50_50', 'V3_75_25', 'V4_curriculum']:
        r = all_results[name]
        print(f"  {name}: AUROC={r['best_auroc']:.4f}, F1={r['best_f1']:.4f}")
    print(f"  Shuffled label check: AUROC={all_results.get('V1_shuffled_auroc', 'N/A')}")


if __name__ == '__main__':
    main()
