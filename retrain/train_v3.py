#!/usr/bin/env python3
"""
Retrain the V3 (75/25 hard-neg) detector — faithful replica of
train_variants_v2.py's recipe, with an optional p=0.5 horizontal-flip
augmentation flag.

Recipe (verbatim from train_variants_v2.py):
  CNN1D (train_models.CNN1D, identical arch), input 158, window 25, seed 42,
  Adam lr=1e-3 wd=1e-4, BCEWithLogitsLoss, ReduceLROnPlateau(max AUROC, p=3, x0.5),
  batch 128, up to 50 epochs, early-stop patience 7, StandardScaler fit on 50k
  random rows then in-place scale. Negatives balanced to n_pos at 75% hard / 25% easy.
  The V3 negative subset is reproduced by REPLAYING the shared RandomState(42)
  draw sequence (V1 -> V2 -> V3) exactly as the banked script consumed it.

Arm 0 (--aug off): control, banked data only -> must reproduce banked behaviour.
Arm 1 (--aug on):  each training window is presented as its banked-orig OR its
  index-aligned flipped counterpart with p=0.5, resampled per window per epoch.
  Flipped windows loaded from retrain/flipped_data/ (built by augment.py).
  Checkpoint selected on min(orig,flip) val F1 (vs AUROC for Arm 0).
"""
import os
import sys
import json
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('~/bsl_project/improved_detection/scripts'))
from train_models import CNN1D  # identical arch to banked variant3

UNIFIED = os.path.expanduser('~/bsl_project/hard_negative_detection/unified_data/')
FLIP_DIR = os.path.join(HERE, 'flipped_data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED, INPUT_DIM, WINDOW_SIZE = 42, 158, 25


def load_pool(name, flip=False):
    d = FLIP_DIR if flip else UNIFIED
    suf = '_flip' if flip else ''
    return np.load(f'{d}/{name}{suf}.npz')['windows']


def v3_subset_indices(n_pos_tr, n_pos_va, lens):
    """Replay the banked shared RandomState(42) draw order (V1,V2,V3) and return
    the V3 train/val (hard_idx, easy_idx)."""
    nh_tr, ne_tr, nh_va, ne_va = lens
    rng = np.random.RandomState(SEED)
    # V1
    rng.choice(nh_tr, size=min(n_pos_tr, nh_tr), replace=False)
    rng.choice(nh_va, size=min(n_pos_va, nh_va), replace=False)
    # V2
    rng.choice(nh_tr, size=min(n_pos_tr // 2, nh_tr), replace=False)
    rng.choice(ne_tr, size=min(n_pos_tr // 2, ne_tr), replace=False)
    rng.choice(nh_va, size=min(n_pos_va // 2, nh_va), replace=False)
    rng.choice(ne_va, size=min(n_pos_va // 2, ne_va), replace=False)
    # V3
    nhard, neasy = int(n_pos_tr * 0.75), n_pos_tr - int(n_pos_tr * 0.75)
    nhard_v, neasy_v = int(n_pos_va * 0.75), n_pos_va - int(n_pos_va * 0.75)
    idx_h3 = rng.choice(nh_tr, size=min(nhard, nh_tr), replace=False)
    idx_e3 = rng.choice(ne_tr, size=min(neasy, ne_tr), replace=False)
    idx_hv3 = rng.choice(nh_va, size=min(nhard_v, nh_va), replace=False)
    idx_ev3 = rng.choice(ne_va, size=min(neasy_v, ne_va), replace=False)
    return idx_h3, idx_e3, idx_hv3, idx_ev3


class AugDS(Dataset):
    def __init__(self, X, Xf, y, aug):
        self.X = torch.from_numpy(X)
        self.Xf = torch.from_numpy(Xf) if Xf is not None else None
        self.y = torch.from_numpy(y.astype(np.float32))
        self.aug = aug and self.Xf is not None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.aug and torch.rand(1).item() < 0.5:
            return self.Xf[i], self.y[i]
        return self.X[i], self.y[i]


def _val_metrics(model, loader):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            ys.extend(yb.numpy())
    ps, ys = np.array(ps), np.array(ys)
    auroc = roc_auc_score(ys, ps)
    pr, rc, th = precision_recall_curve(ys, ps)
    f1s = 2 * pr * rc / (pr + rc + 1e-8)
    j = int(np.argmax(f1s))
    return auroc, float(f1s[j]), float(th[j] if j < len(th) else 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    print(f"=== TRAIN {'AUG' if a.aug else 'CONTROL'} ({a.tag}) device={DEVICE} ===")

    pos_tr = load_pool('positives_train'); pos_va = load_pool('positives_val')
    hard_tr = load_pool('hard_neg_train'); hard_va = load_pool('hard_neg_val')
    easy_tr = load_pool('easy_neg_train'); easy_va = load_pool('easy_neg_val')
    n_pt, n_pv = len(pos_tr), len(pos_va)
    print(f"  pos tr/va={n_pt}/{n_pv} hard={len(hard_tr)}/{len(hard_va)} "
          f"easy={len(easy_tr)}/{len(easy_va)}")

    ih, ie, ihv, iev = v3_subset_indices(
        n_pt, n_pv, (len(hard_tr), len(easy_tr), len(hard_va), len(easy_va)))

    Xtr = np.vstack([pos_tr, hard_tr[ih], easy_tr[ie]]).astype(np.float32)
    ytr = np.concatenate([np.ones(n_pt), np.zeros(len(ih) + len(ie))]).astype(np.float32)
    Xva = np.vstack([pos_va, hard_va[ihv], easy_va[iev]]).astype(np.float32)
    yva = np.concatenate([np.ones(n_pv), np.zeros(len(ihv) + len(iev))]).astype(np.float32)
    print(f"  train {Xtr.shape} ({int(ytr.sum())} pos), val {Xva.shape}")

    Xtrf = Xvaf = None
    if a.aug:
        ptf = load_pool('positives_train', flip=True)
        pvf = load_pool('positives_val', flip=True)
        htf = load_pool('hard_neg_train', flip=True); hvf = load_pool('hard_neg_val', flip=True)
        etf = load_pool('easy_neg_train', flip=True); evf = load_pool('easy_neg_val', flip=True)
        Xtrf = np.vstack([ptf, htf[ih], etf[ie]]).astype(np.float32)
        Xvaf = np.vstack([pvf, hvf[ihv], evf[iev]]).astype(np.float32)
        assert Xtrf.shape == Xtr.shape and Xvaf.shape == Xva.shape
        print(f"  + flipped train {Xtrf.shape}")
        del ptf, pvf, htf, hvf, etf, evf

    # ── scaler: fit on ORIG train (recipe), scale orig+flip in place ──
    scaler = StandardScaler()
    fit_idx = np.random.choice(len(Xtr), size=min(50000, len(Xtr)), replace=False)
    scaler.fit(Xtr[fit_idx].reshape(-1, INPUT_DIM))
    for arr in [Xtr, Xva] + ([Xtrf, Xvaf] if a.aug else []):
        arr.reshape(-1, INPUT_DIM).__isub__(scaler.mean_).__itruediv__(scaler.scale_ + 1e-8)

    tr_loader = DataLoader(AugDS(Xtr, Xtrf, ytr, a.aug), batch_size=128,
                           shuffle=True, num_workers=0, pin_memory=True)
    vo_loader = DataLoader(AugDS(Xva, None, yva, False), batch_size=128, shuffle=False)
    vf_loader = (DataLoader(AugDS(Xvaf, None, yva, False), batch_size=128, shuffle=False)
                 if a.aug else None)

    model = CNN1D(input_dim=INPUT_DIM).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5, mode='max')

    best_sel, best_state, best_meta, no_imp, hist = -1, None, None, 0, []
    for ep in range(a.epochs):
        model.train(); tl = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item()
        tl /= len(tr_loader)
        au_o, f1_o, th_o = _val_metrics(model, vo_loader)
        if a.aug:
            au_f, f1_f, th_f = _val_metrics(model, vf_loader)
            sel = min(f1_o, f1_f)            # select on worst orientation
        else:
            au_f = f1_f = th_f = None
            sel = au_o                       # original recipe: select on AUROC
        sched.step(au_o)
        hist.append({'epoch': ep + 1, 'loss': tl, 'auroc_orig': au_o, 'f1_orig': f1_o,
                     'auroc_flip': au_f, 'f1_flip': f1_f, 'sel': sel})
        if sel > best_sel:
            best_sel, no_imp = sel, 0
            best_state = deepcopy(model.state_dict())
            best_meta = {'auroc': au_o, 'best_f1': f1_o, 'best_threshold': th_o,
                         'f1_flip': f1_f, 'auroc_flip': au_f}
        else:
            no_imp += 1
        if (ep + 1) % 2 == 0 or ep == 0 or no_imp == 0:
            msg = f"  ep{ep+1:3d} loss={tl:.4f} AUROC_o={au_o:.4f} F1_o={f1_o:.4f}"
            if a.aug:
                msg += f" F1_flip={f1_f:.4f} sel={sel:.4f}"
            print(msg + (' *' if no_imp == 0 else ''))
        if no_imp >= 7:
            print(f"  early stop @ {ep+1}"); break

    model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
                'input_dim': INPUT_DIM, 'window_size': WINDOW_SIZE, 'model_type': 'CNN1D',
                'auroc': best_meta['auroc'], 'best_f1': best_meta['best_f1'],
                'best_threshold': best_meta['best_threshold'],
                'variant': f'75_25_flipaug={a.aug}', 'data_source': 'unified+flipaug'},
               a.out)
    json.dump(hist, open(a.out.replace('.pt', '_history.json'), 'w'), indent=2)
    print(f"  saved {a.out}  best_sel={best_sel:.4f} "
          f"(AUROC_o={best_meta['auroc']:.4f} F1_o={best_meta['best_f1']:.4f}"
          + (f" F1_flip={best_meta['f1_flip']:.4f}" if a.aug else "") + ")")


if __name__ == '__main__':
    main()
