#!/usr/bin/env python3
"""
Stage 3b-2: fine-tune variant3_flipaug.pt -> variant4_signhealth_ft.pt on
BOBSL(banked) + SignHealth, 50/50 mixed batches, flip-aug ON.

Key fidelity points:
  * BANKED scaler (variant3_flipaug.pt's scaler_mean/scale) — NOT refit
    (aug_config refit_feature_scaler=false); pools are raw, normalised with it.
  * BOBSL = the banked v3 subset (pos + 0.75*pos hard + 0.25*pos easy), + flips.
  * Each batch: 64 SignHealth + 64 BOBSL; flip-aug = per-window 50% flipped.
  * Early-stop on SignHealth-val F1 ALONE (patience 7, <=50 ep).
  * BOBSL-val F1 logged every epoch (forgetting monitor, NOT optimised);
    tripwire: flag if it ever drops >0.03 below epoch-0 (pre-FT).
  * Epoch-0 eval (both val sets through variant3 as loaded) BEFORE any step.
New file only; variant3_flipaug.pt + banked pools untouched.
"""
import os, sys, json, time, hashlib
from copy import deepcopy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, '/home/gfe/bsl_project/improved_detection/scripts')
sys.path.insert(0, '/home/gfe/bsl_project/robustness_diag/retrain')
from train_models import CNN1D
import train_v3 as TV

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; LR = 1e-4; WD = 1e-4; BATCH_HALF = 64; EPOCHS = 50; PATIENCE = 7
TRIPWIRE = 0.03
CKPT = '/home/gfe/bsl_project/robustness_diag/retrain/variant3_flipaug.pt'
OUT = '/home/gfe/bsl_project/robustness_diag/retrain/variant4_signhealth_ft.pt'
HND = '/home/gfe/bsl_project/hard_negative_detection'
HIST_OUT = '/home/gfe/bsl_project/robustness_diag/retrain/variant4_signhealth_ft_history.json'


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    smean = np.asarray(ck['scaler_mean'], dtype=np.float32)
    sscale = np.asarray(ck['scaler_scale'], dtype=np.float32)

    def norm(X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        X.reshape(-1, 158).__isub__(smean).__itruediv__(sscale + 1e-8)
        return X

    print("loading BOBSL banked pools + flips...", flush=True)
    pos_tr = TV.load_pool('positives_train'); hard_tr = TV.load_pool('hard_neg_train'); easy_tr = TV.load_pool('easy_neg_train')
    pos_va = TV.load_pool('positives_val'); hard_va = TV.load_pool('hard_neg_val'); easy_va = TV.load_pool('easy_neg_val')
    n_pt, n_pv = len(pos_tr), len(pos_va)
    ih, ie, ihv, iev = TV.v3_subset_indices(n_pt, n_pv, (len(hard_tr), len(easy_tr), len(hard_va), len(easy_va)))
    Xb = norm(np.vstack([pos_tr, hard_tr[ih], easy_tr[ie]]))
    yb = np.concatenate([np.ones(n_pt), np.zeros(len(ih) + len(ie))]).astype(np.float32)
    Xb_va = norm(np.vstack([pos_va, hard_va[ihv], easy_va[iev]]))
    yb_va = np.concatenate([np.ones(n_pv), np.zeros(len(ihv) + len(iev))]).astype(np.float32)
    ptf = TV.load_pool('positives_train', flip=True); htf = TV.load_pool('hard_neg_train', flip=True); etf = TV.load_pool('easy_neg_train', flip=True)
    Xbf = norm(np.vstack([ptf, htf[ih], etf[ie]]))
    del pos_tr, hard_tr, easy_tr, ptf, htf, etf, pos_va, hard_va, easy_va

    print("loading SignHealth FT arrays...", flush=True)
    sh = np.load(f'{HND}/sh_ft_train.npz'); Xs = norm(sh['X']); Xsf = norm(sh['Xf']); ys = sh['y'].astype(np.float32)
    shv = np.load(f'{HND}/sh_ft_val.npz'); Xs_va = norm(shv['X']); ys_va = shv['y'].astype(np.float32)
    print(f"  BOBSL train {Xb.shape} val {Xb_va.shape} | SignHealth train {Xs.shape} val {Xs_va.shape}", flush=True)

    model = CNN1D(input_dim=158).to(DEVICE)
    model.load_state_dict(ck['model_state_dict'])

    def best_f1(X, y):
        model.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.from_numpy(X[i:i + 8192]).to(DEVICE)
                ps.append(torch.sigmoid(model(xb)).cpu().numpy())
        p = np.concatenate(ps)
        pr, rc, _ = precision_recall_curve(y, p)
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        return float(np.nanmax(f1))

    # ── epoch 0 (pre-FT baseline) ──
    sh0 = best_f1(Xs_va, ys_va); bo0 = best_f1(Xb_va, yb_va)
    print(f"\n[epoch 0 / variant3 as-loaded]  SignHealth-val F1={sh0:.4f}   BOBSL-val F1={bo0:.4f}\n", flush=True)

    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5, mode='max')
    rng = np.random.RandomState(SEED)

    hist = [{'epoch': 0, 'sh_val_f1': sh0, 'bobsl_val_f1': bo0, 'loss': None}]
    best_f1v, best_state, no_imp = sh0, deepcopy(model.state_dict()), 0
    best_ep = 0
    tripwire_hit = None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Xs)); nb = len(order) // BATCH_HALF
        tl = 0.0
        for b in range(nb):
            si = order[b * BATCH_HALF:(b + 1) * BATCH_HALF]
            bi = rng.randint(0, len(Xb), size=BATCH_HALF)
            sfl = rng.rand(BATCH_HALF) < 0.5; bfl = rng.rand(BATCH_HALF) < 0.5
            xs = np.where(sfl[:, None, None], Xsf[si], Xs[si])
            xbo = np.where(bfl[:, None, None], Xbf[bi], Xb[bi])
            X = np.concatenate([xs, xbo]); Y = np.concatenate([ys[si], yb[bi]])
            xb = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
            yt = torch.from_numpy(Y).to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yt); loss.backward(); opt.step()
            tl += loss.item()
        tl /= nb
        sh_f1 = best_f1(Xs_va, ys_va); bo_f1 = best_f1(Xb_va, yb_va)
        sched.step(sh_f1)
        drop = bo0 - bo_f1
        if drop > TRIPWIRE and tripwire_hit is None:
            tripwire_hit = {'epoch': ep, 'bobsl_val_f1': bo_f1, 'drop': round(drop, 4)}
        hist.append({'epoch': ep, 'loss': round(tl, 4), 'sh_val_f1': round(sh_f1, 4),
                     'bobsl_val_f1': round(bo_f1, 4), 'bobsl_drop_vs_ep0': round(drop, 4)})
        flag = ' TRIPWIRE!' if drop > TRIPWIRE else ''
        star = ''
        if sh_f1 > best_f1v:
            best_f1v, best_state, no_imp, best_ep = sh_f1, deepcopy(model.state_dict()), 0, ep
            star = ' *'
        else:
            no_imp += 1
        print(f"  ep{ep:2d} loss={tl:.4f}  SH-val F1={sh_f1:.4f}  BOBSL-val F1={bo_f1:.4f} (Δ{-drop:+.4f}){flag}{star}", flush=True)
        if no_imp >= PATIENCE:
            print(f"  early stop @ ep{ep} (SignHealth-val F1 patience {PATIENCE})"); break
    wall = time.time() - t0

    model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(),
                'scaler_mean': ck['scaler_mean'], 'scaler_scale': ck['scaler_scale'],
                'input_dim': 158, 'window_size': 25, 'model_type': 'CNN1D',
                'variant': 'variant4_signhealth_ft', 'data_source': 'BOBSL_banked+SignHealth_v1',
                'best_f1_sh_val': best_f1v, 'best_epoch': best_ep}, OUT)
    prov = {
        'output': OUT, 'start_checkpoint': CKPT, 'start_ckpt_md5': md5(CKPT),
        'config': {'lr': LR, 'wd': WD, 'batch': 2 * BATCH_HALF, 'mix': '50/50 BOBSL/SignHealth',
                   'flip_aug': 'ON (per-window 50%)', 'scaler': 'banked (not refit)',
                   'early_stop': 'SignHealth-val F1, patience 7', 'epochs_max': EPOCHS},
        'pool_hashes': {**{f'BOBSL_{p}': md5(f'{os.path.expanduser("~/bsl_project/hard_negative_detection/unified_data")}/{p}.npz')
                           for p in ['positives_train', 'hard_neg_train', 'easy_neg_train']},
                        **{f'SH_{p}': md5(f'{HND}/signhealth_{p}_v1.npz') for p in ['positives', 'flanks', 'easyneg']}},
        'sh_ft_assembly': json.load(open(f'{HND}/sh_ft_assembly.json')),
        'convention': 'DUAL positives (standard>=25f + containment-jitter 5-25f); flanks R1-4; easy BOBSL-conv; done-only negatives',
        'epoch0': {'sh_val_f1': sh0, 'bobsl_val_f1': bo0},
        'result': {'best_epoch': best_ep, 'best_sh_val_f1': best_f1v,
                   'sh_val_f1_gain_vs_ep0': round(best_f1v - sh0, 4),
                   'bobsl_val_f1_at_best': hist[best_ep]['bobsl_val_f1'],
                   'bobsl_delta_at_best': round(hist[best_ep]['bobsl_val_f1'] - bo0, 4),
                   'stopping_epoch': hist[-1]['epoch'], 'wall_clock_s': round(wall, 1)},
        'tripwire': tripwire_hit or 'not triggered (BOBSL-val F1 stayed within 0.03 of epoch-0)',
    }
    json.dump({'provenance': prov, 'history': hist}, open(HIST_OUT, 'w'), indent=2)
    print("\n" + "=" * 64)
    print(f"BEST epoch {best_ep}: SH-val F1 {best_f1v:.4f} (ep0 {sh0:.4f}, +{best_f1v-sh0:.4f})")
    print(f"BOBSL-val F1 at best: {hist[best_ep]['bobsl_val_f1']:.4f} (ep0 {bo0:.4f}, Δ{hist[best_ep]['bobsl_val_f1']-bo0:+.4f})")
    print(f"tripwire: {prov['tripwire']}")
    print(f"wall-clock: {wall:.1f}s  ({(wall)/60:.1f} min)")
    print(f"saved {OUT}\nsaved {HIST_OUT}")


if __name__ == '__main__':
    main()
