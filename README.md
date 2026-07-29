# spot_dat — branch `detector/flip-aug`

Flip augmentation for the BSL fingerspelling detector, applied at the **raw landmark
stage**, together with the two coordinate-convention fixes it depends on.

Baseline this branch modifies: `main` (V3 detector, `models/variant3_75_25.pt`). The
original V3 documentation is preserved unchanged as
[`README_v3_baseline.md`](README_v3_baseline.md).

Downstream branch: `detector/v4-signhealth-ft`, which fine-tunes *from* the checkpoint
produced here.

---

## What this branch changes vs `main`

| | `main` | this branch |
|---|---|---|
| coordinate handling | hand blocks occasionally swapped; anisotropic x/z | wrist-probe de-swap **+** isotropy correction, applied together |
| augmentation | none | horizontal flip at raw-landmark stage, per-window p=0.5 |
| GBT filter | `gbt_v3.pkl`, fitted on one orientation | `gbt_v3_flipaug.pkl`, refitted across both orientations |
| left-handed signers | effectively undetected | detected at F1 0.6410 |

New checkpoints: `models/variant3_flipaug.pt` (the headline), `models/variant3_control.pt`
(same recipe, augmentation off — the control), `models/gbt_v3_flipaug.pkl`.

---

## The two coordinate bugs — and why they had to be fixed together

Two independent defects in the landmark → feature path:

1. **Hand-block swap.** Left and right hand blocks were intermittently transposed in the
   landmark array. Corrected by a **wrist probe**: comparing block-1566 against pose
   landmark 15 to establish which block is which, then de-swapping. Implemented in
   `regen/regen_transforms.py`; each regenerated feature file records
   `wrist_probe_block1566_to_pose15` and the number of valid probe frames in its
   `.provenance.json` sidecar.

2. **Anisotropy.** Landmark x and z were normalised against a different extent than y,
   so the coordinate space was stretched by the source aspect ratio. Corrected by
   `x, z *= W/H`, with `W/H` read per video from `ffprobe`.

> **They must be applied together.** Fixing isotropy alone — de-swap still absent — drove
> validation F1 to **0.022**. The isotropy correction makes the geometry metrically
> honest, which makes a swapped hand block a far worse error than it was in the distorted
> space: the model had been partly absorbing the swap as noise, and could no longer.
> Anyone porting one fix without the other should expect the detector to collapse, not
> to degrade gracefully.

The de-swap is applied to the **raw landmarks**, before feature extraction, so the flip
augmentation below composes with it cleanly.

## Flip augmentation — raw landmark stage, not feature vector

The flip is `transforms.hflip(lm_dsw)` on de-swapped raw landmarks, after which the full
158-dim feature extractor runs again on the flipped landmarks. It is **not** a permutation
or sign-flip applied to an already-built feature vector.

This matters because the 158-dim vector contains derived quantities — inter-hand
distances, motion deltas, presence flags — whose correct values under reflection are not
recoverable by reindexing the unflipped vector. Flipping late produces a vector that is
self-inconsistent; flipping early produces a genuine mirrored example.

Configuration (`retrain/aug_config_stage1.json`): flip enabled, `p=0.5`, coin flipped
**per window**; scale and speed augmentation off for this arm; feature scaler **not**
refitted (banked). Model selection used `min(orig_val_F1, flip_val_F1)` so a checkpoint
could not win by being good in one orientation only.

**The GBT filter is refitted across both orientations** (`retrain/train_gbt_flipaug.py`
→ `models/gbt_v3_flipaug.pkl`). Reusing the original `gbt_v3.pkl` would have left a
segment-level filter tuned on right-handed segment statistics sitting downstream of an
orientation-agnostic frame model.

---

## Results

Every cell below names **checkpoint · test set · post-processing config**. No number in
this repository is quoted without all three.

**Post-processing config `B4`** — `smooth_kernel=7, threshold=0.50, gap_bridge_s=0.0,
min_duration_s=0.2, conf_filter=0.5, boundary_extend_s=0.1`. Matching at **IoU 0.3**.
**GBT off.**

**Test sets.** `R16` = 16 right-handed videos, 447 ground-truth spans. `L3` = 3
left-handed videos, 123 spans. Both frozen; the ground truth itself is held privately.
Figures are micro-pooled (TP/FP/FN summed across videos, then P/R/F1).

| checkpoint | test set | P | R | **F1** |
|---|---|---|---|---|
| `variant3_75_25.pt` (`main` baseline) | R16, B4, GBT off | 0.5253 | 0.7673 | **0.6236** |
| `variant3_75_25.pt` | L3, B4, GBT off | 0.2109 | 0.2195 | **0.2151** |
| **`variant3_flipaug.pt`** | R16, B4, GBT off | 0.6017 | 0.8009 | **0.6871** |
| **`variant3_flipaug.pt`** | L3, B4, GBT off | 0.5291 | 0.8130 | **0.6410** |

The right-handed gain is real but modest, +0.0635 F1. The left-handed change is the
result: **0.2151 → 0.6410**, and the R−L gap closes from 0.4085 to 0.0461.

Read the baseline's L3 row honestly: at P 0.2109 / R 0.2195 the unaugmented detector is
not weak on left-handed signers, it is close to non-functional on them.

### With the GBT filter on

**Post-processing config `native banked`** — `k11 / th0.30 / gap0.5 / min0.5 / cf0.60 /
ext0.2`, GBT keep rule `predict_proba[TP] >= 0.5`. R16 only; **no GBT-on evaluation exists
for any left-handed set.**

| checkpoint + filter | test set | P | R | **F1** |
|---|---|---|---|---|
| `variant3_75_25.pt` + `gbt_v3.pkl` | R16, native banked, GBT on | 0.6351 | 0.5257 | **0.5753** |
| `variant3_flipaug.pt` + `gbt_v3_flipaug.pkl` | R16, native banked, GBT on | 0.7097 | 0.5414 | **0.6142** |

The refitted filter improves precision markedly (0.6351 → 0.7097) at similar recall. Both
GBT-on rows sit below their GBT-off counterparts on F1, which is the beginning of the
case for running the deployed detector with the filter off — the argument is completed on
`detector/v4-signhealth-ft`.

---

## Layout

```
retrain/   train_v3.py, train_gbt_flipaug.py, augment.py, eval_detector.py,
           aug_config_stage1.json
models/    variant3_flipaug.pt, variant3_control.pt, gbt_v3_flipaug.pkl,
           variant3_{flipaug,control}_history.json
           (variant3_75_25.pt and gbt_v3.pkl carried over from main)
regen/     regen_transforms.py       de-swap + isotropy + 25 fps resample
           regen_signhealth_flipaug.py
queue/     build_detection_queue_flipaug.py
eval/      sweep_pp_val.py           post-processing grid sweep
```

`models/variant3_flipaug_history.json` and `variant3_control_history.json` carry the
per-epoch training curves and provenance for the two arms.

## Not in this repository

No large arrays are committed anywhere, and **Git LFS is not used or initialised**. The
training pools are derived and regenerable; the scripts that regenerate them are on
`detector/v4-signhealth-ft`. Sizes, hashes, regeneration commands and input hashes are
documented in the companion private repository, because the inputs are SignHealth-derived.

Ground-truth span sets, annotation databases, cached inference probabilities and the
per-video validation sweep results are held privately. **The numbers above are reported in
full; the material behind them is not published.**

## Environment

conda env **`hamer`** · Python **3.10.19** · torch **2.5.0+cu124** · NVIDIA RTX 4080 SUPER
(driver 595.84). Exact package set in [`requirements-frozen.txt`](requirements-frozen.txt),
captured from `hamer` — not from the machine's default environment, which is a different
Python and a different torch.

Scripts reference paths from the original working tree and are published as a record of
method, not as a turnkey pipeline.

## Standing rule

Generators, extraction scripts and analysis code are **never** written to `/tmp` or any
scratchpad path. They are written into the project tree and committed. Three artefacts
have already been orphaned this way — the corpus figure generators, the medical-match
generator, and the 2×3 evaluation freeze scripts — and the last of those cost an entire
evaluation cell that can no longer be reproduced.
