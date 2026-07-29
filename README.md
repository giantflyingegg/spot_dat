# spot_dat — branch `detector/v4-signhealth-ft`

Hard-negative fine-tune of the flip-augmented detector onto SignHealth-domain video,
plus the frozen evaluation harness and the deployment operating point.

Branched from [`detector/flip-aug`](../../tree/detector/flip-aug), which is where the
coordinate fixes and the flip augmentation are documented. This branch is the fine-tune
alone: `variant4_signhealth_ft.pt` starts from `variant3_flipaug.pt` (recorded in the
training history as `start_ckpt_md5: b05116190d50`).

**The v4 checkpoint itself is not in this repository.** It is fine-tuned on
SignHealth-derived features and is held in a private companion repository. The training
script, the training history, the extraction rules and every reported number are here.

---

## Hard-negative extraction

SignHealth-domain negatives mined from videos annotated exhaustively, excluding the frozen
test set. Window geometry throughout: **25 frames** (1.0 s at 25 fps), **stride 5** for
negative pools, **158-dim** features.

Flank rules, as implemented in `hard_negatives/extract_signhealth_flanks.py`:

| rule | value | meaning |
|---|---|---|
| R1 `NEAR_S` | **5.0 s** | window centre must lie within 5 s of an annotated span — near-miss negatives, not arbitrary background |
| R2 `MARGIN_S` | **0.5 s** | safety margin around every span; nothing within 0.5 s of a positive can become a negative |
| R3 `HAND_FRAC` | **0.5** | at least 50 % of frames in the window must have hand presence, or the window is discarded |
| R4 `OUTLIER` | 100.0 | per-frame outlier rejection threshold |

**Scope: done-minus-19.** Negatives are drawn only from videos marked exhaustively
annotated, minus the 19 frozen test videos. Exhaustive annotation is what licenses the
negative label — on a partially annotated video an unlabelled region is not evidence of
absence. The exclusion list is loaded from the two freeze JSONs, never hand-maintained.

Three pools: `extract_signhealth_positives.py`, `extract_signhealth_flanks.py`,
`extract_signhealth_easyneg.py`; assembled by `assemble_signhealth_ft.py`, which records
what it consumed in `hard_negatives/sh_ft_assembly.json` — seed 42, train 60,844 windows
(pos 10,811 / flank 28,411 / easy 21,622, positive fraction 0.1777), val 11,362, and the
three pool hashes.

## Dual positive-window convention

Positive windows are built two ways depending on span length, because one convention
cannot serve both (`hard_negatives/extract_signhealth_positives.py`):

- **`L >= 25` frames — standard containment.** 25-frame windows fully *inside* the span,
  stride 3. This is the unchanged BOBSL convention.
- **`5 <= L < 25` frames — inverted containment-jitter.** 25-frame windows that fully
  *contain* the span, jittered at stride 3 across every offset where `span ⊆ window`,
  using real frames only and **never zero-padding**. The label is deliberately impure:
  fingerspelling-frame fraction is `L/25`.

Spans shorter than `MIN_SPAN_F = 5` frames are dropped.

Without the inverted branch, every span under one second contributes no positive window at
all and the detector is trained blind to exactly the class the containment annotation
convention exists to capture. The impure label is the price of seeing them.

## Test-set lockout

The 19 frozen test videos are excluded at **both** ends, from a single source of truth —
`analysis/frozen_test_videos.json` and `analysis/frozen_test_L_supplement.json`:

- **build time** — `queue/build_detection_queue_v4.py` and the flip-aug builder both call
  the shared `lockout()` and log the drop count and survivors on every run. The builder
  **aborts** if the lockout list loads empty or its id count fails to match the declared
  count, rather than proceeding unlocked.
- **serve time** — the annotation server loads the same two JSONs into
  `EXCLUDED_TEST_VIDEOS` via `assert_test_lockout()` and reports the locked set on every
  start.

The belt-and-braces arrangement exists because clips from test videos were found live in
the serving queue before the lockout was added. *(The exact count of leaked clips is
recorded in session notes only; no on-disk artefact carries it, so it is not asserted
here.)*

## Fine-tune

`retrain/train_variant4.py`, starting from `variant3_flipaug.pt`. Config from the
recorded history: lr 1e-4, weight decay 1e-4, batch 128, **50/50 BOBSL/SignHealth mix**,
flip augmentation on (per-window p=0.5), feature scaler **banked, not refit**, early stop
on SignHealth-val F1 with patience 7, max 50 epochs.

| | value |
|---|---|
| best epoch | 8 (stopped at 15) |
| SignHealth val F1 | 0.8571 at epoch 0 → **0.8802** at best (+0.0232) |
| BOBSL val F1 at best | 0.9656 (Δ **−0.0015** vs epoch 0) |
| tripwire | **not triggered** — BOBSL val F1 stayed within 0.03 of epoch 0 |
| wall clock | 27.4 s |

The tripwire is the point: the fine-tune was allowed to proceed only while BOBSL
performance held. It bought +2.3 points of in-domain F1 for 0.15 points of out-of-domain
loss.

---

## Results

Every cell names **checkpoint · test set · post-processing config**.

**PP config `B4`** — `k7 / th0.50 / gap0 / min0.2 / cf0.5 / ext0.1`, IoU 0.3, **GBT off**.
**Test sets:** `R16` = 16 right-handed videos / 447 spans; `L3` = 3 left-handed videos /
123 spans. Micro-pooled.

| checkpoint | test set | P | R | **F1** |
|---|---|---|---|---|
| `variant3_75_25.pt` | R16, B4, GBT off | 0.5253 | 0.7673 | 0.6236 |
| `variant3_flipaug.pt` | R16, B4, GBT off | 0.6017 | 0.8009 | 0.6871 |
| **`variant4_signhealth_ft.pt`** | R16, B4, GBT off | 0.7756 | 0.8121 | **0.7934** |
| `variant3_75_25.pt` | L3, B4, GBT off | 0.2109 | 0.2195 | 0.2151 |
| `variant3_flipaug.pt` | L3, B4, GBT off | 0.5291 | 0.8130 | 0.6410 |
| **`variant4_signhealth_ft.pt`** | L3, B4, GBT off | 0.7165 | 0.7398 | **0.7280** |

Right-handed F1 0.6236 → 0.6871 → 0.7934; left-handed 0.2151 → 0.6410 → 0.7280. The
R−L gap runs 0.4085 → 0.0461 → 0.0654.

### Deployment operating point

**`k3 / th0.30 / gap0 / min0.2 / cf0 / ext0`** — the full config, from
`deploy/regen_signhealth_v4_deploy.py`. Note `cf0` and `ext0`: the confidence filter and
boundary extension are both **off**, which is easy to drop when the point is quoted
informally as "k3/th.3/min.2".

Selected on a **7-video validation set**, not on the test set. The per-video validation
results are held privately, so the selection cannot be re-derived from this repository —
the claim stands on the recorded rationale and on the test figures below, which are
independent of the selection.

Rationale, verbatim from the script: the `min_dur` floor is aligned to the training span
floor (0.2 s = 5 frames) because *"min_dur is a structural exclusion not a soft knob, so a
0.3 s floor would re-blind the queue to sub-0.3s single letters"* — the same class the
containment convention above exists to capture. Recall-priority, per the false-negative
cost argument.

`variant4_signhealth_ft.pt` at DEPLOY, IoU 0.3, GBT off, micro-pooled:

| test set | TP | FP | FN | P | R | **F1** |
|---|---:|---:|---:|---|---|---|
| R16 | 376 | 152 | 71 | 0.7121 | **0.8412** | 0.7713 |
| L3 | 95 | 62 | 28 | 0.6051 | 0.7724 | 0.6786 |
| pooled 19 | 471 | 214 | 99 | 0.6876 | 0.8263 | 0.7506 |

DEPLOY trades ~2 points of F1 against B4 for ~3 points of recall on R16 — the intended
direction for a queue that feeds human review, where a miss costs more than a false alarm.

### The case for running with GBT off

| configuration | test set | **F1** |
|---|---|---|
| `variant3_75_25.pt` + `gbt_v3.pkl`, native banked PP, GBT **on** | R16 | 0.5753 |
| `variant3_flipaug.pt` + `gbt_v3_flipaug.pkl`, native banked PP, GBT **on** | R16 | 0.6142 |
| `variant3_flipaug.pt`, B4, GBT **off** | R16 | 0.6871 |
| `variant4_signhealth_ft.pt`, B4, GBT **off** | R16 | **0.7934** |

Every GBT-on row sits below its GBT-off counterpart. On the earlier BOBSL held-out
evaluation the filter was worth **+0.001 F1** (V3 alone 0.870 → V3+GBT 0.871) while
cutting false positives per minute by 12.4 % — a precision instrument, not an accuracy
one. Once the frame model itself became precise (v4 R16 precision 0.7756 against 0.5253
for the V3 baseline), the filter had little left to remove and was costing recall.

**No GBT-on evaluation exists for any left-handed set, or for v4 on any set.** Any
GBT-on claim beyond the two rows above would have to be computed first.

## Frozen evaluation harness

`eval/eval_570_frozen.py` is the citable evaluator. It reads ground truth from a frozen
JSON snapshot rather than the live annotation database, and scores archived cached
inference probabilities — it does **not** re-run inference. Results are therefore a pure
function of (frozen GT, archived probs) and cannot drift as annotation continues.

`eval/eval_570.py` reads GT live from the mutable database and is retained **only** to
document how the original archived pass was produced. Its numbers are not citable; see
`eval/README_eval570.md`, which also explains why the right-handed count is 447 and not
449 (two zero-duration spans fail the `t_end_s > t_start_s` validity filter).

`eval/test570_probs_provenance.json` records the checkpoints, per-video feature-run dates,
and all three post-processing operating points (`V3_LOCK`, `V4_LOCK`, `B4`) for the
archived inference pass. The frozen GT, the cached probability archives and the result
JSONs are held privately.

Reproduction check, 2026-07-23: `eval_570_frozen.py` reproduced the archived results
**byte-for-byte** — archived and frozen result files share sha256 `d048ad69…`.

---

## Layout

```
retrain/         train_variant4.py            (+ flip-aug branch training code)
models/          variant4_signhealth_ft_history.json   (weights held privately)
hard_negatives/  extract_signhealth_{positives,flanks,easyneg}.py
                 assemble_signhealth_ft.py, sh_ft_assembly.json
eval/            eval_570_frozen.py  (citable) · eval_570.py (provenance only)
                 README_eval570.md · test570_probs_provenance.json
deploy/          regen_signhealth_v4_deploy.py
queue/           build_detection_queue_v4.py  (+ flip-aug builder)
```

## Not in this repository

Five derived training arrays totalling **1.00 GB** are excluded from git entirely, and
**Git LFS is not used or initialised**. They are regenerable by the four
`hard_negatives/` scripts, which take no arguments:

```bash
conda run -n hamer python3 extract_signhealth_positives.py
conda run -n hamer python3 extract_signhealth_flanks.py
conda run -n hamer python3 extract_signhealth_easyneg.py
conda run -n hamer python3 assemble_signhealth_ft.py     # consumes the three above
```

Sizes, sha256s and the input hashes those commands consume are documented in the private
companion repository, because the inputs are SignHealth-derived.

Also held privately: `variant4_signhealth_ft.pt`, the annotation databases, the frozen
ground-truth span sets, the cached inference probabilities, the per-video validation
sweep, and the annotation tool. **All reported numbers appear above in full; the material
behind them is not published.**

## Environment

conda env **`hamer`** · Python **3.10.19** · torch **2.5.0+cu124** · NVIDIA RTX 4080 SUPER
(driver 595.84). Exact package set in [`requirements-frozen.txt`](requirements-frozen.txt),
captured from `hamer` — not from the machine's default environment.

Scripts reference paths from the original working tree and are published as a record of
method, not as a turnkey pipeline.

## Standing rule

Generators, extraction scripts and analysis code are **never** written to `/tmp` or any
scratchpad path. They are written into the project tree and committed. Three artefacts
have already been orphaned this way — the corpus figure generators, the medical-match
generator, and the 2×3 evaluation freeze scripts — and the last of those cost an entire
evaluation cell that can no longer be reproduced.
