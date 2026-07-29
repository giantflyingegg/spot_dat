# frozen-570 eval — canonical vs provenance

**TL;DR — cite the frozen path. Do not report numbers from the live-DB script.**

## Canonical (citable) eval

- **`eval_570_frozen.py`** + **`test570_gt_frozen_20260723.json`**
  - GT is read from the frozen JSON, **not** the database, so it cannot drift as
    annotations change.
  - GT payload **sha256 `85dc158ddc3694c59abb56564e0bfd2ff11fa1178f5aae9e3b3b0c79a1dd0fba`**
    (`85dc158d…`) — 19 videos (16 R + 3 L supplement), the exact eval query
    `t_end_s > t_start_s`, **570 spans** (447 R + 123 L).
  - Runs against the archived 2026-07-19 cached probs
    (`test570_probs_variant{3,4}.npz`); it does **not** re-run inference.
  - Writes `test570_results_frozen.json`.
  - **Use this for all reportable / paper numbers.**

## Provenance only — do NOT use for reportable numbers

- **`eval_570.py`** reads GT **live from the mutable DB**
  (`/home/gfe/bsl_project/wife_drypass/wife_drypass.db`) via
  `SELECT t_start_s,t_end_s FROM spans WHERE video_id=? AND t_end_s>t_start_s`.
  Its GT count — and therefore every metric — shifts whenever spans are added,
  edited, or deleted. Retained to document how the original archived pass was
  produced. **Not citable.**

## Reproduction check (2026-07-23)

`eval_570_frozen.py` reproduced the archived results **byte-for-byte**:

- archived `test570_results.json` sha256
  **`d048ad692cc8fa48602508c69ef1a446fed773cca2a640eaf86fd59a57e2fa14`** (`d048ad69…`)
- `test570_results_frozen.json` sha256 — **identical** (`d048ad69…`);
  `diff` empty, JSON semantic compare equal (0 differing leaves).
- The archive (`test570_results.json`, `test570_probs_*.npz`,
  `test570_probs_provenance.json`) was not modified; pre-work copies exist as
  `*.bak.CCgtfreeze_20260723`.

## Note

Per-video GT counts in the freeze reflect the `t_end_s > t_start_s` validity
filter, so two videos differ from raw span counts by 1 (`3OQrBoj-3E0` 68 vs 69,
`3uFZS1_wMfI` 48 vs 49) — two zero-duration R spans are dropped. This is why the
R side is 447 (not 449) and the total is 570.
