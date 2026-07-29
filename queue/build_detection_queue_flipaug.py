#!/usr/bin/env python3
"""
Rebuild the SignHealth detection triage queue from the CORRECTED detections
(detections_v3_flipaug.json: flip-aug + de-swap + iso, B4 recall, GBT off).

Clears the old detection rows (segment_idx >= 193) from queue + clip_serve
(the original 74 FS clips, segment_idx < 193, stay untouched), then loads the
new detections through the filter cascade and inserts survivors.

det_score = mean_prob  (segment CNN confidence; GBT is OFF so there is no
gbt_score — mean_prob is the validated ranking signal).

Filter cascade (in order, each reports drop + survivors):
  0. resolve video_id -> parent MP4 (needed for clip_serve.parent_video_path)
  1. drop videos with video_status.status='done'
  2. drop videos with video_status.status='rejected'
  3. drop detections overlapping an existing span at IoU > 0.3
  4. keep duration >= 3.0
  5. rank by det_score (mean_prob) DESC   [no duration-proximity sort]

Insert (only with --commit):
  new segment_idx continues from the historical MAX across queue/clip_serve/
  verdicts/triage/spans (+1) so new clips never collide with retained triage/
  verdict rows. ord continues from the original-74 max (73)+1. served window
  = [max(0,t_start-3), t_end+3]; orig_win = [t_start,t_end];
  timing_source='v3_flipaug_detection', flag='detection'.

Read-only w.r.t. verdicts / qa_oracle / spans. Guards the spans hash.
"""
import os, sys, json, glob, sqlite3, hashlib

ROOT = '/home/gfe/bsl_project/wife_drypass'
DB = os.path.join(ROOT, 'wife_drypass.db')
DET_GLOB = '/home/gfe/bsl_project/youtube_mining/output_signhealth/*/detections_v3_flipaug.json'
VIDEOS_DIR = '/home/gfe/bsl_project/youtube_mining/videos'
KEEPSET_CSV = '/home/gfe/bsl_project/youtube_mining/signhealth_keepset.csv'
# ── FROZEN TEST-SET LOCKOUT (single source of truth: the freeze JSONs) ────────
# The held-out detector TEST videos must never enter a queue build — a rebuild
# (e.g. the Stage-3 v4 diff run) would otherwise leave test-video rows sitting
# inert in queue/clip_serve. Serve-time filtering in server.py is the real
# enforcement; this is belt-and-suspenders so the rows never land at all.
FREEZE_TEST_JSONS = (
    '/home/gfe/bsl_project/analysis/frozen_test_videos.json',
    '/home/gfe/bsl_project/analysis/frozen_test_L_supplement.json',
)

DUR_MAIN = 3.0
IOU_DROP = 0.3
PAD = 3.0
KEEP_SEG_BELOW = 193          # rows < 193 (the original 74) are untouched
TIMING_SOURCE = 'v3_flipaug_detection'
COMMIT = '--commit' in sys.argv


def log(*a): print(*a, flush=True)


def iou(a0, a1, b0, b1):
    lo = max(a0, b0); hi = min(a1, b1)
    inter = max(0.0, hi - lo)
    union = (a1 - a0) + (b1 - b0) - inter
    return (inter / union) if union > 0 else 0.0


def load_excluded_test_videos():
    """Load the frozen TEST video_ids from the freeze JSONs. Fails LOUD (SystemExit)
    if a file is missing or the set is empty — we must never rebuild the queue
    without the lockout, or test-video rows leak back in."""
    ids, declared = set(), 0
    for pth in FREEZE_TEST_JSONS:
        if not os.path.exists(pth):
            sys.exit(f"ABORT: frozen-test JSON missing: {pth}  (refusing to build un-locked)")
        with open(pth) as f:
            doc = json.load(f)
        vids = [v['video_id'] for v in doc.get('videos', []) if v.get('video_id')]
        declared += int(doc.get('n_videos', len(vids)))
        ids.update(vids)
    if not ids:
        sys.exit("ABORT: frozen-test lockout list is EMPTY — refusing to build un-locked.")
    if len(ids) != declared:
        sys.exit(f"ABORT: test-lockout id count mismatch (loaded {len(ids)} != declared {declared})")
    return ids


def load_keepset_paths():
    import csv
    by_id = {}
    try:
        with open(KEEPSET_CSV, newline='') as f:
            for r in csv.DictReader(f):
                vid = (r.get('video_id') or '').strip()
                if vid:
                    by_id[vid] = (r.get('local_path') or '').strip()
    except FileNotFoundError:
        pass
    return by_id


def resolve_path(video_id, keepset):
    cand = os.path.join(VIDEOS_DIR, f'{video_id}.mp4')
    if os.path.exists(cand):
        return cand
    lp = keepset.get(video_id)
    if lp and os.path.exists(lp):
        return lp
    return None


def spans_hash(c):
    rows = c.execute("SELECT id,segment_idx,event_index,t_start_s,t_end_s,"
                     "spelled_word,created_at,annotator,video_id,comment "
                     "FROM spans ORDER BY id").fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    import math
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def main():
    keepset = load_keepset_paths()
    excluded_test = load_excluded_test_videos()

    # ── STEP 0: load + resolve ──
    files = sorted(glob.glob(DET_GLOB))
    dets = []
    unresolved = {}
    raw_total = 0
    for f in files:
        d = json.load(open(f))
        vid = d.get('video_id') or os.path.basename(os.path.dirname(f))
        path = resolve_path(vid, keepset)
        for e in d.get('detections', []):
            raw_total += 1
            ts = float(e['start_time']); te = float(e['end_time'])
            dur = float(e.get('duration', te - ts))
            score = float(e['mean_prob'])          # det_score = mean_prob (GBT off)
            if path is None:
                unresolved[vid] = unresolved.get(vid, 0) + 1
                continue
            dets.append({'video_id': vid, 't_start': ts, 't_end': te,
                         'duration': dur, 'score': score, 'path': path})
    log('=' * 72)
    log(f'STEP 0  raw detections            : {raw_total}   (from {len(files)} video files)')
    log(f'        unresolved (no parent MP4): -{sum(unresolved.values()):<5} '
        f'across {len(unresolved)} video_ids   survivors: {len(dets)}')

    # ── LOCKOUT: drop frozen TEST videos (single source of truth: freeze JSONs).
    #    Independent of video_status — a test video must never enter the queue even
    #    if it were (wrongly) un-done'd. Runs before the done/rejected filters. ──
    before = len(dets)
    hit = {d['video_id'] for d in dets if d['video_id'] in excluded_test}
    dets = [d for d in dets if d['video_id'] not in excluded_test]
    log(f'LOCKOUT drop frozen TEST videos   : -{before-len(dets):<5} survivors: {len(dets)}  '
        f'({len(excluded_test)} locked ids; {len(hit)} had detections this build)')

    # ── DB-derived filter inputs (all read before any write) ──
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    done = {r['video_id'] for r in con.execute("SELECT video_id FROM video_status WHERE status='done'")}
    rejected = {r['video_id'] for r in con.execute("SELECT video_id FROM video_status WHERE status='rejected'")}
    cs_vid = {r['segment_idx']: r['video_id'] for r in con.execute("SELECT segment_idx,video_id FROM clip_serve")}
    spans_by_vid = {}
    n_span_used = 0
    for r in con.execute("SELECT segment_idx,video_id,t_start_s,t_end_s FROM spans"):
        if r['t_start_s'] is None or r['t_end_s'] is None:
            continue
        v = r['video_id'] or cs_vid.get(r['segment_idx'])
        if not v:
            continue
        spans_by_vid.setdefault(v, []).append((r['t_start_s'], r['t_end_s']))
        n_span_used += 1
    # historical max segment_idx across ALL tables -> new ids never collide
    seg_hist = max(
        con.execute("SELECT IFNULL(MAX(segment_idx),0) FROM queue").fetchone()[0],
        con.execute("SELECT IFNULL(MAX(segment_idx),0) FROM clip_serve").fetchone()[0],
        con.execute("SELECT IFNULL(MAX(segment_idx),0) FROM verdicts").fetchone()[0],
        con.execute("SELECT IFNULL(MAX(segment_idx),0) FROM triage").fetchone()[0],
        con.execute("SELECT IFNULL(MAX(segment_idx),0) FROM spans").fetchone()[0],
    )
    ord_orig_max = con.execute(
        f"SELECT IFNULL(MAX(ord),-1) FROM queue WHERE segment_idx < {KEEP_SEG_BELOW}").fetchone()[0]
    con.close()

    # ── STEP 1: drop done ──
    before = len(dets)
    dets = [d for d in dets if d['video_id'] not in done]
    log(f'STEP 1  drop status=done          : -{before-len(dets):<5} survivors: {len(dets)}  '
        f'({len(done)} done videos)')
    # ── STEP 2: drop rejected ──
    before = len(dets)
    dets = [d for d in dets if d['video_id'] not in rejected]
    log(f'STEP 2  drop status=rejected      : -{before-len(dets):<5} survivors: {len(dets)}  '
        f'({len(rejected)} rejected videos)')
    # ── STEP 3: drop IoU>0.3 overlap w/ existing span ──
    before = len(dets)
    dets = [d for d in dets
            if not any(iou(d['t_start'], d['t_end'], s0, s1) > IOU_DROP
                       for s0, s1 in spans_by_vid.get(d['video_id'], []))]
    log(f'STEP 3  drop IoU>{IOU_DROP} span overlap : -{before-len(dets):<5} survivors: {len(dets)}  '
        f'({n_span_used} spans across {len(spans_by_vid)} videos)')
    # ── STEP 4: duration >= 3.0 ──
    before = len(dets)
    survivors = [d for d in dets if d['duration'] >= DUR_MAIN]
    log(f'STEP 4  keep duration >= {DUR_MAIN}     : -{before-len(survivors):<5} survivors: {len(survivors)}')
    # ── STEP 5: rank by det_score (mean_prob) desc ──
    survivors.sort(key=lambda d: (-d['score'], d['video_id'], d['t_start']))
    log(f'STEP 5  ranked by det_score desc  : {len(survivors)}')

    scores = [d['score'] for d in survivors]
    if scores:
        log(f'        det_score dist  min={min(scores):.4f} p25={pctl(scores,25):.4f} '
            f'median={pctl(scores,50):.4f} p75={pctl(scores,75):.4f} max={max(scores):.4f}')
    seg_start = seg_hist + 1
    ord_start = ord_orig_max + 1
    log('-' * 72)
    log(f'  historical MAX(segment_idx) across all tables = {seg_hist}  -> new seg {seg_start}..{seg_start+len(survivors)-1}')
    log(f'  original-74 MAX(ord) = {ord_orig_max}  -> new ord {ord_start}..{ord_start+len(survivors)-1}')
    log('  NEW TOP 20 (ord, seg, video_id, det_score, duration):')
    for i, d in enumerate(survivors[:20]):
        log(f'    ord={ord_start+i:<4} seg={seg_start+i:<5} {d["video_id"]:<14} '
            f'score={d["score"]:.4f} dur={d["duration"]:.2f}')

    if not COMMIT:
        log('  DRY RUN — no DB writes. Re-run with --commit.')
        return

    # ── COMMIT: clear old detection rows, insert survivors (single transaction) ──
    con = sqlite3.connect(DB)
    h_before = spans_hash(con)
    nverd_b = con.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    nqa_b = con.execute("SELECT COUNT(*) FROM qa_oracle").fetchone()[0]
    nspan_b = con.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    try:
        con.execute("BEGIN")
        con.execute(f"DELETE FROM queue WHERE segment_idx >= {KEEP_SEG_BELOW}")
        con.execute(f"DELETE FROM clip_serve WHERE segment_idx >= {KEEP_SEG_BELOW}")
        seg = seg_start - 1; ordn = ord_start - 1
        for d in survivors:
            seg += 1; ordn += 1
            clip_id = f'det_{d["video_id"]}_{d["t_start"]:.2f}'
            con.execute("""INSERT INTO queue
                (segment_idx,canonical,surface_form,match_tier,timing_source,video_id,
                 signer,t_start,raw_event_dur_s,det_score,flag,reel_timestamp,clen,clip_id,
                 human_confirmed_word,ord)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (seg, None, None, None, TIMING_SOURCE, d['video_id'],
                 d['video_id'], d['t_start'], d['duration'], d['score'], 'detection',
                 None, None, clip_id, 0, ordn))
            con.execute("""INSERT INTO clip_serve
                (segment_idx,video_id,parent_video_path,orig_win_s,orig_win_e,served_s,served_e)
                VALUES (?,?,?,?,?,?,?)""",
                (seg, d['video_id'], d['path'], d['t_start'], d['t_end'],
                 max(0.0, d['t_start'] - PAD), d['t_end'] + PAD))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK"); con.close(); raise

    h_after = spans_hash(con)
    nverd_a = con.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    nqa_a = con.execute("SELECT COUNT(*) FROM qa_oracle").fetchone()[0]
    nspan_a = con.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    nq = con.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    ncs = con.execute("SELECT COUNT(*) FROM clip_serve").fetchone()[0]
    ndet = con.execute("SELECT COUNT(*) FROM queue WHERE segment_idx >= ?", (KEEP_SEG_BELOW,)).fetchone()[0]
    orig = con.execute("SELECT COUNT(*) FROM queue WHERE segment_idx < ?", (KEEP_SEG_BELOW,)).fetchone()[0]
    con.close()
    log('=' * 72)
    log(f'  COMMITTED. queue total={nq}  (original {orig} + new detection {ndet})   clip_serve={ncs}')
    log(f'  spans  before hash={h_before}')
    log(f'  spans  after  hash={h_after}  {"UNCHANGED OK" if h_before==h_after else "*** CHANGED ***"}')
    log(f'  spans rows {nspan_b}->{nspan_a} | verdicts {nverd_b}->{nverd_a} | qa_oracle {nqa_b}->{nqa_a} (all untouched)')


if __name__ == '__main__':
    main()
