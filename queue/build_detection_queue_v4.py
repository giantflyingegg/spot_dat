#!/usr/bin/env python3
"""
Stage 4c: rebuild the detection triage queue from the v4 DEPLOYMENT detections
(detections_v4_signhealth_ft.json), with diff-classification vs the banked v3
detections and per-detection model_tag.

Reuses build_detection_queue_flipaug helpers (lockout, resolve, spans_hash, iou).
Differences from the v3 builder:
  * source = detections_v4_signhealth_ft.json
  * timing_source = 'v4_signhealth_ft_detection'  (the current serve query does NOT
    recognise this -> the rebuilt queue is INERT until server.py is updated, which
    implements "HALT before serving resumes")
  * NO duration>=3.0 filter (that structurally excludes every sub-3s event — the
    opposite of the short-spell deployment goal; the user's 4c filter list omits it)
  * per-detection model_tag: IoU>=0.3 vs a v3 detection on the same video ->
    'v4_confirmed', else 'v4_new'.
Filters kept: lockout(test), drop done, drop rejected, drop IoU>0.3 span overlap.
Ranking: det_score (mean_prob) desc — unchanged. Guards the spans hash.
"""
import os, sys, json, glob, sqlite3
sys.path.insert(0, '/home/gfe/bsl_project/wife_drypass')
import build_detection_queue_flipaug as B

DET_V4 = '/home/gfe/bsl_project/youtube_mining/output_signhealth/*/detections_v4_signhealth_ft.json'
DET_V3_NAME = 'detections_v3_flipaug.json'
TIMING = 'v4_signhealth_ft_detection'
IOU_CONFIRM = 0.3
COMMIT = '--commit' in sys.argv


def load_v3_by_vid():
    out = {}
    for f in glob.glob('/home/gfe/bsl_project/youtube_mining/output_signhealth/*/' + DET_V3_NAME):
        d = json.load(open(f)); vid = d.get('video_id') or os.path.basename(os.path.dirname(f))
        out[vid] = [(float(e['start_time']), float(e['end_time'])) for e in d.get('detections', [])]
    return out


def main():
    keepset = B.load_keepset_paths()
    excluded = B.load_excluded_test_videos()
    v3 = load_v3_by_vid()

    # STEP 0: load v4 detections
    files = sorted(glob.glob(DET_V4))
    dets = []; unresolved = {}; raw = 0
    for f in files:
        d = json.load(open(f)); vid = d.get('video_id') or os.path.basename(os.path.dirname(f))
        path = B.resolve_path(vid, keepset)
        for e in d.get('detections', []):
            raw += 1
            ts, te = float(e['start_time']), float(e['end_time'])
            if path is None:
                unresolved[vid] = unresolved.get(vid, 0) + 1; continue
            dets.append({'video_id': vid, 't_start': ts, 't_end': te,
                         'duration': float(e.get('duration', te - ts)), 'score': float(e['mean_prob']), 'path': path})
    B.log('=' * 72)
    B.log(f'STEP 0  raw v4 detections         : {raw}  (from {len(files)} files)  survivors: {len(dets)} '
          f'(-{sum(unresolved.values())} unresolved)')

    before = len(dets)
    dets = [d for d in dets if d['video_id'] not in excluded]
    B.log(f'LOCKOUT drop frozen TEST videos   : -{before-len(dets):<5} survivors: {len(dets)} ({len(excluded)} locked)')

    con = sqlite3.connect(B.DB); con.row_factory = sqlite3.Row
    done = {r['video_id'] for r in con.execute("SELECT video_id FROM video_status WHERE status='done'")}
    rejected = {r['video_id'] for r in con.execute("SELECT video_id FROM video_status WHERE status='rejected'")}
    cs_vid = {r['segment_idx']: r['video_id'] for r in con.execute("SELECT segment_idx,video_id FROM clip_serve")}
    spans_by_vid = {}
    for r in con.execute("SELECT segment_idx,video_id,t_start_s,t_end_s FROM spans"):
        if r['t_start_s'] is None or r['t_end_s'] is None: continue
        v = r['video_id'] or cs_vid.get(r['segment_idx'])
        if v: spans_by_vid.setdefault(v, []).append((r['t_start_s'], r['t_end_s']))
    seg_hist = max(con.execute(f"SELECT IFNULL(MAX(segment_idx),0) FROM {t}").fetchone()[0]
                   for t in ('queue', 'clip_serve', 'verdicts', 'triage', 'spans'))
    ord_orig_max = con.execute(f"SELECT IFNULL(MAX(ord),-1) FROM queue WHERE segment_idx < {B.KEEP_SEG_BELOW}").fetchone()[0]
    con.close()

    before = len(dets); dets = [d for d in dets if d['video_id'] not in done]
    B.log(f'STEP 1  drop status=done          : -{before-len(dets):<5} survivors: {len(dets)} ({len(done)} done)')
    before = len(dets); dets = [d for d in dets if d['video_id'] not in rejected]
    B.log(f'STEP 2  drop status=rejected      : -{before-len(dets):<5} survivors: {len(dets)} ({len(rejected)} rejected)')
    before = len(dets)
    dets = [d for d in dets if not any(B.iou(d['t_start'], d['t_end'], s0, s1) > B.IOU_DROP
                                       for s0, s1 in spans_by_vid.get(d['video_id'], []))]
    B.log(f'STEP 3  drop IoU>{B.IOU_DROP} span overlap : -{before-len(dets):<5} survivors: {len(dets)}')
    B.log(f'(NO duration>=3.0 filter — deployment targets short spells; sub-3s events kept)')

    # diff-classify
    for d in dets:
        d['model_tag'] = 'v4_new'
        for v0, v1 in v3.get(d['video_id'], []):
            if B.iou(d['t_start'], d['t_end'], v0, v1) >= IOU_CONFIRM:
                d['model_tag'] = 'v4_confirmed'; break
    n_conf = sum(1 for d in dets if d['model_tag'] == 'v4_confirmed')
    n_new = len(dets) - n_conf
    B.log(f'DIFF    vs v3 detections          : v4_confirmed={n_conf}  v4_new={n_new}')

    # ── SERVE ORDER (encoded in ord): calibration block first, then 2:1 interleave ──
    import numpy as np
    SERVE_SEED = 42; CALIB_N = 150; RATIO_CONF, RATIO_NEW = 2, 1
    CALIB_OUT = '/home/gfe/bsl_project/wife_drypass/v4_calib_sample.json'
    rng = np.random.RandomState(SERVE_SEED)
    key = lambda d: (-d['score'], d['video_id'], d['t_start'])
    confirmed = sorted([d for d in dets if d['model_tag'] == 'v4_confirmed'], key=key)
    new_all = sorted([d for d in dets if d['model_tag'] == 'v4_new'], key=key)
    # calibration: uniform random sample of v4_new -> tag calib_v4new, served FIRST in
    # random order (so partial completion stays ~unbiased). These give the v4_new
    # precision estimate (keep-rate on this sample).
    n_calib = min(CALIB_N, len(new_all))
    calib_pos = set(rng.choice(len(new_all), size=n_calib, replace=False).tolist())
    calib = [new_all[i] for i in sorted(calib_pos)]
    for d in calib:
        d['model_tag'] = 'calib_v4new'
    rng.shuffle(calib)
    new_remain = [new_all[i] for i in range(len(new_all)) if i not in calib_pos]  # stays 'v4_new'
    # interleave the remainder: RATIO_CONF confirmed : RATIO_NEW new, each score-desc
    order = list(calib); ci = ni = 0
    while ci < len(confirmed) or ni < len(new_remain):
        for _ in range(RATIO_CONF):
            if ci < len(confirmed): order.append(confirmed[ci]); ci += 1
        for _ in range(RATIO_NEW):
            if ni < len(new_remain): order.append(new_remain[ni]); ni += 1
    seg_start = seg_hist + 1; ord_start = ord_orig_max + 1
    B.log(f'  queue rows to insert: {len(order)}  (calib block={len(calib)} first, then '
          f'{RATIO_CONF}:{RATIO_NEW} confirmed:new interleave; seg {seg_start}..{seg_start+len(order)-1})')
    nd = np.array([d['duration'] for d in new_all])
    B.log(f'  v4_new(+calib) duration: <0.5s={int((nd<0.5).sum())} 0.5-1.0s={int(((nd>=0.5)&(nd<1.0)).sum())} '
          f'1.0-2.0s={int(((nd>=1.0)&(nd<2.0)).sum())} >=2.0s={int((nd>=2.0).sum())} median={np.median(nd):.2f}s')

    if not COMMIT:
        B.log('  DRY RUN — no DB writes. Re-run with --commit.')
        return {'raw': raw, 'insert': len(order), 'confirmed': n_conf, 'new': n_new, 'calib': len(calib)}

    # write calibration-sample provenance BEFORE the DB write
    json.dump({'seed': SERVE_SEED, 'n_calib': len(calib), 'sampled_from_v4_new': len(new_all),
               'purpose': 'unbiased v4_new precision estimate (keep-rate on this sample)',
               'served_order_ord_start': ord_start,
               'clips': [{'video_id': d['video_id'], 't_start': round(d['t_start'], 3),
                          't_end': round(d['t_end'], 3), 'det_score': round(d['score'], 4)} for d in calib]},
              open(CALIB_OUT, 'w'), indent=2)
    B.log(f'  wrote calibration sample -> {CALIB_OUT}')
    survivors = order

    con = sqlite3.connect(B.DB)
    h_before = B.spans_hash(con)
    nq_before = con.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    ndet_before = con.execute(f"SELECT COUNT(*) FROM queue WHERE segment_idx >= {B.KEEP_SEG_BELOW}").fetchone()[0]
    try:
        con.execute("BEGIN")
        con.execute(f"DELETE FROM queue WHERE segment_idx >= {B.KEEP_SEG_BELOW}")
        con.execute(f"DELETE FROM clip_serve WHERE segment_idx >= {B.KEEP_SEG_BELOW}")
        seg = seg_start - 1; ordn = ord_start - 1
        for d in survivors:
            seg += 1; ordn += 1
            con.execute("""INSERT INTO queue (segment_idx,canonical,surface_form,match_tier,timing_source,
                video_id,signer,t_start,raw_event_dur_s,det_score,flag,reel_timestamp,clen,clip_id,
                human_confirmed_word,ord,model_tag) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (seg, None, None, None, TIMING, d['video_id'], d['video_id'], d['t_start'], d['duration'],
                 d['score'], 'detection', None, None, f'det_{d["video_id"]}_{d["t_start"]:.2f}', 0, ordn, d['model_tag']))
            con.execute("""INSERT INTO clip_serve (segment_idx,video_id,parent_video_path,orig_win_s,orig_win_e,
                served_s,served_e,model_tag) VALUES (?,?,?,?,?,?,?,?)""",
                (seg, d['video_id'], d['path'], d['t_start'], d['t_end'],
                 max(0.0, d['t_start'] - B.PAD), d['t_end'] + B.PAD, d['model_tag']))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK"); con.close(); raise
    h_after = B.spans_hash(con)
    nq_after = con.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    serveable_new = con.execute(f"SELECT COUNT(*) FROM queue WHERE model_tag='v4_new'").fetchone()[0]
    serveable_conf = con.execute(f"SELECT COUNT(*) FROM queue WHERE model_tag='v4_confirmed'").fetchone()[0]
    con.close()
    B.log('=' * 72)
    B.log(f'  COMMITTED (timing_source={TIMING}, INERT until serve query updated).')
    B.log(f'  queue rows {nq_before} -> {nq_after}  (cleared {ndet_before} old detection rows, inserted {len(survivors)})')
    B.log(f'  queue v4_confirmed={serveable_conf}  v4_new={serveable_new}')
    B.log(f'  spans hash {"UNCHANGED OK" if h_before==h_after else "*** CHANGED ***"}')


if __name__ == '__main__':
    main()
