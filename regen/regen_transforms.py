"""
Cache-reuse transform chain for SignHealth detection regeneration.

Reproduces, bit-exact, the 2026-07-13 domain-adaptation correction chain whose
generator script was lost (only its outputs for 53 videos survive). Reverse-
engineered and validated against those known-good artifacts.

Chain per video:  raw cached landmarks
   -> resample to 25fps  (linear interp on real-time axis)
   -> isotropy correction (x,z *= W/H, y unchanged; W/H = ffprobe width/height)
   -> per-frame geometric hand de-swap (blocks 1566/1629 by pose-wrist nearest)

1692-dim layout: pose 33*(x,y,z,vis)=132, face 478*(x,y,z)=1434, handL 1566:1629,
handR 1629:1692 (each 21*(x,y,z)).
"""
import numpy as np

POSE = 132
LA, LB = 1566, 1629          # hand-block A (slot "left"), hand-block B (slot "right")

# x / y / z column index sets across the whole 1692 vector
_XC = np.array(list(range(0, POSE, 4)) + list(range(POSE, 1566, 3)) + list(range(1566, 1692, 3)))
_YC = np.array(list(range(1, POSE, 4)) + list(range(POSE + 1, 1566, 3)) + list(range(1567, 1692, 3)))
_ZC = np.array(list(range(2, POSE, 4)) + list(range(POSE + 2, 1566, 3)) + list(range(1568, 1692, 3)))


def resample_25fps(lm, fps):
    """Linear-resample landmarks to 25fps on the real-time axis.
    n25 = round(n * 25 / fps); already-25fps is an exact no-op."""
    n = len(lm)
    if abs(fps - 25.0) < 1e-9:
        return lm.astype(np.float32, copy=True)
    n25 = int(round(n * 25.0 / fps))
    xo = np.arange(n) / fps
    xn = np.arange(n25) / 25.0
    out = np.empty((n25, lm.shape[1]), dtype=np.float32)
    for c in range(lm.shape[1]):
        out[:, c] = np.interp(xn, xo, lm[:, c])
    return out


def iso_correct(lm, wh):
    """Multiply x and z columns by W/H (float32); y and visibility unchanged."""
    out = lm.copy()
    whf = np.float32(wh)
    out[:, _XC] *= whf
    out[:, _ZC] *= whf
    return out


def _deswap_decision(I):
    """Per-frame: True => swap hand blocks A<->B. Faithful to the geometry-based
    assignment patched into run_signhealth_pipeline.py:_tasks_to_array (wrist lm0,
    2D distance to pose-15/16 on the *iso-corrected* coords)."""
    N = len(I)
    ax, ay = I[:, LA], I[:, LA + 1]
    bx, by = I[:, LB], I[:, LB + 1]
    Ap = ~((ax == 0) & (ay == 0))          # presence keyed on wrist landmark 0
    Bp = ~((bx == 0) & (by == 0))
    p15x, p15y = I[:, 60], I[:, 61]
    p16x, p16y = I[:, 64], I[:, 65]
    p15ok = ~((p15x == 0) & (p15y == 0))
    p16ok = ~((p16x == 0) & (p16y == 0))
    INF = np.inf
    dA15 = np.where(p15ok, np.hypot(ax - p15x, ay - p15y), INF)
    dA16 = np.where(p16ok, np.hypot(ax - p16x, ay - p16y), INF)
    dB15 = np.where(p15ok, np.hypot(bx - p15x, by - p15y), INF)
    dB16 = np.where(p16ok, np.hypot(bx - p16x, by - p16y), INF)

    swap = np.zeros(N, dtype=bool)
    both = Ap & Bp & p15ok & p16ok
    swap[both] = (dA15[both] + dB16[both]) > (dB15[both] + dA16[both])
    a_only = Ap & ~Bp
    swap[a_only] = dA16[a_only] < dA15[a_only]        # A moves to slot 1629
    b_only = Bp & ~Ap
    swap[b_only] = dB15[b_only] <= dB16[b_only]       # B moves to slot 1566
    # both hands present but a pose wrist missing: per-hand assign (A then B), collide->other
    edge = Ap & Bp & ~(p15ok & p16ok)
    for i in np.where(edge)[0]:
        assign = {}
        for name, d15, d16 in (('A', dA15[i], dA16[i]), ('B', dB15[i], dB16[i])):
            start = 1566 if d15 <= d16 else 1629
            if start in assign:
                start = 1629 if start == 1566 else 1566
            assign[start] = name
        swap[i] = (assign.get(1566) == 'B') or (assign.get(1629) == 'A')
    # geometry degenerate (both pose wrists undetected): fall back to the
    # systematic block swap that de-swap exists to apply.
    bothmiss = (~p15ok) & (~p16ok)
    swap[bothmiss] = True
    return swap


def deswap(I):
    """Apply the per-frame geometric hand-block de-swap."""
    swap = _deswap_decision(I)
    out = I.copy()
    A = I[:, LA:LA + 63]
    B = I[:, LB:LB + 63]
    out[swap, LA:LA + 63] = B[swap]
    out[swap, LB:LB + 63] = A[swap]
    return out


def transform_chain(raw_lm, fps, wh):
    """raw landmarks -> 25fps -> iso -> deswap."""
    return deswap(iso_correct(resample_25fps(raw_lm, fps), wh))
