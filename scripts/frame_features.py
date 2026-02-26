#!/usr/bin/env python3
"""
Per-frame feature extraction from MediaPipe landmarks (1692 dims).

Extracts rich features per frame for improved frame-level detection:
  Group A: Body kinematics (wrists, elbows relative to neck/shoulders) — 24 dims
  Group B: Hand shape (normalized positions, inter-finger distances, spread, etc.) — 108 dims
  Group C: Mouth features (opening, width, aspect ratio) — 6 dims
  Group D: Temporal derivatives (velocity, acceleration of wrists/fingertips) — 22 dims

Total: 160 dims per frame (vs 258 raw dims in baseline).
"""
import numpy as np
from typing import Optional

# ── MediaPipe landmark layout (1692 dims per frame) ──────────────────────
LANDMARK_DIM = 1692

# Pose: 33 landmarks × 4 (x, y, z, visibility) = 132 dims
POSE_START = 0
POSE_END = 132

# Face: 478 landmarks × 3 (x, y, z) = 1434 dims
FACE_START = 132
FACE_END = 1566

# Hands: 21 landmarks × 3 (x, y, z) = 63 dims each
HAND_L_START = 1566
HAND_L_END = 1629
HAND_R_START = 1629
HAND_R_END = 1692

# ── Pose landmark indices (×4 stride: x, y, z, vis) ─────────────────────
P_NOSE = 0
P_L_SHOULDER = 11
P_R_SHOULDER = 12
P_L_ELBOW = 13
P_R_ELBOW = 14
P_L_WRIST = 15
P_R_WRIST = 16

# ── Hand landmark indices (×3 stride: x, y, z) ──────────────────────────
H_WRIST = 0
H_THUMB_TIP = 4
H_INDEX_TIP = 8
H_MIDDLE_TIP = 12
H_RING_TIP = 16
H_PINKY_TIP = 20
H_INDEX_MCP = 5
H_MIDDLE_MCP = 9
H_RING_MCP = 13
H_PINKY_MCP = 17
H_THUMB_CMC = 1

# All 21 hand landmark indices
ALL_HAND_LANDMARKS = list(range(21))
FINGERTIP_INDICES = [H_THUMB_TIP, H_INDEX_TIP, H_MIDDLE_TIP, H_RING_TIP, H_PINKY_TIP]

# ── Face landmark indices (×3 stride: x, y, z) ──────────────────────────
F_UPPER_LIP_TOP = 13
F_LOWER_LIP_BOTTOM = 14
F_LEFT_LIP_CORNER = 61
F_RIGHT_LIP_CORNER = 291
F_UPPER_LIP_INNER = 0
F_LOWER_LIP_INNER = 17


def _pose_xyz(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """Get (N, 3) xyz for a pose landmark. landmarks shape: (N, 1692)."""
    base = POSE_START + idx * 4
    return landmarks[:, base:base + 3]


def _hand_xyz(landmarks: np.ndarray, hand_start: int, idx: int) -> np.ndarray:
    """Get (N, 3) xyz for a hand landmark."""
    base = hand_start + idx * 3
    return landmarks[:, base:base + 3]


def _hand_all_xyz(landmarks: np.ndarray, hand_start: int) -> np.ndarray:
    """Get (N, 21, 3) for all hand landmarks."""
    raw = landmarks[:, hand_start:hand_start + 63]  # (N, 63)
    return raw.reshape(-1, 21, 3)


def _face_xyz(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """Get (N, 3) xyz for a face landmark."""
    base = FACE_START + idx * 3
    return landmarks[:, base:base + 3]


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between (N, 3) arrays -> (N,)."""
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


# ══════════════════════════════════════════════════════════════════════════
# GROUP A: Body kinematics (24 dims)
# ══════════════════════════════════════════════════════════════════════════

def extract_group_a(landmarks: np.ndarray) -> np.ndarray:
    """
    Body kinematics relative to neck/shoulders.
    24 dims per frame:
      - L/R wrist relative to neck (nose): 6
      - L/R wrist relative to shoulder centre: 6
      - L/R elbow relative to shoulder centre: 6
      - L/R arm extension (wrist-shoulder distance): 2
      - L/R wrist height above shoulder: 2
      - Inter-wrist distance: 1
      - Shoulder width (normalisation reference): 1
    """
    N = len(landmarks)

    nose = _pose_xyz(landmarks, P_NOSE)  # (N,3)
    l_shoulder = _pose_xyz(landmarks, P_L_SHOULDER)
    r_shoulder = _pose_xyz(landmarks, P_R_SHOULDER)
    l_elbow = _pose_xyz(landmarks, P_L_ELBOW)
    r_elbow = _pose_xyz(landmarks, P_R_ELBOW)
    l_wrist = _pose_xyz(landmarks, P_L_WRIST)
    r_wrist = _pose_xyz(landmarks, P_R_WRIST)

    shoulder_c = (l_shoulder + r_shoulder) / 2  # (N,3)
    shoulder_w = _dist(l_shoulder, r_shoulder)[:, None]  # (N,1) for normalisation
    shoulder_w = np.maximum(shoulder_w, 1e-6)

    feats = np.concatenate([
        (l_wrist - nose) / shoulder_w,          # 3
        (r_wrist - nose) / shoulder_w,          # 3
        (l_wrist - shoulder_c) / shoulder_w,    # 3
        (r_wrist - shoulder_c) / shoulder_w,    # 3
        (l_elbow - shoulder_c) / shoulder_w,    # 3
        (r_elbow - shoulder_c) / shoulder_w,    # 3
        (_dist(l_wrist, l_shoulder) / shoulder_w.squeeze(-1))[:, None],  # 1
        (_dist(r_wrist, r_shoulder) / shoulder_w.squeeze(-1))[:, None],  # 1
        ((shoulder_c[:, 1:2] - l_wrist[:, 1:2]) / shoulder_w),  # 1 (positive = above)
        ((shoulder_c[:, 1:2] - r_wrist[:, 1:2]) / shoulder_w),  # 1
        (_dist(l_wrist, r_wrist) / shoulder_w.squeeze(-1))[:, None],    # 1
        shoulder_w,                              # 1 (scale reference)
    ], axis=1)

    assert feats.shape == (N, 24), f"Group A: expected 24, got {feats.shape[1]}"
    return feats


# ══════════════════════════════════════════════════════════════════════════
# GROUP B: Hand shape features (108 dims)
# ══════════════════════════════════════════════════════════════════════════

def _hand_shape_features(landmarks: np.ndarray, hand_start: int) -> np.ndarray:
    """
    Per-hand shape features (54 dims per hand):
      - 21 landmark positions normalised relative to wrist: 21*3 = 63 -> PCA'd to 30
        (Actually we'll keep key landmarks + derived features instead of raw 63)

      Design (54 dims per hand):
        - 5 fingertip positions relative to wrist: 15
        - 5 MCP positions relative to wrist: 15 (actually 4 MCP + thumb CMC = 5)
        - Inter-finger distances (thumb-index, index-middle, middle-ring, ring-pinky): 4
        - Fingertip spread (max pairwise distance among tips): 1
        - Hand openness (mean fingertip-to-palm distance): 1
        - Hand compactness (variance of all landmark positions): 3 (x,y,z variances)
        - Finger extension ratios (tip-MCP distance / MCP-wrist distance): 5
        - Thumb-to-each-finger distances: 4
        - Palm size (mean MCP-wrist distance): 1
        - Wrist-relative hand depth range: 1
        Total: 15 + 15 + 4 + 1 + 1 + 3 + 5 + 4 + 1 + 1 = 50

      Adjusted to 54: add 4 more (finger curl angles approximated as distance ratios)
        - DIP-to-MCP / TIP-to-MCP for 4 fingers: 4
        Total: 54
    """
    N = len(landmarks)
    all_xyz = _hand_all_xyz(landmarks, hand_start)  # (N, 21, 3)
    wrist_xyz = all_xyz[:, H_WRIST, :]  # (N, 3)

    # Shoulder width for normalisation
    l_shoulder = _pose_xyz(landmarks, P_L_SHOULDER)
    r_shoulder = _pose_xyz(landmarks, P_R_SHOULDER)
    sw = np.maximum(_dist(l_shoulder, r_shoulder), 1e-6)[:, None]  # (N,1)

    # Normalise all hand landmarks relative to wrist and shoulder width
    rel = (all_xyz - wrist_xyz[:, None, :])  # (N, 21, 3)
    rel_norm = rel / sw[:, None, :]  # normalise by shoulder width

    # ── Fingertip positions relative to wrist (5 × 3 = 15) ──
    tips_rel = rel_norm[:, FINGERTIP_INDICES, :].reshape(N, -1)  # (N, 15)

    # ── MCP/CMC positions relative to wrist (5 × 3 = 15) ──
    mcp_indices = [H_THUMB_CMC, H_INDEX_MCP, H_MIDDLE_MCP, H_RING_MCP, H_PINKY_MCP]
    mcp_rel = rel_norm[:, mcp_indices, :].reshape(N, -1)  # (N, 15)

    # ── Inter-finger distances (4) ──
    pairs = [(H_THUMB_TIP, H_INDEX_TIP),
             (H_INDEX_TIP, H_MIDDLE_TIP),
             (H_MIDDLE_TIP, H_RING_TIP),
             (H_RING_TIP, H_PINKY_TIP)]
    inter_dists = np.stack([
        _dist(all_xyz[:, a], all_xyz[:, b]) / sw.squeeze(-1)
        for a, b in pairs
    ], axis=1)  # (N, 4)

    # ── Fingertip spread: max pairwise distance among 5 tips (1) ──
    tip_xyz = all_xyz[:, FINGERTIP_INDICES, :]  # (N, 5, 3)
    max_spread = np.zeros(N)
    for i in range(5):
        for j in range(i + 1, 5):
            d = _dist(tip_xyz[:, i], tip_xyz[:, j])
            max_spread = np.maximum(max_spread, d)
    spread = (max_spread / sw.squeeze(-1))[:, None]  # (N, 1)

    # ── Hand openness: mean fingertip-to-palm distance (1) ──
    palm_center = np.mean(all_xyz[:, mcp_indices, :], axis=1)  # (N, 3)
    tip_to_palm = np.mean([
        _dist(all_xyz[:, t], palm_center) for t in FINGERTIP_INDICES
    ], axis=0)
    openness = (tip_to_palm / sw.squeeze(-1))[:, None]  # (N, 1)

    # ── Hand compactness: variance of all landmark positions (3) ──
    compactness = np.var(rel_norm, axis=1)  # (N, 3)

    # ── Finger extension ratios: tip-MCP dist / MCP-wrist dist (5) ──
    ext_pairs = [
        (H_THUMB_TIP, H_THUMB_CMC),
        (H_INDEX_TIP, H_INDEX_MCP),
        (H_MIDDLE_TIP, H_MIDDLE_MCP),
        (H_RING_TIP, H_RING_MCP),
        (H_PINKY_TIP, H_PINKY_MCP),
    ]
    extensions = []
    for tip, mcp in ext_pairs:
        tip_mcp = _dist(all_xyz[:, tip], all_xyz[:, mcp])
        mcp_wrist = _dist(all_xyz[:, mcp], wrist_xyz)
        ratio = tip_mcp / np.maximum(mcp_wrist, 1e-6)
        extensions.append(ratio)
    extensions = np.stack(extensions, axis=1)  # (N, 5)

    # ── Thumb-to-each-finger distances (4) ──
    thumb = all_xyz[:, H_THUMB_TIP]
    thumb_dists = np.stack([
        _dist(thumb, all_xyz[:, t]) / sw.squeeze(-1)
        for t in [H_INDEX_TIP, H_MIDDLE_TIP, H_RING_TIP, H_PINKY_TIP]
    ], axis=1)  # (N, 4)

    # ── Palm size (1) ──
    palm_size = np.mean([
        _dist(all_xyz[:, m], wrist_xyz) for m in mcp_indices
    ], axis=0)
    palm_size = (palm_size / sw.squeeze(-1))[:, None]  # (N, 1)

    # ── Hand depth range (1) ──
    z_range = (np.max(rel_norm[:, :, 2], axis=1) -
               np.min(rel_norm[:, :, 2], axis=1))[:, None]  # (N, 1)

    # ── Finger curl (DIP-MCP / TIP-MCP for 4 non-thumb fingers) (4) ──
    dip_indices = [7, 11, 15, 19]  # index_DIP, middle_DIP, ring_DIP, pinky_DIP
    mcp_list = [H_INDEX_MCP, H_MIDDLE_MCP, H_RING_MCP, H_PINKY_MCP]
    tip_list = [H_INDEX_TIP, H_MIDDLE_TIP, H_RING_TIP, H_PINKY_TIP]
    curls = []
    for dip, mcp, tip in zip(dip_indices, mcp_list, tip_list):
        dip_mcp = _dist(all_xyz[:, dip], all_xyz[:, mcp])
        tip_mcp = _dist(all_xyz[:, tip], all_xyz[:, mcp])
        curl = dip_mcp / np.maximum(tip_mcp, 1e-6)
        curls.append(curl)
    curls = np.stack(curls, axis=1)  # (N, 4)

    feats = np.concatenate([
        tips_rel,      # 15
        mcp_rel,       # 15
        inter_dists,   # 4
        spread,        # 1
        openness,      # 1
        compactness,   # 3
        extensions,    # 5
        thumb_dists,   # 4
        palm_size,     # 1
        z_range,       # 1
        curls,         # 4
    ], axis=1)

    assert feats.shape == (N, 54), f"Hand shape: expected 54, got {feats.shape[1]}"
    return feats


def extract_group_b(landmarks: np.ndarray) -> np.ndarray:
    """Hand shape features for both hands: 54 × 2 = 108 dims."""
    left = _hand_shape_features(landmarks, HAND_L_START)
    right = _hand_shape_features(landmarks, HAND_R_START)
    return np.concatenate([left, right], axis=1)


# ══════════════════════════════════════════════════════════════════════════
# GROUP C: Mouth features (6 dims)
# ══════════════════════════════════════════════════════════════════════════

def extract_group_c(landmarks: np.ndarray) -> np.ndarray:
    """
    Mouth features (6 dims per frame):
      - Mouth opening (upper-lower lip distance): 1
      - Mouth width (corner-corner distance): 1
      - Mouth aspect ratio (opening / width): 1
      - Inner lip opening: 1
      - Mouth opening normalised by shoulder width: 1
      - Mouth width normalised by shoulder width: 1
    """
    N = len(landmarks)

    upper = _face_xyz(landmarks, F_UPPER_LIP_TOP)
    lower = _face_xyz(landmarks, F_LOWER_LIP_BOTTOM)
    left_corner = _face_xyz(landmarks, F_LEFT_LIP_CORNER)
    right_corner = _face_xyz(landmarks, F_RIGHT_LIP_CORNER)
    upper_inner = _face_xyz(landmarks, F_UPPER_LIP_INNER)
    lower_inner = _face_xyz(landmarks, F_LOWER_LIP_INNER)

    l_shoulder = _pose_xyz(landmarks, P_L_SHOULDER)
    r_shoulder = _pose_xyz(landmarks, P_R_SHOULDER)
    sw = np.maximum(_dist(l_shoulder, r_shoulder), 1e-6)

    opening = _dist(upper, lower)
    width = _dist(left_corner, right_corner)
    aspect = opening / np.maximum(width, 1e-6)
    inner_opening = _dist(upper_inner, lower_inner)

    feats = np.column_stack([
        opening,                  # 1
        width,                    # 1
        aspect,                   # 1
        inner_opening,            # 1
        opening / sw,             # 1 normalised
        width / sw,               # 1 normalised
    ])

    assert feats.shape == (N, 6), f"Group C: expected 6, got {feats.shape[1]}"
    return feats


# ══════════════════════════════════════════════════════════════════════════
# GROUP D: Temporal derivatives (22 dims)
# ══════════════════════════════════════════════════════════════════════════

def extract_group_d(landmarks: np.ndarray) -> np.ndarray:
    """
    Temporal derivatives (22 dims per frame):
      - L/R wrist velocity (xyz): 6
      - L/R wrist acceleration (xyz): 6
      - 5 fingertip velocities per hand (magnitude only): 10 (5 left + 5 right, but too many)

    Actually, to keep it tight:
      - L/R wrist velocity (xyz): 6
      - L/R wrist speed (magnitude): 2
      - L/R wrist acceleration (magnitude): 2
      - L/R mean fingertip velocity (magnitude): 2
      - L/R max fingertip velocity (magnitude): 2
      - L/R fingertip velocity variance: 2
      - Mean inter-finger velocity difference (L/R): 2
      - Dominant hand relative speed: 1
      - Wrist jerk (magnitude): 1
      Total: 22
    """
    N = len(landmarks)

    # Wrist positions
    l_wrist = _pose_xyz(landmarks, P_L_WRIST)  # (N, 3)
    r_wrist = _pose_xyz(landmarks, P_R_WRIST)

    # Velocity (first derivative, padded with 0 at start)
    l_vel = np.diff(l_wrist, axis=0, prepend=l_wrist[:1])  # (N, 3)
    r_vel = np.diff(r_wrist, axis=0, prepend=r_wrist[:1])

    l_speed = np.sqrt(np.sum(l_vel ** 2, axis=-1))  # (N,)
    r_speed = np.sqrt(np.sum(r_vel ** 2, axis=-1))

    # Acceleration (second derivative)
    l_acc = np.diff(l_vel, axis=0, prepend=l_vel[:1])
    r_acc = np.diff(r_vel, axis=0, prepend=r_vel[:1])
    l_acc_mag = np.sqrt(np.sum(l_acc ** 2, axis=-1))
    r_acc_mag = np.sqrt(np.sum(r_acc ** 2, axis=-1))

    # Jerk (third derivative) - just magnitude of right wrist
    r_jerk = np.diff(r_acc, axis=0, prepend=r_acc[:1])
    r_jerk_mag = np.sqrt(np.sum(r_jerk ** 2, axis=-1))

    # Fingertip velocities (per hand)
    def fingertip_vel_stats(hand_start):
        speeds = []
        for tip in FINGERTIP_INDICES:
            pos = _hand_xyz(landmarks, hand_start, tip)  # (N, 3)
            vel = np.diff(pos, axis=0, prepend=pos[:1])
            speed = np.sqrt(np.sum(vel ** 2, axis=-1))  # (N,)
            speeds.append(speed)
        speeds = np.stack(speeds, axis=1)  # (N, 5)
        mean_speed = np.mean(speeds, axis=1)  # (N,)
        max_speed = np.max(speeds, axis=1)   # (N,)
        var_speed = np.var(speeds, axis=1)   # (N,)
        # Mean difference between adjacent finger speeds
        diffs = np.abs(np.diff(speeds, axis=1))  # (N, 4)
        mean_diff = np.mean(diffs, axis=1)  # (N,)
        return mean_speed, max_speed, var_speed, mean_diff

    l_ft_mean, l_ft_max, l_ft_var, l_ft_diff = fingertip_vel_stats(HAND_L_START)
    r_ft_mean, r_ft_max, r_ft_var, r_ft_diff = fingertip_vel_stats(HAND_R_START)

    # Dominant hand relative speed
    total_speed = l_speed + r_speed + 1e-6
    dom_speed = (r_speed - l_speed) / total_speed  # positive = right dominant

    feats = np.column_stack([
        l_vel,          # 3
        r_vel,          # 3
        l_speed,        # 1
        r_speed,        # 1
        l_acc_mag,      # 1
        r_acc_mag,      # 1
        l_ft_mean,      # 1
        r_ft_mean,      # 1
        l_ft_max,       # 1
        r_ft_max,       # 1
        l_ft_var,       # 1
        r_ft_var,       # 1
        l_ft_diff,      # 1
        r_ft_diff,      # 1
        dom_speed,      # 1
        r_jerk_mag,     # 1
    ])  # Total: 3+3+1+1+1+1+1+1+1+1+1+1+1+1+1+1 = 22 -- wait, let me count
    # l_vel(3) + r_vel(3) + l_speed(1) + r_speed(1) + l_acc_mag(1) + r_acc_mag(1)
    # + l_ft_mean(1) + r_ft_mean(1) + l_ft_max(1) + r_ft_max(1)
    # + l_ft_var(1) + r_ft_var(1) + l_ft_diff(1) + r_ft_diff(1)
    # + dom_speed(1) + r_jerk_mag(1) = 22

    assert feats.shape == (N, 20), f"Group D: expected 20, got {feats.shape[1]}"
    return feats


# ══════════════════════════════════════════════════════════════════════════
# COMBINED FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

# Group dimensions
GROUP_A_DIM = 24
GROUP_B_DIM = 108
GROUP_C_DIM = 6
GROUP_D_DIM = 20
TOTAL_FEATURE_DIM = GROUP_A_DIM + GROUP_B_DIM + GROUP_C_DIM + GROUP_D_DIM  # 158

# Group offsets (for ablation)
GROUP_OFFSETS = {
    'A': (0, GROUP_A_DIM),
    'B': (GROUP_A_DIM, GROUP_A_DIM + GROUP_B_DIM),
    'C': (GROUP_A_DIM + GROUP_B_DIM, GROUP_A_DIM + GROUP_B_DIM + GROUP_C_DIM),
    'D': (GROUP_A_DIM + GROUP_B_DIM + GROUP_C_DIM, TOTAL_FEATURE_DIM),
}


def extract_all_frame_features(landmarks: np.ndarray,
                                groups: str = 'ABCD') -> np.ndarray:
    """
    Extract per-frame features from raw MediaPipe landmarks.

    Args:
        landmarks: (N, 1692) array of MediaPipe holistic landmarks
        groups: which feature groups to include, e.g. 'AB', 'ABCD'

    Returns:
        (N, D) feature array where D depends on groups selected
    """
    landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

    parts = []
    if 'A' in groups:
        parts.append(extract_group_a(landmarks))
    if 'B' in groups:
        parts.append(extract_group_b(landmarks))
    if 'C' in groups:
        parts.append(extract_group_c(landmarks))
    if 'D' in groups:
        parts.append(extract_group_d(landmarks))

    feats = np.concatenate(parts, axis=1)
    # Final NaN cleanup
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def get_feature_dim(groups: str = 'ABCD') -> int:
    """Return feature dimension for given group combination."""
    dims = {'A': GROUP_A_DIM, 'B': GROUP_B_DIM, 'C': GROUP_C_DIM, 'D': GROUP_D_DIM}
    return sum(dims[g] for g in groups)


# ── Also provide the baseline 258-dim feature extractor for comparison ───

def extract_baseline_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Baseline: raw pose (132) + left hand (63) + right hand (63) = 258 dims.
    Same as the existing frame detector.
    """
    landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
    pose = landmarks[:, :POSE_END]
    lh = landmarks[:, HAND_L_START:HAND_L_END]
    rh = landmarks[:, HAND_R_START:HAND_R_END]
    return np.concatenate([pose, lh, rh], axis=1)


BASELINE_FEATURE_DIM = 258


if __name__ == '__main__':
    # Test on dummy data
    print("Testing per-frame feature extraction...")
    dummy = np.random.randn(50, LANDMARK_DIM).astype(np.float32) * 0.1 + 0.5

    for groups in ['A', 'AB', 'ABC', 'ABCD']:
        feats = extract_all_frame_features(dummy, groups=groups)
        expected_dim = get_feature_dim(groups)
        print(f"  Groups '{groups}': shape={feats.shape}, expected dim={expected_dim}")
        assert feats.shape == (50, expected_dim)

    baseline = extract_baseline_features(dummy)
    print(f"  Baseline: shape={baseline.shape}")
    assert baseline.shape == (50, 258)

    print(f"\nFeature dimension summary:")
    print(f"  Group A (body kinematics): {GROUP_A_DIM}")
    print(f"  Group B (hand shape):      {GROUP_B_DIM}")
    print(f"  Group C (mouth):           {GROUP_C_DIM}")
    print(f"  Group D (temporal):        {GROUP_D_DIM}")
    print(f"  Total:                     {TOTAL_FEATURE_DIM}")
    print(f"  Baseline (raw landmarks):  {BASELINE_FEATURE_DIM}")
    print("\nAll tests passed!")
