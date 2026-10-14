"""
Feature extraction and normalization module for Hand Landmarks.
Converts raw 21 3D MediaPipe landmarks into translation-invariant,
scale-invariant geometric features for chord classification,
and extracts temporal movement dynamics for strumming recognition.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGERTIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCP_INDICES = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate 3D joint angle at vertex p2 between vectors (p1 - p2) and (p3 - p2).
    Returns angle in radians in range [0, pi].
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def normalize_hand_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Translates hand landmarks so the wrist is at (0, 0, 0),
    and scales by the palm distance (wrist to middle MCP).
    
    Args:
        landmarks: np.ndarray of shape (21, 3) or (63,)
        
    Returns:
        normalized_landmarks: np.ndarray of shape (21, 3)
        palm_scale: float scale factor used
    """
    lm = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = lm[WRIST].copy()
    
    # Translation invariance: subtract wrist position
    centered = lm - wrist
    
    # Scale invariance: distance from wrist to middle MCP knuckle
    palm_scale = float(np.linalg.norm(centered[MIDDLE_MCP]))
    if palm_scale < 1e-5:
        palm_scale = 1.0
        
    normalized = centered / palm_scale
    return normalized, palm_scale


def extract_chord_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extracts complete invariant feature vector for chord classification.
    
    Feature vector structure (total 98 features):
      - 63: Centered and scale-normalized 3D coordinates (21 * 3)
      - 20: Normalized Euclidean distances between anatomical keypoints:
            * 5 fingertip-to-wrist distances
            * 5 fingertip-to-corresponding-knuckle distances
            * 4 adjacent fingertip distances (thumb-index, index-mid, mid-ring, ring-pinky)
            * 5 fingertip-to-palm center distances
            * 1 thumb tip to pinky tip distance
      - 15: Joint flexion angles in radians:
            * Thumb: (1, 2, 3), (2, 3, 4)
            * Index: (0, 5, 6), (5, 6, 7), (6, 7, 8)
            * Middle: (0, 9, 10), (9, 10, 11), (10, 11, 12)
            * Ring: (0, 13, 14), (13, 14, 15), (14, 15, 16)
            * Pinky: (0, 17, 18), (17, 18, 19), (18, 19, 20)
            
    Args:
        landmarks: np.ndarray of shape (21, 3) or (63,)
        
    Returns:
        features: 1D np.ndarray of length 98, float32
    """
    norm_lm, _ = normalize_hand_landmarks(landmarks)
    
    # 1. 63 normalized coordinates
    coords_flat = norm_lm.flatten()
    
    # 2. Distance features
    wrist_pos = norm_lm[WRIST]
    palm_center = np.mean(norm_lm[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]], axis=0)
    
    distances = []
    # 5 fingertip to wrist
    for tip in FINGERTIP_INDICES:
        distances.append(np.linalg.norm(norm_lm[tip] - wrist_pos))
        
    # 5 fingertip to corresponding MCP knuckle
    for tip, mcp in zip(FINGERTIP_INDICES, MCP_INDICES):
        distances.append(np.linalg.norm(norm_lm[tip] - norm_lm[mcp]))
        
    # 4 adjacent fingertip distances
    distances.append(np.linalg.norm(norm_lm[THUMB_TIP] - norm_lm[INDEX_TIP]))
    distances.append(np.linalg.norm(norm_lm[INDEX_TIP] - norm_lm[MIDDLE_TIP]))
    distances.append(np.linalg.norm(norm_lm[MIDDLE_TIP] - norm_lm[RING_TIP]))
    distances.append(np.linalg.norm(norm_lm[RING_TIP] - norm_lm[PINKY_TIP]))
    
    # 5 fingertip to palm center
    for tip in FINGERTIP_INDICES:
        distances.append(np.linalg.norm(norm_lm[tip] - palm_center))
        
    # 1 span distance (thumb tip to pinky tip)
    distances.append(np.linalg.norm(norm_lm[THUMB_TIP] - norm_lm[PINKY_TIP]))
    
    # 3. Joint angles (15 angles)
    angles = [
        # Thumb
        calculate_angle(norm_lm[WRIST], norm_lm[THUMB_CMC], norm_lm[THUMB_MCP]),
        calculate_angle(norm_lm[THUMB_CMC], norm_lm[THUMB_MCP], norm_lm[THUMB_IP]),
        calculate_angle(norm_lm[THUMB_MCP], norm_lm[THUMB_IP], norm_lm[THUMB_TIP]),
        # Index
        calculate_angle(norm_lm[WRIST], norm_lm[INDEX_MCP], norm_lm[INDEX_PIP]),
        calculate_angle(norm_lm[INDEX_MCP], norm_lm[INDEX_PIP], norm_lm[INDEX_DIP]),
        calculate_angle(norm_lm[INDEX_PIP], norm_lm[INDEX_DIP], norm_lm[INDEX_TIP]),
        # Middle
        calculate_angle(norm_lm[WRIST], norm_lm[MIDDLE_MCP], norm_lm[MIDDLE_PIP]),
        calculate_angle(norm_lm[MIDDLE_MCP], norm_lm[MIDDLE_PIP], norm_lm[MIDDLE_DIP]),
        calculate_angle(norm_lm[MIDDLE_PIP], norm_lm[MIDDLE_DIP], norm_lm[MIDDLE_TIP]),
        # Ring
        calculate_angle(norm_lm[WRIST], norm_lm[RING_MCP], norm_lm[RING_PIP]),
        calculate_angle(norm_lm[RING_MCP], norm_lm[RING_PIP], norm_lm[RING_DIP]),
        calculate_angle(norm_lm[RING_PIP], norm_lm[RING_DIP], norm_lm[RING_TIP]),
        # Pinky
        calculate_angle(norm_lm[WRIST], norm_lm[PINKY_MCP], norm_lm[PINKY_PIP]),
        calculate_angle(norm_lm[PINKY_MCP], norm_lm[PINKY_PIP], norm_lm[PINKY_DIP]),
        calculate_angle(norm_lm[PINKY_PIP], norm_lm[PINKY_DIP], norm_lm[PINKY_TIP]),
    ]
    
    features = np.concatenate([
        coords_flat,
        np.array(distances, dtype=np.float32),
        np.array(angles, dtype=np.float32)
    ]).astype(np.float32)
    
    return features


def extract_strum_frame_features(
    curr_landmarks: np.ndarray,
    prev_landmarks: Optional[np.ndarray] = None,
    dt: float = 1.0 / 30.0
) -> np.ndarray:
    """
    Extracts movement and displacement features for a single strumming frame.
    
    Feature dimension: 18 features
      - 3: Wrist position (x, y, z)
      - 3: Wrist displacement (dx, dy, dz)
      - 3: Wrist velocity (vx, vy, vz)
      - 3: Middle MCP position (x, y, z)
      - 3: Middle MCP displacement (dx, dy, dz)
      - 2: Index MCP displacement (dx, dy)
      - 1: Vertical velocity indicator (dy / dt)
    """
    curr = np.asarray(curr_landmarks, dtype=np.float32).reshape(21, 3)
    curr_wrist = curr[WRIST]
    curr_mid_mcp = curr[MIDDLE_MCP]
    curr_idx_mcp = curr[INDEX_MCP]
    
    if prev_landmarks is not None:
        prev = np.asarray(prev_landmarks, dtype=np.float32).reshape(21, 3)
        prev_wrist = prev[WRIST]
        prev_mid_mcp = prev[MIDDLE_MCP]
        prev_idx_mcp = prev[INDEX_MCP]
        
        disp_wrist = curr_wrist - prev_wrist
        vel_wrist = disp_wrist / max(dt, 1e-4)
        disp_mid = curr_mid_mcp - prev_mid_mcp
        disp_idx = curr_idx_mcp - prev_idx_mcp
    else:
        disp_wrist = np.zeros(3, dtype=np.float32)
        vel_wrist = np.zeros(3, dtype=np.float32)
        disp_mid = np.zeros(3, dtype=np.float32)
        disp_idx = np.zeros(3, dtype=np.float32)
        
    vert_indicator = float(vel_wrist[1])  # Y velocity (down is positive in screen coords)
    
    features = np.concatenate([
        curr_wrist,          # 3
        disp_wrist,          # 3
        vel_wrist,           # 3
        curr_mid_mcp,        # 3
        disp_mid,            # 3
        disp_idx[:2],        # 2
        np.array([vert_indicator], dtype=np.float32)  # 1
    ]).astype(np.float32)
    
    return features


def get_chord_feature_names() -> List[str]:
    """Returns ordered list of 98 chord feature column names."""
    names = []
    # 63 normalized coords
    for i in range(21):
        names.extend([f"norm_x_{i}", f"norm_y_{i}", f"norm_z_{i}"])
        
    # 20 distances
    tip_names = ["thumb", "index", "middle", "ring", "pinky"]
    for t in tip_names:
        names.append(f"dist_{t}_tip_wrist")
    for t in tip_names:
        names.append(f"dist_{t}_tip_mcp")
    names.extend([
        "dist_thumb_index", "dist_index_middle",
        "dist_middle_ring", "dist_ring_pinky"
    ])
    for t in tip_names:
        names.append(f"dist_{t}_tip_palm_center")
    names.append("dist_thumb_pinky_span")
    
    # 15 angles
    names.extend([
        "angle_thumb_wrist_cmc_mcp", "angle_thumb_cmc_mcp_ip", "angle_thumb_mcp_ip_tip",
        "angle_index_wrist_mcp_pip", "angle_index_mcp_pip_dip", "angle_index_pip_dip_tip",
        "angle_middle_wrist_mcp_pip", "angle_middle_mcp_pip_dip", "angle_middle_pip_dip_tip",
        "angle_ring_wrist_mcp_pip", "angle_ring_mcp_pip_dip", "angle_ring_pip_dip_tip",
        "angle_pinky_wrist_mcp_pip", "angle_pinky_mcp_pip_dip", "angle_pinky_pip_dip_tip",
    ])
    return names
