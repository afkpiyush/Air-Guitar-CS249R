"""
Unit tests for landmark normalization and feature extraction.
Verifies dimensions, scale invariance, angle computations, and displacement features.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extraction import (
    normalize_hand_landmarks,
    extract_chord_features,
    extract_strum_frame_features,
    calculate_angle,
    get_chord_feature_names,
    WRIST,
    MIDDLE_MCP
)


def test_normalize_hand_landmarks():
    """Wrist should be translated to (0, 0, 0) and middle MCP distance scaled."""
    raw_lms = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    norm_lms, scale = normalize_hand_landmarks(raw_lms)

    assert norm_lms.shape == (21, 3)
    assert scale > 0.0
    # Wrist should be at origin
    np.testing.assert_allclose(norm_lms[WRIST], [0.0, 0.0, 0.0], atol=1e-6)
    # Distance to middle MCP should be 1.0 (by definition of scale)
    mcp_dist = np.linalg.norm(norm_lms[MIDDLE_MCP])
    np.testing.assert_allclose(mcp_dist, 1.0, atol=1e-5)


def test_scale_invariance():
    """Scaling raw landmarks by 2x should yield identical normalized landmarks."""
    raw_lms = np.random.uniform(0.2, 0.8, (21, 3)).astype(np.float32)
    wrist = raw_lms[WRIST].copy()
    centered = raw_lms - wrist

    scaled_lms = wrist + (centered * 2.5)

    norm1, _ = normalize_hand_landmarks(raw_lms)
    norm2, _ = normalize_hand_landmarks(scaled_lms)

    np.testing.assert_allclose(norm1, norm2, atol=1e-5)


def test_extract_chord_features_dimension():
    """Chord feature vector must have exactly 98 features and no NaNs."""
    raw_lms = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    features = extract_chord_features(raw_lms)

    assert features.shape == (98,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    names = get_chord_feature_names()
    assert len(names) == 98


def test_calculate_angle():
    """Tests joint angle computation for known right angles and collinear points."""
    # Right angle (90 deg = pi/2)
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    angle = calculate_angle(p1, p2, p3)
    np.testing.assert_allclose(angle, np.pi / 2.0, atol=1e-5)

    # Collinear opposite (180 deg = pi)
    p4 = np.array([-1.0, 0.0, 0.0])
    angle_180 = calculate_angle(p1, p2, p4)
    np.testing.assert_allclose(angle_180, np.pi, atol=1e-5)

    # Degenerate zero distance
    angle_zero = calculate_angle(p2, p2, p3)
    assert angle_zero == 0.0


def test_strum_frame_features_dimension():
    """Strumming frame feature vector must have 18 features."""
    curr_lms = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    prev_lms = curr_lms + np.array([0.0, 0.05, 0.0], dtype=np.float32)

    feat_single = extract_strum_frame_features(curr_lms)
    assert feat_single.shape == (18,)

    feat_temporal = extract_strum_frame_features(curr_lms, prev_lms, dt=1.0/30.0)
    assert feat_temporal.shape == (18,)
    assert not np.isnan(feat_temporal).any()
