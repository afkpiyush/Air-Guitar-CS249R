"""
Unit tests for data preprocessing, schema validation, and session splitting.
Verifies no data leakage between train/test sessions and handles invalid data.
"""

import sys
import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    SMOKE_TEST_DATA_DIR,
    SUPPORTED_CHORDS,
    STRUM_CLASSES,
    SEQUENCE_LENGTH,
    STRUM_FEATURE_DIM
)
from src.feature_extraction import extract_chord_features


def test_smoke_chord_dataset_schema():
    """Validates the synthetic chord features schema and values."""
    csv_path = SMOKE_TEST_DATA_DIR / "smoke_chord_features.csv"
    assert csv_path.exists(), "smoke_chord_features.csv fixture not found"

    df = pd.read_csv(csv_path)
    assert "session_id" in df.columns
    assert "label" in df.columns
    assert "label_encoded" in df.columns

    # Verify all 13 chords are present
    unique_labels = set(df["label"].unique())
    for chord in SUPPORTED_CHORDS:
        assert chord in unique_labels

    # Verify feature columns
    feat_cols = [c for c in df.columns if c not in ("session_id", "label", "label_encoded")]
    assert len(feat_cols) == 98


def test_smoke_strum_dataset_schema():
    """Validates the synthetic strumming sequence NPZ schema."""
    npz_path = SMOKE_TEST_DATA_DIR / "smoke_strumming_sequences.npz"
    assert npz_path.exists(), "smoke_strumming_sequences.npz fixture not found"

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    session_ids = data["session_ids"]

    assert len(X.shape) == 3
    assert X.shape[1] == SEQUENCE_LENGTH
    assert X.shape[2] == STRUM_FEATURE_DIM
    assert len(y) == len(X)
    assert len(session_ids) == len(X)

    unique_classes = set(y.tolist())
    assert unique_classes.issubset({0, 1, 2})


def test_session_split_no_leakage():
    """Ensures GroupShuffleSplit produces disjoint sessions with zero data leakage."""
    csv_path = SMOKE_TEST_DATA_DIR / "smoke_chord_features.csv"
    df = pd.read_csv(csv_path)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["session_id"]))

    train_sessions = set(df.iloc[train_idx]["session_id"].unique())
    test_sessions = set(df.iloc[test_idx]["session_id"].unique())

    # Critical check: zero session overlap
    overlap = train_sessions.intersection(test_sessions)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping sessions: {overlap}"


def test_invalid_input_landmarks_handling():
    """All zeros or missing landmarks should not raise unhandled exceptions."""
    zero_lms = np.zeros((21, 3), dtype=np.float32)
    features = extract_chord_features(zero_lms)
    assert features.shape == (98,)
    assert not np.isnan(features).any()
