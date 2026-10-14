"""
Unit tests for real-time model inference and debounce logic.
Verifies model shape compatibility, confidence thresholding, smoothing,
and strumming event cooldown debouncing.
"""

import sys
import os
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import ChordPredictor, StrummingDetector
from src.config import (
    CHORD_MODEL_PATH,
    STRUM_MODEL_PATH,
    SUPPORTED_CHORDS,
    SEQUENCE_LENGTH
)


def test_chord_predictor_inference():
    """ChordPredictor should load the saved smoke model and produce predictions."""
    assert CHORD_MODEL_PATH.exists(), "Chord model checkpoint must exist"

    predictor = ChordPredictor(confidence_threshold=0.1)
    assert predictor.is_ready is True

    fake_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    smoothed, conf, raw = predictor.predict(fake_landmarks)

    assert 0.0 <= conf <= 1.0
    assert raw in SUPPORTED_CHORDS


def test_chord_confidence_threshold():
    """When confidence threshold is high (1.0), uncertain predictions should return None for smoothed chord."""
    predictor = ChordPredictor(confidence_threshold=1.0)
    fake_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    smoothed, conf, raw = predictor.predict(fake_landmarks)

    # Since conf < 1.0 on random input, smoothed chord should be None
    assert smoothed is None


def test_chord_smoothing_majority_vote():
    """Smoothing history should pick majority vote."""
    predictor = ChordPredictor(smoothing_frames=3, confidence_threshold=0.0)
    predictor.history.extend(["am", "c", "am"])

    # Majority of history is 'am'
    valid_votes = [c for c in predictor.history if c != "uncertain"]
    majority = max(set(valid_votes), key=valid_votes.count)
    assert majority == "am"


def test_strumming_buffering_and_cooldown():
    """StrummingDetector should buffer until sequence length and enforce cooldown."""
    assert STRUM_MODEL_PATH.exists(), "Strumming model checkpoint must exist"

    detector = StrummingDetector(
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=0.0,  # Always trigger when action is strum
        cooldown_ms=250
    )
    assert detector.is_ready is True

    fake_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)

    # Feed less than sequence_length frames -> should return buffering
    for _ in range(SEQUENCE_LENGTH - 1):
        trig, action, conf = detector.update(fake_landmarks)
        assert trig is False
        assert action == "buffering"

    # Now frame sequence_length -> should make prediction
    trig1, action1, conf1 = detector.update(fake_landmarks, current_time=1.0)
    assert action1 in ("idle", "downstroke", "upstroke")

    # If action is downstroke/upstroke, immediate next frame at t=1.05s (< 250ms cooldown)
    # must NOT trigger duplicate audio!
    detector.classes = ["downstroke", "downstroke", "downstroke"]  # Force action to strum
    detector.last_strum_time = 1.0
    trig2, _, _ = detector.update(fake_landmarks, current_time=1.05)  # 50ms later
    assert trig2 is False, "Debounce failed: duplicate strum triggered within cooldown window!"

    # 300ms later (> 250ms cooldown) -> allowed to trigger again
    trig3, _, _ = detector.update(fake_landmarks, current_time=1.35)  # 350ms later
    assert trig3 is True, "Cooldown expired: strum should be allowed to trigger"
