"""
Unit tests for AudioEngine.
Verifies audio asset verification, volume control, muting, and headless execution.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_engine import AudioEngine
from src.config import SUPPORTED_CHORDS


def test_audio_engine_asset_verification():
    """All 13 supported guitar chord files must exist in the audio directory."""
    engine = AudioEngine()
    report = engine.verify_assets()

    assert report["total_supported"] == 13
    assert report["found_count"] == 13
    assert report["missing_count"] == 0
    assert report["all_present"] is True

    for chord in SUPPORTED_CHORDS:
        assert chord in report["found_chords"]


def test_audio_engine_volume_and_mute():
    """Volume should clip between 0.0 and 1.0; muting should toggle correctly."""
    engine = AudioEngine()
    engine.set_volume(0.5)
    assert engine.volume == 0.5

    # Out of bounds clamping
    engine.set_volume(1.8)
    assert engine.volume == 1.0

    engine.set_volume(-0.5)
    assert engine.volume == 0.0

    # Muting
    assert engine.is_muted is False
    is_muted = engine.toggle_mute()
    assert is_muted is True
    assert engine.is_muted is True

    # Play while muted should return False
    played = engine.play_chord("am")
    assert played is False

    # Unmute
    engine.toggle_mute()
    assert engine.is_muted is False


def test_audio_engine_missing_chord_handling():
    """Requesting an unsupported or nonexistent chord should return False gracefully."""
    engine = AudioEngine()
    played = engine.play_chord("nonexistent_chord_xyz")
    assert played is False


def test_audio_engine_headless_mode():
    """Headless mode should initialize cleanly without hardware dependencies."""
    engine = AudioEngine(headless=True)
    assert engine.headless is True
    assert engine.play_chord("c") is False
