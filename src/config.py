"""
Configuration module for Air-Guitar-CS249R (Software-Only ML Air Guitar).
Centralizes all hardware-free parameters, paths, model settings, and thresholds.
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
SMOKE_TEST_DATA_DIR = DATASET_DIR / "smoke_test"

# Raw dataset files
RAW_CHORD_CSV = RAW_DATA_DIR / "chord_landmarks.csv"
RAW_STRUM_CSV = RAW_DATA_DIR / "strumming_frames.csv"

# Processed dataset files
PROCESSED_CHORD_CSV = PROCESSED_DATA_DIR / "chord_features.csv"
PROCESSED_STRUM_NPZ = PROCESSED_DATA_DIR / "strumming_sequences.npz"
PROCESSED_METADATA_JSON = PROCESSED_DATA_DIR / "metadata.json"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHORD_MODEL_DIR = MODELS_DIR / "chord_classifier"
STRUM_MODEL_DIR = MODELS_DIR / "strumming_lstm"

CHORD_MODEL_PATH = CHORD_MODEL_DIR / "chord_classifier.keras"
CHORD_LABEL_ENCODER_PATH = CHORD_MODEL_DIR / "label_encoder.joblib"
CHORD_SCHEMA_PATH = CHORD_MODEL_DIR / "feature_schema.json"

STRUM_MODEL_PATH = STRUM_MODEL_DIR / "strumming_lstm.keras"
STRUM_CONFIG_PATH = STRUM_MODEL_DIR / "strum_config.json"

# Audio paths
SOUNDS_DIR = PROJECT_ROOT / "sounds"
CHORD_AUDIO_DIR = SOUNDS_DIR / "chord_recordings"

# Hand assignment configuration
# By default: Left hand selects chords, Right hand performs strumming
CHORD_HAND = "Left"     # MediaPipe label for chord hand
STRUM_HAND = "Right"    # MediaPipe label for strumming hand
MIRROR_CAMERA = True    # Flip webcam horizontally for natural mirror display

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# MediaPipe Hands settings
MP_MAX_HANDS = 2
MP_MIN_DETECTION_CONFIDENCE = 0.70
MP_MIN_TRACKING_CONFIDENCE = 0.60

# Supported Chords (13 chords + neutral/unknown)
SUPPORTED_CHORDS = [
    "a", "am", "b", "bm",
    "c", "cm", "d", "dm",
    "e", "em", "f", "g", "gm"
]
NEUTRAL_CHORD_LABEL = "neutral"

# Supported Strumming Actions
STRUM_CLASSES = ["idle", "downstroke", "upstroke"]

# Temporal sequence parameters for Strumming LSTM
SEQUENCE_LENGTH = 15  # frames in sliding window (~0.5s at 30fps)
STRUM_FEATURE_DIM = 18

# Real-time inference & trigger debouncing
CHORD_CONFIDENCE_THRESHOLD = 0.60
STRUM_CONFIDENCE_THRESHOLD = 0.70
CHORD_SMOOTHING_FRAMES = 5  # Majority vote over last N frames
STRUM_COOLDOWN_MS = 250     # Min milliseconds between consecutive strum triggers

# Audio Engine settings
AUDIO_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 512     # Low latency buffer
DEFAULT_VOLUME = 0.85
