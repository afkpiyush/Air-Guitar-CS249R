"""
Real-time inference pipeline and trigger engine for Air Guitar.
Coordinates model loading, feature extraction, chord prediction smoothing,
temporal strumming sequence classification, and debounce state machine.
"""

from typing import Tuple, Optional, Dict, Any, List
from collections import deque
from pathlib import Path
import time
import json
import numpy as np
import joblib

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.config import (
    CHORD_MODEL_PATH,
    CHORD_LABEL_ENCODER_PATH,
    CHORD_SCHEMA_PATH,
    STRUM_MODEL_PATH,
    STRUM_CONFIG_PATH,
    CHORD_CONFIDENCE_THRESHOLD,
    STRUM_CONFIDENCE_THRESHOLD,
    CHORD_SMOOTHING_FRAMES,
    STRUM_COOLDOWN_MS,
    SEQUENCE_LENGTH,
    STRUM_CLASSES
)
from src.feature_extraction import (
    extract_chord_features,
    extract_strum_frame_features
)


class ChordPredictor:
    """Predicts guitar chords from static hand landmarks with confidence thresholding and smoothing."""

    def __init__(
        self,
        model_path: Path = CHORD_MODEL_PATH,
        encoder_path: Path = CHORD_LABEL_ENCODER_PATH,
        confidence_threshold: float = CHORD_CONFIDENCE_THRESHOLD,
        smoothing_frames: int = CHORD_SMOOTHING_FRAMES
    ):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.confidence_threshold = confidence_threshold
        self.smoothing_frames = smoothing_frames

        self.model = None
        self.label_encoder = None
        self.classes_: List[str] = []
        self.history: deque = deque(maxlen=smoothing_frames)
        self.is_ready = False

        self._load()

    def _load(self) -> bool:
        """Loads trained Keras model and label encoder."""
        if not self.model_path.exists():
            return False
        if tf is None:
            return False

        if self.encoder_path.exists():
            try:
                self.label_encoder = joblib.load(str(self.encoder_path))
                self.classes_ = list(self.label_encoder.classes_)
            except Exception:
                pass

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            self.is_ready = True
            return True
        except Exception as e:
            # Fallback: Keras version mismatch during deserialization (e.g. quantization_config)
            # Reconstruct model architecture and load weights
            try:
                num_classes = len(self.classes_) if self.classes_ else len(SUPPORTED_CHORDS)
                fallback_model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(98,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.20),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                fallback_model.load_weights(str(self.model_path))
                self.model = fallback_model
                self.is_ready = True
                return True
            except Exception as inner_e:
                print(f"[ChordPredictor Error] Failed to load model: {e}")
                self.is_ready = False
                return False

    def predict(self, landmarks: np.ndarray) -> Tuple[Optional[str], float, str]:
        """
        Predicts chord label for detected hand landmarks.
        
        Args:
            landmarks: np.ndarray of shape (21, 3)
            
        Returns:
            smoothed_chord: Most frequent chord in recent window (or None if unconfident)
            confidence: Float probability of the raw top class
            raw_prediction: Unfiltered raw top predicted chord class
        """
        if not self.is_ready or self.model is None:
            return None, 0.0, "Model Not Loaded"

        features = extract_chord_features(landmarks).reshape(1, -1)
        probs = self.model.predict(features, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        if self.classes_ and top_idx < len(self.classes_):
            raw_chord = self.classes_[top_idx]
        else:
            raw_chord = str(top_idx)

        # Apply confidence threshold
        if confidence >= self.confidence_threshold:
            self.history.append(raw_chord)
        else:
            self.history.append("uncertain")

        # Temporal smoothing via majority vote over recent frames
        valid_votes = [c for c in self.history if c != "uncertain"]
        if valid_votes:
            # Majority vote
            smoothed = max(set(valid_votes), key=valid_votes.count)
        else:
            smoothed = None

        return smoothed, confidence, raw_chord

    def reset_history(self):
        """Clears smoothing history buffer."""
        self.history.clear()


class StrummingDetector:
    """
    Maintains a rolling buffer of right-hand movement frames, classifies temporal
    strumming gestures via LSTM, and debounces audio trigger events.
    """

    def __init__(
        self,
        model_path: Path = STRUM_MODEL_PATH,
        config_path: Path = STRUM_CONFIG_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
        confidence_threshold: float = STRUM_CONFIDENCE_THRESHOLD,
        cooldown_ms: int = STRUM_COOLDOWN_MS
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.cooldown_ms = cooldown_ms

        self.model = None
        self.classes: List[str] = STRUM_CLASSES
        self.is_ready = False

        self.frame_buffer: deque = deque(maxlen=sequence_length)
        self.prev_landmarks: Optional[np.ndarray] = None
        self.prev_time: float = time.time()
        self.last_strum_time: float = 0.0
        self.last_state: str = "idle"

        self._load()

    def _load(self) -> bool:
        """Loads trained LSTM model."""
        if not self.model_path.exists():
            return False
        if tf is None:
            return False

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                    self.classes = cfg.get("classes", STRUM_CLASSES)
            except Exception:
                pass

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            self.is_ready = True
            return True
        except Exception as e:
            # Fallback: Keras version mismatch during deserialization
            try:
                seq_len = SEQUENCE_LENGTH
                feat_dim = 18
                num_strum_classes = len(self.classes)
                fallback_lstm = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(seq_len, feat_dim)),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(num_strum_classes, activation='softmax')
                ])
                fallback_lstm.load_weights(str(self.model_path))
                self.model = fallback_lstm
                self.is_ready = True
                return True
            except Exception as inner_e:
                print(f"[StrummingDetector Error] Failed to load LSTM model: {e}")
                self.is_ready = False
                return False

    def update(
        self,
        landmarks: np.ndarray,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Updates the strumming tracker with a new frame of right-hand landmarks.
        
        Args:
            landmarks: np.ndarray of shape (21, 3)
            current_time: Optional current timestamp (seconds)
            
        Returns:
            triggered: True if a fresh strumming action occurred and passed cooldown
            predicted_action: 'idle', 'downstroke', or 'upstroke'
            confidence: Probability of predicted action
        """
        now = time.time() if current_time is None else current_time
        dt = max(now - self.prev_time, 1e-4)
        self.prev_time = now

        # Extract 18 movement features
        frame_feat = extract_strum_frame_features(
            landmarks,
            self.prev_landmarks,
            dt=dt
        )
        self.prev_landmarks = landmarks.copy()
        self.frame_buffer.append(frame_feat)

        # Need full sequence window for LSTM
        if len(self.frame_buffer) < self.sequence_length or not self.is_ready or self.model is None:
            return False, "buffering" if self.is_ready else "not_ready", 0.0

        # Construct input sequence (1, sequence_length, 18)
        seq_input = np.array(self.frame_buffer, dtype=np.float32).reshape(1, self.sequence_length, -1)
        probs = self.model.predict(seq_input, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        action = self.classes[top_idx] if top_idx < len(self.classes) else "idle"

        # Check debounce and state transition
        triggered = False
        time_since_last_strum_ms = (now - self.last_strum_time) * 1000.0

        if action in ("downstroke", "upstroke") and confidence >= self.confidence_threshold:
            # Transition from idle OR cooldown period expired
            if time_since_last_strum_ms >= self.cooldown_ms:
                triggered = True
                self.last_strum_time = now

        self.last_state = action
        return triggered, action, confidence

    def reset(self):
        """Resets buffer and state tracking."""
        self.frame_buffer.clear()
        self.prev_landmarks = None
        self.last_state = "idle"
