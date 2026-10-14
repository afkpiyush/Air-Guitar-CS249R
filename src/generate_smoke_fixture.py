"""
Generates synthetic smoke-test fixture data for offline verification,
testing, and CI without requiring real webcam capture.

IMPORTANT:
This data is purely synthetic geometry (clearly marked) and is NOT real collected
webcam data. Real models should be trained on observations collected via
notebooks/01_data_collection.ipynb.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.config import (
    SMOKE_TEST_DATA_DIR,
    CHORD_MODEL_DIR,
    STRUM_MODEL_DIR,
    SUPPORTED_CHORDS,
    STRUM_CLASSES,
    SEQUENCE_LENGTH,
    STRUM_FEATURE_DIM
)
from src.feature_extraction import get_chord_feature_names


def generate_smoke_test_fixtures():
    """Generates synthetic dataset files for offline pipeline testing."""
    SMOKE_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHORD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STRUM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    feature_names = get_chord_feature_names()
    num_features = len(feature_names)
    assert num_features == 98, f"Expected 98 chord features, got {num_features}"

    # 1. Generate Synthetic Chord Features
    # Create 3 sessions, each having samples for all 13 chords
    rows = []
    sessions = ["smoke_sess_01", "smoke_sess_02", "smoke_sess_03"]
    samples_per_chord = 10

    for sess_idx, session_id in enumerate(sessions):
        for chord_idx, chord in enumerate(SUPPORTED_CHORDS):
            # Deterministic base pattern per chord + small noise
            base_pattern = np.sin(np.linspace(0, 3.14 * (chord_idx + 1), num_features))
            for s in range(samples_per_chord):
                noise = np.random.normal(0, 0.05, num_features)
                feat = (base_pattern + noise).astype(np.float32)
                row_dict = {
                    "session_id": session_id,
                    "label": chord,
                    "label_encoded": chord_idx
                }
                for f_name, f_val in zip(feature_names, feat):
                    row_dict[f_name] = f_val
                rows.append(row_dict)

    df_chord = pd.DataFrame(rows)
    smoke_chord_path = SMOKE_TEST_DATA_DIR / "smoke_chord_features.csv"
    df_chord.to_csv(smoke_chord_path, index=False)
    print(f"[Smoke Fixture] Saved {len(df_chord)} synthetic chord rows to {smoke_chord_path}")

    # 2. Generate Synthetic Strumming Sequences
    num_sequences = 60
    X_seq = []
    y_seq = []
    seq_sessions = []

    for i in range(num_sequences):
        c_idx = i % len(STRUM_CLASSES)  # 0: idle, 1: downstroke, 2: upstroke
        # Temporal trajectory
        t = np.linspace(0, 1, SEQUENCE_LENGTH)
        seq = np.zeros((SEQUENCE_LENGTH, STRUM_FEATURE_DIM), dtype=np.float32)
        if c_idx == 1:  # downstroke: downward velocity in Y
            seq[:, 7] = 2.0 * t  # vy positive
            seq[:, 1] = t        # y moving down
        elif c_idx == 2:  # upstroke: upward velocity in Y
            seq[:, 7] = -2.0 * t # vy negative
            seq[:, 1] = 1.0 - t
        else:  # idle
            seq[:, :] = np.random.normal(0, 0.02, (SEQUENCE_LENGTH, STRUM_FEATURE_DIM))
            
        seq += np.random.normal(0, 0.03, seq.shape).astype(np.float32)
        X_seq.append(seq)
        y_seq.append(c_idx)
        seq_sessions.append(sessions[i % len(sessions)])

    X_seq_arr = np.array(X_seq, dtype=np.float32)
    y_seq_arr = np.array(y_seq, dtype=np.int32)
    seq_sess_arr = np.array(seq_sessions)

    smoke_strum_path = SMOKE_TEST_DATA_DIR / "smoke_strumming_sequences.npz"
    np.savez_compressed(
        smoke_strum_path,
        X=X_seq_arr,
        y=y_seq_arr,
        session_ids=seq_sess_arr,
        class_names=np.array(STRUM_CLASSES)
    )
    print(f"[Smoke Fixture] Saved {num_sequences} synthetic strum sequences to {smoke_strum_path}")

    # 3. Save Smoke Metadata
    smoke_meta = {
        "is_synthetic_smoke_test": True,
        "description": "Synthetic smoke-test fixture for pipeline verification only.",
        "num_chord_features": num_features,
        "feature_names": feature_names,
        "chord_classes": SUPPORTED_CHORDS,
        "strum_classes": STRUM_CLASSES,
        "sequence_length": SEQUENCE_LENGTH,
        "strum_feature_dim": STRUM_FEATURE_DIM
    }
    with open(SMOKE_TEST_DATA_DIR / "smoke_metadata.json", "w") as f:
        json.dump(smoke_meta, f, indent=2)

    # 4. Train Minimal Smoke Checkpoint Models (so tests & app can load valid models out of the box)
    if tf is not None:
        print("[Smoke Fixture] Building and exporting minimal verifiable model checkpoints...")
        
        # Chord MLP
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(SUPPORTED_CHORDS)
        joblib.dump(le, CHORD_MODEL_DIR / "label_encoder.joblib")
        with open(CHORD_MODEL_DIR / "feature_schema.json", "w") as f:
            json.dump({"features": feature_names, "num_features": num_features}, f, indent=2)

        chord_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(SUPPORTED_CHORDS), activation="softmax")
        ])
        chord_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        X_chord = df_chord[feature_names].values.astype(np.float32)
        y_chord = df_chord["label_encoded"].values.astype(np.int32)
        chord_model.fit(X_chord, y_chord, epochs=5, batch_size=16, verbose=0)
        chord_model.save(str(CHORD_MODEL_DIR / "chord_classifier.keras"))
        print(f"[Smoke Fixture] Saved smoke chord model to {CHORD_MODEL_DIR / 'chord_classifier.keras'}")

        # Strumming LSTM
        strum_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, STRUM_FEATURE_DIM)),
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dense(len(STRUM_CLASSES), activation="softmax")
        ])
        strum_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        strum_model.fit(X_seq_arr, y_seq_arr, epochs=5, batch_size=8, verbose=0)
        strum_model.save(str(STRUM_MODEL_DIR / "strumming_lstm.keras"))
        with open(STRUM_MODEL_DIR / "strum_config.json", "w") as f:
            json.dump({
                "sequence_length": SEQUENCE_LENGTH,
                "feature_dim": STRUM_FEATURE_DIM,
                "classes": STRUM_CLASSES
            }, f, indent=2)
        print(f"[Smoke Fixture] Saved smoke strum model to {STRUM_MODEL_DIR / 'strumming_lstm.keras'}")

    print("[Smoke Fixture] Fixture generation complete.")


if __name__ == "__main__":
    generate_smoke_test_fixtures()
