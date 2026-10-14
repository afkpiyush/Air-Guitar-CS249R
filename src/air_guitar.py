"""
Real-Time Air Guitar Application.
Executes live webcam-based hand tracking, left-hand chord classification,
right-hand temporal strumming detection, and low-latency acoustic guitar playback.

Usage:
    python src/air_guitar.py
    python src/air_guitar.py --camera 0 --volume 0.9
"""

import sys
import os
import time
import argparse
from typing import Optional, Dict, Any
import cv2
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CHORD_HAND,
    STRUM_HAND,
    MIRROR_CAMERA,
    DEFAULT_VOLUME,
    CHORD_MODEL_PATH,
    STRUM_MODEL_PATH
)
from src.hand_tracking import HandTracker
from src.inference import ChordPredictor, StrummingDetector
from src.audio_engine import AudioEngine


def draw_hud(
    frame: np.ndarray,
    chord: Optional[str],
    chord_conf: float,
    strum_action: str,
    strum_conf: float,
    strum_triggered: bool,
    volume: float,
    is_muted: bool,
    chord_hand_label: str,
    strum_hand_label: str,
    chord_ready: bool,
    strum_ready: bool,
    fps: float
) -> np.ndarray:
    """Renders real-time HUD status overlay onto frame."""
    h, w, _ = frame.shape
    
    # Top overlay bar (dark transparent background)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    # Bottom status bar
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title
    cv2.putText(
        frame, "AIR GUITAR ML (Software Edition)",
        (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (w - 140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA
    )

    # Chord Status (Left display)
    chord_color = (0, 255, 0) if chord else (100, 100, 100)
    chord_text = f"Chord ({chord_hand_label} Hand): {chord.upper() if chord else 'NONE'}"
    if not chord_ready:
        chord_text += " [TRAIN: 03_chord_classification]"
        chord_color = (0, 165, 255)
    cv2.putText(
        frame, chord_text,
        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, chord_color, 2, cv2.LINE_AA
    )
    if chord_ready and chord:
        cv2.putText(
            frame, f"Confidence: {chord_conf:.1%}",
            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
        )

    # Strumming Status (Right display)
    strum_color = (0, 255, 255) if strum_triggered else (200, 200, 200)
    strum_text = f"Strum ({strum_hand_label} Hand): {strum_action.upper()}"
    if not strum_ready:
        strum_text += " [TRAIN: 04_strumming_lstm]"
        strum_color = (0, 165, 255)
    cv2.putText(
        frame, strum_text,
        (w // 2, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, strum_color, 2, cv2.LINE_AA
    )
    if strum_ready:
        cv2.putText(
            frame, f"Confidence: {strum_conf:.1%}",
            (w // 2, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
        )

    # Trigger Animation Visual Flash
    if strum_triggered:
        cv2.putText(
            frame, "♫ STRUM! ♫",
            (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 255), 3, cv2.LINE_AA
        )
        cv2.circle(frame, (w // 2, h // 2 + 50), 40, (0, 255, 255), 4)

    # Bottom Bar Controls
    mute_status = "MUTED" if is_muted else f"VOL: {int(volume * 100)}%"
    controls_text = f"[Q]: Quit  |  [M]: Mute ({mute_status})  |  [+/-]: Vol  |  [S]: Swap Hands"
    cv2.putText(
        frame, controls_text,
        (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
    )

    return frame


def run_air_guitar(
    camera_idx: int = CAMERA_INDEX,
    chord_hand: str = CHORD_HAND,
    strum_hand: str = STRUM_HAND,
    mirror: bool = MIRROR_CAMERA,
    volume: float = DEFAULT_VOLUME
):
    """Main application loop."""
    print("=" * 70)
    print("Starting Air Guitar ML (Pure Software Edition)")
    print(f"Chord Hand: {chord_hand}  |  Strum Hand: {strum_hand}  |  Mirror: {mirror}")
    print("=" * 70)

    # Check model artifacts
    chord_ready = CHORD_MODEL_PATH.exists()
    strum_ready = STRUM_MODEL_PATH.exists()
    if not chord_ready:
        print(f"[Notice] Chord model not found at {CHORD_MODEL_PATH}.")
        print("         Run 'notebooks/03_chord_classification.ipynb' to train the chord model.")
    if not strum_ready:
        print(f"[Notice] Strumming model not found at {STRUM_MODEL_PATH}.")
        print("         Run 'notebooks/04_strumming_lstm.ipynb' to train the strumming model.")

    # Initialize modules
    tracker = HandTracker()
    audio = AudioEngine(volume=volume)
    chord_predictor = ChordPredictor()
    strum_detector = StrummingDetector()

    # Verify audio assets
    asset_status = audio.verify_assets()
    print(f"[Audio] Loaded {asset_status['found_count']}/{asset_status['total_supported']} chord audio files.")
    if asset_status['missing_count'] > 0:
        print(f"[Audio Warning] Missing files: {asset_status['missing_chords']}")

    # Open webcam
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[Fatal Error] Could not open webcam at index {camera_idx}.")
        print("Please check your camera connection or specify a different camera with --camera <index>.")
        tracker.close()
        audio.cleanup()
        return

    print("\nWebcam successfully started. Press 'q' to exit.\n")

    current_volume = volume
    prev_frame_time = time.time()
    fps = 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to grab webcam frame.")
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_time, 1e-4))
            prev_frame_time = now

            # Process hand tracking
            frame, detected_hands = tracker.process_frame(frame, mirror=mirror)
            tracker.draw_landmarks(frame, detected_hands)

            # Left Hand: Chord recognition
            predicted_chord = None
            chord_conf = 0.0
            if chord_hand in detected_hands:
                hand_data = detected_hands[chord_hand]
                predicted_chord, chord_conf, _ = chord_predictor.predict(hand_data["landmarks"])
            else:
                chord_predictor.reset_history()

            # Right Hand: Strumming recognition
            strum_triggered = False
            strum_action = "idle"
            strum_conf = 0.0
            if strum_hand in detected_hands:
                hand_data = detected_hands[strum_hand]
                strum_triggered, strum_action, strum_conf = strum_detector.update(
                    hand_data["landmarks"],
                    current_time=now
                )
            else:
                strum_detector.reset()

            # Audio Playback Event Trigger
            if strum_triggered:
                if predicted_chord and predicted_chord in audio.sounds:
                    audio.play_chord(predicted_chord)
                    print(f"♫ Strum triggered -> Playing chord: {predicted_chord.upper()} (conf: {chord_conf:.2f})")
                else:
                    if not predicted_chord:
                        print(f"♫ Strum triggered, but no confident chord selected on {chord_hand} hand.")

            # Draw HUD
            frame = draw_hud(
                frame,
                chord=predicted_chord,
                chord_conf=chord_conf,
                strum_action=strum_action,
                strum_conf=strum_conf,
                strum_triggered=strum_triggered,
                volume=current_volume,
                is_muted=audio.is_muted,
                chord_hand_label=chord_hand,
                strum_hand_label=strum_hand,
                chord_ready=chord_predictor.is_ready,
                strum_ready=strum_detector.is_ready,
                fps=fps
            )

            cv2.imshow("Air Guitar ML", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('m'):
                audio.toggle_mute()
            elif key in (ord('+'), ord('=')):
                current_volume = min(1.0, current_volume + 0.05)
                audio.set_volume(current_volume)
            elif key in (ord('-'), ord('_')):
                current_volume = max(0.0, current_volume - 0.05)
                audio.set_volume(current_volume)
            elif key == ord('s'):
                # Swap handedness
                chord_hand, strum_hand = strum_hand, chord_hand
                chord_predictor.reset_history()
                strum_detector.reset()
                print(f"[Handedness Swapped] Chord Hand: {chord_hand}, Strum Hand: {strum_hand}")

    finally:
        print("\nReleasing resources and exiting...")
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        audio.cleanup()
        print("Air Guitar shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Guitar ML Application")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--chord-hand", type=str, default=CHORD_HAND, choices=["Left", "Right"])
    parser.add_argument("--strum-hand", type=str, default=STRUM_HAND, choices=["Left", "Right"])
    parser.add_argument("--no-mirror", action="store_true", help="Disable camera mirroring")
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME, help="Volume (0.0 to 1.0)")
    args = parser.parse_args()

    run_air_guitar(
        camera_idx=args.camera,
        chord_hand=args.chord_hand,
        strum_hand=args.strum_hand,
        mirror=not args.no_mirror,
        volume=args.volume
    )
