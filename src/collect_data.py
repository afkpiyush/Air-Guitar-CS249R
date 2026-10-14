"""
Interactive Webcam Dataset Collection Tool for Air Guitar ML.
Can be run directly from terminal or alongside JupyterLab.

Usage:
    python src/collect_data.py
    python src/collect_data.py --camera 0
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RAW_DATA_DIR,
    RAW_CHORD_CSV,
    RAW_STRUM_CSV,
    SUPPORTED_CHORDS,
    STRUM_CLASSES,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MIRROR_CAMERA
)
from src.hand_tracking import HandTracker


class DatasetCollector:
    """Manages incremental recording of hand landmarks to raw CSV files."""

    def __init__(self, camera_idx: int = CAMERA_INDEX, mirror: bool = MIRROR_CAMERA):
        self.camera_idx = camera_idx
        self.mirror = mirror
        self.tracker = HandTracker()
        self.session_id = f"sess_{time.strftime('%Y%m%d_%H%M%S')}"
        
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.chord_csv = RAW_CHORD_CSV
        self.strum_csv = RAW_STRUM_CSV
        self._init_csv_files()

        self.chord_counts = {c: 0 for c in SUPPORTED_CHORDS + ["neutral"]}
        self.strum_counts = {s: 0 for s in STRUM_CLASSES}
        self._load_existing_counts()

        self.active_chord_idx = 0
        self.active_strum_idx = 1  # default to downstroke
        self.recording_strum = False
        self.strum_buffer = []

    def _init_csv_files(self):
        coord_cols = [f"{axis}_{i}" for i in range(21) for axis in ("x", "y", "z")]
        if not self.chord_csv.exists() or self.chord_csv.stat().st_size == 0:
            cols = ["timestamp", "session_id", "sample_id", "handedness", "label", "confidence"] + coord_cols
            pd.DataFrame(columns=cols).to_csv(self.chord_csv, index=False)

        if not self.strum_csv.exists() or self.strum_csv.stat().st_size == 0:
            cols = ["timestamp", "session_id", "sequence_id", "frame_idx", "handedness", "label"] + coord_cols
            pd.DataFrame(columns=cols).to_csv(self.strum_csv, index=False)

    def _load_existing_counts(self):
        if self.chord_csv.exists() and self.chord_csv.stat().st_size > 100:
            try:
                df = pd.read_csv(self.chord_csv)
                for label, cnt in df["label"].value_counts().items():
                    if label in self.chord_counts:
                        self.chord_counts[label] = int(cnt)
            except Exception:
                pass

        if self.strum_csv.exists() and self.strum_csv.stat().st_size > 100:
            try:
                df = pd.read_csv(self.strum_csv)
                # Count unique sequences per label
                seq_df = df.groupby(["session_id", "sequence_id", "label"]).size().reset_index()
                for label, cnt in seq_df["label"].value_counts().items():
                    if label in self.strum_counts:
                        self.strum_counts[label] = int(cnt)
            except Exception:
                pass

    def save_chord_sample(self, hand_data, label: str) -> bool:
        if hand_data is None:
            return False
        lms = hand_data["landmarks"].flatten()
        sample_id = sum(self.chord_counts.values()) + 1
        row = [time.time(), self.session_id, sample_id, "Left", label, hand_data["confidence"]] + list(lms)
        pd.DataFrame([row]).to_csv(self.chord_csv, mode="a", header=False, index=False)
        self.chord_counts[label] += 1
        return True

    def save_strum_sequence(self, frames_list, label: str) -> bool:
        if len(frames_list) < 10:
            return False
        seq_id = self.strum_counts[label] + 1
        rows = []
        for f_idx, (ts, lms) in enumerate(frames_list):
            row = [ts, self.session_id, seq_id, f_idx, "Right", label] + list(lms.flatten())
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.strum_csv, mode="a", header=False, index=False)
        self.strum_counts[label] += 1
        return True

    def run(self):
        cap = cv2.VideoCapture(self.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print(f"[Error] Could not open camera at index {self.camera_idx}")
            return

        all_chords = SUPPORTED_CHORDS + ["neutral"]
        print("=" * 70)
        print("Starting Interactive Air Guitar Dataset Collector")
        print(f"Session ID: {self.session_id}")
        print("=" * 70)

        recording_seq_frames = []
        is_recording_seq = False
        target_seq_len = 15

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame, detected_hands = self.tracker.process_frame(frame, mirror=self.mirror)
                self.tracker.draw_landmarks(frame, detected_hands)

                h, w, _ = frame.shape
                curr_chord = all_chords[self.active_chord_idx]
                curr_strum = STRUM_CLASSES[self.active_strum_idx]

                # HUD Overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 120), (15, 15, 15), -1)
                cv2.rectangle(overlay, (0, h - 50), (w, h), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                # Status info
                cv2.putText(
                    frame, f"CHORD TARGET [c]: {curr_chord.upper()} (Count: {self.chord_counts[curr_chord]})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, f"STRUM TARGET [1/2/3]: {curr_strum.upper()} (Sequences: {self.strum_counts[curr_strum]})",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA
                )
                
                # Hand detection indicators
                left_stat = "DETECTED" if "Left" in detected_hands else "MISSING"
                left_col = (0, 255, 0) if "Left" in detected_hands else (0, 0, 255)
                right_stat = "DETECTED" if "Right" in detected_hands else "MISSING"
                right_col = (0, 255, 0) if "Right" in detected_hands else (0, 0, 255)
                
                cv2.putText(frame, f"Left Hand: {left_stat}", (w - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, left_col, 1)
                cv2.putText(frame, f"Right Hand: {right_stat}", (w - 280, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, right_col, 1)

                # Sequence recording indicator
                if is_recording_seq:
                    cv2.putText(
                        frame, f"● RECORDING STRUM SEQUENCE ({len(recording_seq_frames)}/{target_seq_len})",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA
                    )
                    if "Right" in detected_hands:
                        recording_seq_frames.append((time.time(), detected_hands["Right"]["landmarks"]))
                    if len(recording_seq_frames) >= target_seq_len:
                        self.save_strum_sequence(recording_seq_frames, curr_strum)
                        print(f"[Saved] Recorded sequence for '{curr_strum}' (#{self.strum_counts[curr_strum]})")
                        recording_seq_frames = []
                        is_recording_seq = False

                # Bottom navigation instructions
                controls = "[SPACE]: Record Chord  |  [R]: Record Strum  |  [C]: Next Chord  |  [Q]: Quit"
                cv2.putText(frame, controls, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.imshow("Air Guitar Dataset Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):
                    break
                elif key == ord('c'):
                    self.active_chord_idx = (self.active_chord_idx + 1) % len(all_chords)
                elif key == ord(' '):
                    # Accept Left hand if present, otherwise whichever hand is visible
                    target_hand = detected_hands.get("Left") or (next(iter(detected_hands.values())) if detected_hands else None)
                    handedness_tag = "Left" if "Left" in detected_hands else ("Right" if "Right" in detected_hands else "Unknown")
                    if target_hand is not None:
                        self.save_chord_sample(target_hand, curr_chord)
                        print(f"[Saved] Chord sample for '{curr_chord}' (Hand: {handedness_tag}, Total: {self.chord_counts[curr_chord]})")
                    else:
                        print("[Warning] No hand visible to save sample!")
                elif key == ord('1'):
                    self.active_strum_idx = 0  # idle
                elif key == ord('2'):
                    self.active_strum_idx = 1  # downstroke
                elif key == ord('3'):
                    self.active_strum_idx = 2  # upstroke
                elif key == ord('r'):
                    if not is_recording_seq:
                        is_recording_seq = True
                        recording_seq_frames = []
                        print(f"[Recording] Perform a '{curr_strum}' gesture now...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()
            print("\nDataset collection session finished.")
            print(f"Chord samples saved to: {self.chord_csv}")
            print(f"Strumming frames saved to: {self.strum_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Guitar Data Collection")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror mode")
    args = parser.parse_args()

    collector = DatasetCollector(camera_idx=args.camera, mirror=not args.no_mirror)
    collector.run()
