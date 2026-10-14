"""
Hand tracking module using MediaPipe Hands.
Extracts 21 3D landmarks for both hands, resolves handedness,
and provides visualization drawing utilities.
"""

import sys
import types
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

# Defensive guard against MediaPipe optional_dependencies bug:
# MediaPipe's tasks API tries 'from tensorflow.tools.docs import doc_controls',
# which raises an unhandled ImportError if TensorFlow and protobuf have a version mismatch.
if "tensorflow.tools.docs" not in sys.modules:
    try:
        from tensorflow.tools.docs import doc_controls  # type: ignore
    except Exception:
        # Provide lightweight mock so MediaPipe import succeeds seamlessly
        class _DummyDocControls:
            @staticmethod
            def do_not_generate_docs(fn):
                return fn
            @staticmethod
            def doc_private(fn):
                return fn

        _tf_docs_dummy = types.ModuleType("tensorflow.tools.docs")
        _tf_docs_dummy.doc_controls = _DummyDocControls
        sys.modules["tensorflow.tools.docs"] = _tf_docs_dummy

import mediapipe as mp

from src.config import (
    MP_MAX_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MIRROR_CAMERA
)


class HandTracker:
    """Wrapper around MediaPipe Hands solution with convenient handedness resolution."""

    def __init__(
        self,
        max_hands: int = MP_MAX_HANDS,
        min_detection_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(
        self,
        frame: np.ndarray,
        mirror: bool = MIRROR_CAMERA
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
        """
        Processes a BGR webcam frame, optionally flips horizontally for mirroring,
        detects hands, and returns annotated/unannotated frame + detected hands data.
        
        Returns:
            processed_frame: BGR frame (mirrored if mirror=True)
            detected_hands: dict mapping 'Left' and/or 'Right' to:
                {
                    'landmarks': np.ndarray of shape (21, 3) (x, y, z normalized),
                    'confidence': float,
                    'raw_landmarks': list of mp landmark objects
                }
        """
        if mirror:
            frame = cv2.flip(frame, 1)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        detected_hands: Dict[str, Dict[str, Any]] = {}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # MediaPipe handedness label
                label = hand_info.classification[0].label
                confidence = float(hand_info.classification[0].score)
                
                # When frame is mirrored, MediaPipe's Left/Right may need swapping to match user perspective
                if mirror:
                    # In a mirrored selfie view, user's physical left hand appears on user's left
                    # MediaPipe already computes handedness from camera's POV
                    resolved_label = "Left" if label == "Right" else "Right"
                else:
                    resolved_label = label
                    
                coords = []
                for lm in hand_lms.landmark:
                    coords.append([lm.x, lm.y, lm.z])
                coords_arr = np.array(coords, dtype=np.float32)
                
                detected_hands[resolved_label] = {
                    "landmarks": coords_arr,
                    "confidence": confidence,
                    "raw_landmarks": hand_lms
                }
                
        return frame, detected_hands

    def draw_landmarks(
        self,
        frame: np.ndarray,
        detected_hands: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """Draws skeleton and landmarks for all detected hands onto the frame."""
        for hand_label, hand_data in detected_hands.items():
            raw_lms = hand_data.get("raw_landmarks")
            if raw_lms is not None:
                self.mp_drawing.draw_landmarks(
                    frame,
                    raw_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame

    def close(self):
        """Releases MediaPipe resources."""
        self.hands.close()
