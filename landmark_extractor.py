"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
Landmark Extraction & Preprocessing Module
=============================================================================
This module encapsulates Google MediaPipe (supporting modern Tasks Vision API
with automatic model asset management and legacy Solutions fallback) to detect,
extract, and normalize multi-modal spatial landmarks (Pose, Face, Left Hand, Right Hand)
from video frames.

Matrix Dimensions per Video Frame:
----------------------------------
1. Pose Landmarks:
   - 33 landmarks × 4 attributes (x, y, z, visibility) = 132 values
   - Shape: (132,)
2. Face Landmarks:
   - 468 landmarks × 3 attributes (x, y, z)            = 1404 values
   - Shape: (1404,)
3. Left Hand Landmarks:
   - 21 landmarks × 3 attributes (x, y, z)             = 63 values
   - Shape: (63,)
4. Right Hand Landmarks:
   - 21 landmarks × 3 attributes (x, y, z)             = 63 values
   - Shape: (63,)
-----------------------------------------------------------------------------
TOTAL FLATTENED KEYPOINT VECTOR PER FRAME:
   132 + 1404 + 63 + 63 = 1662 float32 values -> Shape: (1662,)
=============================================================================
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    POSE_FEATURES_TOTAL,
    FACE_FEATURES_TOTAL,
    HAND_FEATURES_TOTAL,
    TOTAL_KEYPOINTS_PER_FRAME,
)

# Model download URLs for MediaPipe Tasks API
MODEL_URLS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

# Standard skeleton connection pairs
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Shoulders & Arms
    (11, 23), (12, 24), (23, 24),                      # Torso
    (23, 25), (24, 26), (25, 27), (26, 28),           # Legs
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),(0, 17) # Pinky & Palm
]


@dataclass
class HolisticResults:
    """Standardized representation of multi-modal landmarks."""
    pose_landmarks: Optional[List[Any]] = None
    face_landmarks: Optional[List[Any]] = None
    left_hand_landmarks: Optional[List[Any]] = None
    right_hand_landmarks: Optional[List[Any]] = None


def ensure_model_downloaded(model_name: str, models_dir: Path) -> Path:
    """Ensures that the required MediaPipe model asset exists locally."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name
    if not model_path.exists():
        url = MODEL_URLS[model_name]
        print(f"[*] Downloading {model_name} from Google MediaPipe repo...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"[+] Downloaded {model_name} successfully.")
    return model_path


class MediaPipeHolisticExtractor:
    """
    Modular landmark extractor supporting modern MediaPipe Tasks Vision API
    (Python 3.10+) with fallback to legacy solutions.
    """

    def __init__(
        self,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        models_dir: Path = Path("models"),
    ) -> None:
        """
        Initializes the holistic detectors.
        """
        self.models_dir = models_dir
        self.use_tasks_api = not hasattr(mp, "solutions")

        if self.use_tasks_api:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self.BaseOptions = mp_python.BaseOptions
            self.VisionRunningMode = vision.RunningMode

            # 1. Pose Landmarker
            pose_path = ensure_model_downloaded("pose_landmarker.task", models_dir)
            pose_options = vision.PoseLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=self.VisionRunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

            # 2. Hand Landmarker (tracks up to 2 hands)
            hand_path = ensure_model_downloaded("hand_landmarker.task", models_dir)
            hand_options = vision.HandLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=self.VisionRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

            # 3. Face Landmarker
            face_path = ensure_model_downloaded("face_landmarker.task", models_dir)
            face_options = vision.FaceLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=str(face_path)),
                running_mode=self.VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        else:
            # Legacy solutions fallback
            self.mp_holistic = mp.solutions.holistic
            self.legacy_holistic = self.mp_holistic.Holistic(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        # Pre-allocated zero arrays for missing landmark fallback (Performance optimization)
        self._empty_pose = np.zeros(POSE_FEATURES_TOTAL, dtype=np.float32)      # Shape: (132,)
        self._empty_face = np.zeros(FACE_FEATURES_TOTAL, dtype=np.float32)      # Shape: (1404,)
        self._empty_hand = np.zeros(HAND_FEATURES_TOTAL, dtype=np.float32)      # Shape: (63,)

    def process_frame(self, bgr_image: np.ndarray) -> HolisticResults:
        """
        Processes a single BGR image frame and returns standard HolisticResults.

        Args:
            bgr_image: Input frame from OpenCV (BGR), Shape: (H, W, 3)

        Returns:
            HolisticResults containing standardized landmarks.
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            pose_res = self.pose_detector.detect(mp_image)
            hand_res = self.hand_detector.detect(mp_image)
            face_res = self.face_detector.detect(mp_image)

            # Extract pose
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            # Extract face
            face_lms = face_res.face_landmarks[0] if face_res.face_landmarks else None

            # Extract hands by handedness
            lh_lms = None
            rh_lms = None
            if hand_res.hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_res.hand_landmarks):
                    handedness_label = hand_res.handedness[idx][0].category_name
                    # Note: MediaPipe returns mirrored labels for front camera by default
                    if handedness_label.lower() == "left":
                        lh_lms = hand_landmarks
                    else:
                        rh_lms = hand_landmarks

            return HolisticResults(
                pose_landmarks=pose_lms,
                face_landmarks=face_lms,
                left_hand_landmarks=lh_lms,
                right_hand_landmarks=rh_lms,
            )
        else:
            rgb_image.flags.writeable = False
            results = self.legacy_holistic.process(rgb_image)
            rgb_image.flags.writeable = True

            pose_lms = results.pose_landmarks.landmark if results.pose_landmarks else None
            face_lms = results.face_landmarks.landmark if results.face_landmarks else None
            lh_lms = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
            rh_lms = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

            return HolisticResults(
                pose_landmarks=pose_lms,
                face_landmarks=face_lms,
                left_hand_landmarks=lh_lms,
                right_hand_landmarks=rh_lms,
            )

    def extract_keypoints(self, results: HolisticResults) -> np.ndarray:
        """
        Extracts and flattens spatial landmark coordinates into a unified 1D NumPy array.
        Applies zero-padding if any landmark subset (e.g. left hand, right hand) is missing
        to ensure deterministic tensor shapes for downstream LSTM models.

        Args:
            results: Standardized HolisticResults.

        Returns:
            np.ndarray: 1D feature vector of shape (1662,), dtype=np.float32
        """
        # ---------------------------------------------------------------------
        # 1. Pose Keypoints: 33 landmarks × [x, y, z, visibility] -> (132,)
        # ---------------------------------------------------------------------
        if results.pose_landmarks:
            pose_list = []
            for lm in results.pose_landmarks[:33]:
                vis = getattr(lm, "visibility", 1.0)
                pose_list.extend([lm.x, lm.y, lm.z, vis if vis is not None else 1.0])
            pose = np.array(pose_list, dtype=np.float32)
            if pose.shape[0] < POSE_FEATURES_TOTAL:
                pose = np.pad(pose, (0, POSE_FEATURES_TOTAL - pose.shape[0]))
        else:
            pose = self._empty_pose

        # ---------------------------------------------------------------------
        # 2. Face Keypoints: 468 landmarks × [x, y, z] -> (1404,)
        # ---------------------------------------------------------------------
        if results.face_landmarks:
            face_list = []
            for lm in results.face_landmarks[:468]:
                face_list.extend([lm.x, lm.y, lm.z])
            face = np.array(face_list, dtype=np.float32)
            if face.shape[0] < FACE_FEATURES_TOTAL:
                face = np.pad(face, (0, FACE_FEATURES_TOTAL - face.shape[0]))
        else:
            face = self._empty_face

        # ---------------------------------------------------------------------
        # 3. Left Hand Keypoints: 21 landmarks × [x, y, z] -> (63,)
        # ---------------------------------------------------------------------
        if results.left_hand_landmarks:
            lh_list = []
            for lm in results.left_hand_landmarks[:21]:
                lh_list.extend([lm.x, lm.y, lm.z])
            lh = np.array(lh_list, dtype=np.float32)
            if lh.shape[0] < HAND_FEATURES_TOTAL:
                lh = np.pad(lh, (0, HAND_FEATURES_TOTAL - lh.shape[0]))
        else:
            lh = self._empty_hand

        # ---------------------------------------------------------------------
        # 4. Right Hand Keypoints: 21 landmarks × [x, y, z] -> (63,)
        # ---------------------------------------------------------------------
        if results.right_hand_landmarks:
            rh_list = []
            for lm in results.right_hand_landmarks[:21]:
                rh_list.extend([lm.x, lm.y, lm.z])
            rh = np.array(rh_list, dtype=np.float32)
            if rh.shape[0] < HAND_FEATURES_TOTAL:
                rh = np.pad(rh, (0, HAND_FEATURES_TOTAL - rh.shape[0]))
        else:
            rh = self._empty_hand

        # ---------------------------------------------------------------------
        # 5. Concatenate all modalities: [132 + 1404 + 63 + 63] = 1662 keypoints
        # ---------------------------------------------------------------------
        keypoints = np.concatenate([pose, face, lh, rh], dtype=np.float32)

        # Guarantee shape invariant
        assert keypoints.shape == (TOTAL_KEYPOINTS_PER_FRAME,), (
            f"Expected keypoints shape ({TOTAL_KEYPOINTS_PER_FRAME},), "
            f"got {keypoints.shape}"
        )

        return keypoints

    def draw_styled_landmarks(
        self,
        image: np.ndarray,
        results: HolisticResults,
        draw_face_mesh: bool = False,
    ) -> np.ndarray:
        """
        Renders styled skeleton and joint annotations on the OpenCV BGR frame.

        Args:
            image: OpenCV BGR image frame, Shape: (H, W, 3)
            results: Standardized HolisticResults.
            draw_face_mesh: If True, draws subtle facial mesh dots.

        Returns:
            Annotated OpenCV BGR image.
        """
        annotated_image = image.copy()
        h, w, _ = image.shape

        # Helper to convert normalized coordinate to pixel
        def to_px(lm):
            return int(lm.x * w), int(lm.y * h)

        # 1. Draw Face Landmarks
        if draw_face_mesh and results.face_landmarks:
            for lm in results.face_landmarks:
                px = to_px(lm)
                cv2.circle(annotated_image, px, 1, (180, 180, 180), -1)

        # 2. Draw Pose Landmarks & Connections
        if results.pose_landmarks:
            # Draw Connections
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(results.pose_landmarks) and end_idx < len(results.pose_landmarks):
                    p1 = to_px(results.pose_landmarks[start_idx])
                    p2 = to_px(results.pose_landmarks[end_idx])
                    cv2.line(annotated_image, p1, p2, (80, 255, 120), 2, cv2.LINE_AA)
            # Draw Joints
            for lm in results.pose_landmarks:
                px = to_px(lm)
                cv2.circle(annotated_image, px, 3, (80, 110, 10), -1)

        # 3. Draw Left Hand (Cyan / Teal palette)
        if results.left_hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(results.left_hand_landmarks) and end_idx < len(results.left_hand_landmarks):
                    p1 = to_px(results.left_hand_landmarks[start_idx])
                    p2 = to_px(results.left_hand_landmarks[end_idx])
                    cv2.line(annotated_image, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)
            for lm in results.left_hand_landmarks:
                px = to_px(lm)
                cv2.circle(annotated_image, px, 4, (255, 144, 30), -1)

        # 4. Draw Right Hand (Magenta / Pink palette)
        if results.right_hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(results.right_hand_landmarks) and end_idx < len(results.right_hand_landmarks):
                    p1 = to_px(results.right_hand_landmarks[start_idx])
                    p2 = to_px(results.right_hand_landmarks[end_idx])
                    cv2.line(annotated_image, p1, p2, (255, 50, 180), 2, cv2.LINE_AA)
            for lm in results.right_hand_landmarks:
                px = to_px(lm)
                cv2.circle(annotated_image, px, 4, (180, 0, 255), -1)

        return annotated_image

    def release(self) -> None:
        """Releases underlying detector resources."""
        if self.use_tasks_api:
            if hasattr(self, "pose_detector") and self.pose_detector:
                self.pose_detector.close()
            if hasattr(self, "hand_detector") and self.hand_detector:
                self.hand_detector.close()
            if hasattr(self, "face_detector") and self.face_detector:
                self.face_detector.close()
        else:
            if hasattr(self, "legacy_holistic") and self.legacy_holistic:
                self.legacy_holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
