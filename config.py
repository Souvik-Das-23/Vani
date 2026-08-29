"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
Configuration Module
=============================================================================
This module centralizes pipeline hyperparameters, directory paths, camera
specifications, and landmark dimension calculations.
"""

from pathlib import Path

# =============================================================================
# DATASET & STORAGE CONFIGURATION
# =============================================================================
# Root directory where action-categorized NumPy sequences are saved
DATA_PATH = Path("dataset")

# Default sign language actions / classes to record and recognize
ACTIONS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "help",
    "please",
    "i_love_you"
]

# Number of gesture sequences to record per action during dataset collection
NUM_SEQUENCES = 30

# =============================================================================
# TEMPORAL & SEQUENCE CONFIGURATION
# =============================================================================
# Number of consecutive frames captured per sequence (approx. 1 second @ 30 FPS)
SEQUENCE_LENGTH = 30

# Delay (in seconds) given to the user to get into position before a sequence records
PREPARATION_DELAY_SECONDS = 2.0

# =============================================================================
# HARDWARE & CAMERA CONFIGURATION
# =============================================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# =============================================================================
# MEDIAPIPE HOLISTIC HYPERPARAMETERS
# =============================================================================
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1  # 0: Lite, 1: Full, 2: Heavy

# =============================================================================
# KEYPOINT DIMENSION BREAKDOWN (Spatial Feature Geometry)
# =============================================================================
# Pose: 33 landmarks × (x, y, z, visibility) = 132 features
POSE_LANDMARKS_COUNT = 33
POSE_FEATURES_PER_LANDMARK = 4
POSE_FEATURES_TOTAL = POSE_LANDMARKS_COUNT * POSE_FEATURES_PER_LANDMARK  # 132

# Face: 468 landmarks × (x, y, z) = 1404 features
FACE_LANDMARKS_COUNT = 468
FACE_FEATURES_PER_LANDMARK = 3
FACE_FEATURES_TOTAL = FACE_LANDMARKS_COUNT * FACE_FEATURES_PER_LANDMARK  # 1404

# Left Hand: 21 landmarks × (x, y, z) = 63 features
HAND_LANDMARKS_COUNT = 21
HAND_FEATURES_PER_LANDMARK = 3
HAND_FEATURES_TOTAL = HAND_LANDMARKS_COUNT * HAND_FEATURES_PER_LANDMARK  # 63

# Right Hand: 21 landmarks × (x, y, z) = 63 features
# Total feature vector dimension per frame: 132 + 1404 + 63 + 63 = 1662 keypoints
TOTAL_KEYPOINTS_PER_FRAME = (
    POSE_FEATURES_TOTAL
    + FACE_FEATURES_TOTAL
    + HAND_FEATURES_TOTAL
    + HAND_FEATURES_TOTAL
)  # 1662

# Sequence tensor shape: (SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME) -> (30, 1662)
SEQUENCE_SHAPE = (SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME)
