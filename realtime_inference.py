"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
Real-Time Gesture Inference & Sentence Translation Engine
=============================================================================
This script connects the live webcam stream to the full Vani AI pipeline:
1. MediaPipe Holistic feature extractor -> extracts (1662,) coordinate vector
2. TemporalSequenceBuffer -> maintains (30, 1662) sliding window
3. Trained Keras LSTM Model -> classifies gesture probability
4. Real-time HUD -> renders predicted text, confidence bar, and sentence builder

Controls:
---------
- C: Clear translated sentence history
- L: Toggle face mesh overlay
- Q: Quit real-time recognition
=============================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# Suppress verbose TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SEQUENCE_LENGTH,
    TOTAL_KEYPOINTS_PER_FRAME,
)
from landmark_extractor import MediaPipeHolisticExtractor
from sequence_buffer import TemporalSequenceBuffer


def load_model_and_metadata(models_dir: Path) -> Tuple[tf.keras.Model, List[str]]:
    """Loads trained model and class labels."""
    keras_path = models_dir / "vani_gesture_model.keras"
    h5_path = models_dir / "vani_gesture_model.h5"
    meta_path = models_dir / "actions.json"

    if keras_path.exists():
        model_file = keras_path
    elif h5_path.exists():
        model_file = h5_path
    else:
        raise FileNotFoundError(
            f"No trained model found in '{models_dir}'. Please run 'train_model.py' first."
        )

    print(f"[*] Loading model from {model_file.resolve()}...")
    model = tf.keras.models.load_model(str(model_file))

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            actions = metadata.get("actions", [])
    else:
        # Fallback default
        actions = ["hello", "thank_you", "yes", "no", "help", "please", "i_love_you"]

    print(f"[+] Loaded model successfully with {len(actions)} classes: {actions}")
    return model, actions


def draw_inference_hud(
    frame: np.ndarray,
    predicted_action: str,
    confidence: float,
    sentence: List[str],
    fps: float,
    buffer_progress: float,
    threshold: float = 0.70,
) -> np.ndarray:
    """
    Renders real-time HUD with prediction badges, probability bars, and sentence history.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Glassmorphic top banner
    cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 25), -1)
    # Glassmorphic bottom banner for sentence history
    cv2.rectangle(overlay, (0, h - 70), (w, h), (15, 15, 25), -1)

    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 1. Top Section: Prediction & Confidence
    is_confident = confidence >= threshold and predicted_action != "..."
    status_color = (0, 255, 180) if is_confident else (150, 150, 150)

    # Gesture label display
    cv2.putText(
        frame,
        "GESTURE:",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        predicted_action.upper(),
        (130, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Confidence Bar
    bar_x = 25
    bar_y = 65
    bar_w = 280
    bar_h = 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), status_color, -1)

    conf_text = f"{confidence * 100:.1f}%"
    cv2.putText(
        frame,
        conf_text,
        (bar_x + bar_w + 15, bar_y + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Top Right Stats (FPS & Buffer filling progress)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 150, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )

    buf_text = "BUFFER READY" if buffer_progress >= 1.0 else f"BUFFER: {int(buffer_progress * 100)}%"
    cv2.putText(
        frame,
        buf_text,
        (w - 200, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 120) if buffer_progress >= 1.0 else (0, 180, 255),
        1,
        cv2.LINE_AA,
    )

    # 2. Bottom Section: Translated Sentence History
    sentence_str = " ".join(sentence[-6:]) if sentence else "(Perform a sign gesture to translate)"
    cv2.putText(
        frame,
        "TRANSLATION:",
        (25, h - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        sentence_str,
        (160, h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Instructions subtitle
    cv2.putText(
        frame,
        "[C] Clear Sentence | [L] Toggle Mesh | [Q] Quit",
        (25, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_realtime_inference(
    models_dir: Path = Path("models"),
    camera_index: int = CAMERA_INDEX,
    threshold: float = 0.75,
) -> None:
    """
    Main loop executing camera capture, feature extraction, sequence buffer sliding,
    LSTM classification, and HUD visualization.
    """
    model, actions = load_model_and_metadata(models_dir)

    print("\n" + "=" * 70)
    print(" PROJECT VANI - REAL-TIME GESTURE RECOGNITION & TRANSLATION")
    print("=" * 70)
    print(f" Gesture Classes : {actions}")
    print(f" Confidence Thresh: {threshold * 100:.0f}%")
    print("=" * 70)

    # Initialize OpenCV Camera
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[-] ERROR: Could not open camera {camera_index}")
        return

    extractor = MediaPipeHolisticExtractor()
    buffer = TemporalSequenceBuffer(sequence_length=SEQUENCE_LENGTH, feature_dim=TOTAL_KEYPOINTS_PER_FRAME)

    sentence = []
    current_prediction = "..."
    current_confidence = 0.0
    last_prediction_time = 0.0
    draw_mesh = False

    window_name = "Vani - Real-Time Sign Language Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)

            # 1. Extract holistic spatial landmarks
            results = extractor.process_frame(frame)
            keypoints = extractor.extract_keypoints(results)

            # 2. Append to temporal sequence buffer (30 frames)
            buffer.append(keypoints)

            # 3. Perform LSTM Inference when buffer is full
            if buffer.is_ready():
                batch_sequence = buffer.get_batch_sequence()  # Shape: (1, 30, 1662)
                probabilities = model.predict(batch_sequence, verbose=0)[0]
                best_class_idx = int(np.argmax(probabilities))
                current_confidence = float(probabilities[best_class_idx])
                current_prediction = actions[best_class_idx]

                # Update translated sentence builder with debounce
                now = time.time()
                if current_confidence >= threshold and (now - last_prediction_time) > 1.5:
                    if not sentence or sentence[-1] != current_prediction:
                        sentence.append(current_prediction)
                        last_prediction_time = now
            else:
                current_prediction = "Buffering..."
                current_confidence = 0.0

            # 4. Render skeleton & HUD
            annotated_frame = extractor.draw_styled_landmarks(
                frame, results, draw_face_mesh=draw_mesh
            )

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            buffer_progress = len(buffer) / SEQUENCE_LENGTH
            hud_frame = draw_inference_hud(
                annotated_frame,
                predicted_action=current_prediction,
                confidence=current_confidence,
                sentence=sentence,
                fps=fps,
                buffer_progress=buffer_progress,
                threshold=threshold,
            )

            cv2.imshow(window_name, hud_frame)

            # Keybindings
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[!] Exiting real-time inference.")
                break
            elif key == ord("c"):
                sentence.clear()
            elif key == ord("l"):
                draw_mesh = not draw_mesh

    finally:
        cap.release()
        extractor.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Vani Real-Time Sign Language Inference & Translation"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model weights and actions.json",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera device index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for sentence translation (default: 0.75)",
    )

    args = parser.parse_args()

    run_realtime_inference(
        models_dir=args.models_dir,
        camera_index=args.camera,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
