"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
Interactive Dataset Collection Engine
=============================================================================
This script provides an interactive OpenCV recorder to capture sign language
gestures, extract spatial landmarks using MediaPipe Holistic, buffer them into
30-frame temporal sequences (shape [30, 1662]), and save formatted NumPy (.npy)
arrays categorized into action folders for LSTM model training.

Directory Structure Created:
----------------------------
dataset/
├── hello/
│   ├── seq_000.npy   # Array shape: (30, 1662)
│   ├── seq_001.npy
│   └── ...
├── thank_you/
│   ├── seq_000.npy
│   └── ...
...

Controls:
---------
- SPACE: Pause / Resume recording
- S:     Skip to next sequence
- L:     Toggle dense face mesh overlay
- Q:     Quit dataset collection
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config import (
    DATA_PATH,
    ACTIONS,
    NUM_SEQUENCES,
    SEQUENCE_LENGTH,
    PREPARATION_DELAY_SECONDS,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TOTAL_KEYPOINTS_PER_FRAME,
)
from landmark_extractor import MediaPipeHolisticExtractor
from sequence_buffer import TemporalSequenceBuffer


def get_existing_sequence_count(action_dir: Path) -> int:
    """Returns the count of existing .npy sequence files in the action directory."""
    if not action_dir.exists():
        return 0
    return len(list(action_dir.glob("seq_*.npy")))


def draw_hud(
    frame: np.ndarray,
    action: str,
    action_idx: int,
    total_actions: int,
    seq_idx: int,
    total_seqs: int,
    frame_idx: int,
    total_frames: int,
    state: str,
    countdown: float = 0.0,
) -> np.ndarray:
    """
    Renders an interactive Heads-Up Display (HUD) overlay on the video feed.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Top banner background
    cv2.rectangle(overlay, (0, 0), (w, 90), (15, 15, 25), -1)
    # Bottom status bar background
    cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 15, 25), -1)

    # Blend overlay for glassmorphic effect
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Top Header Information
    action_text = f"Action [{action_idx + 1}/{total_actions}]: {action.upper()}"
    cv2.putText(
        frame,
        action_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 220),
        2,
        cv2.LINE_AA,
    )

    seq_text = f"Sequence: {seq_idx + 1}/{total_seqs}"
    cv2.putText(
        frame,
        seq_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # State Badge (RECORDING / PREPARING / PAUSED)
    if state == "RECORDING":
        badge_color = (50, 50, 255)  # Red
        badge_text = f"● REC [{frame_idx + 1}/{total_frames}]"
        # Progress Bar for current sequence
        progress_w = int((w - 360) * ((frame_idx + 1) / total_frames))
        cv2.rectangle(frame, (340, 52), (w - 20, 68), (50, 50, 50), -1)
        cv2.rectangle(frame, (340, 52), (340 + progress_w, 68), (0, 200, 255), -1)
    elif state == "PREPARING":
        badge_color = (0, 165, 255)  # Orange
        badge_text = f"GET READY: {countdown:.1f}s"
    else:  # PAUSED
        badge_color = (255, 200, 0)  # Cyan
        badge_text = "PAUSED"

    cv2.putText(
        frame,
        badge_text,
        (w - 320, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        badge_color,
        2,
        cv2.LINE_AA,
    )

    # Bottom Instructions Bar
    instructions = "[SPACE] Pause/Resume | [S] Skip Seq | [L] Toggle Mesh | [Q] Quit"
    cv2.putText(
        frame,
        instructions,
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Large Center Countdown during Preparation
    if state == "PREPARING":
        countdown_int = int(np.ceil(countdown))
        center_text = str(countdown_int if countdown_int > 0 else "GO!")
        text_size = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_DUPLEX, 4.0, 5)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Glow circle
        cv2.circle(frame, (w // 2, h // 2), 90, (0, 165, 255), 4)
        cv2.putText(
            frame,
            center_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            4.0,
            (0, 230, 255),
            5,
            cv2.LINE_AA,
        )

    return frame


def run_data_collection(
    actions: List[str],
    num_sequences: int,
    sequence_length: int,
    data_path: Path,
    camera_index: int,
    prep_delay: float,
) -> None:
    """
    Main loop executing camera capture, MediaPipe landmark extraction,
    and sequence persistence.
    """
    data_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" PROJECT VANI - SIGN LANGUAGE DATASET COLLECTION PIPELINE")
    print("=" * 70)
    print(f" Target Actions      : {actions}")
    print(f" Sequences / Action  : {num_sequences}")
    print(f" Frames / Sequence   : {sequence_length} (shape: [{sequence_length}, {TOTAL_KEYPOINTS_PER_FRAME}])")
    print(f" Destination Folder  : {data_path.resolve()}")
    print("=" * 70)

    # Initialize Video Capture
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[-] ERROR: Could not access video camera index {camera_index}.")
        sys.exit(1)

    extractor = MediaPipeHolisticExtractor()
    draw_mesh = False
    is_paused = False

    window_name = "Vani - Sign Language Data Collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    try:
        total_actions = len(actions)

        for action_idx, action in enumerate(actions):
            action_dir = data_path / action
            action_dir.mkdir(parents=True, exist_ok=True)

            existing_count = get_existing_sequence_count(action_dir)
            start_seq = existing_count

            if start_seq >= num_sequences:
                print(f"[+] Action '{action}' already has {existing_count} sequences. Skipping...")
                continue

            print(f"\n[*] Starting collection for action: '{action.upper()}' (Starting from seq {start_seq})")

            seq_idx = start_seq
            while seq_idx < num_sequences:
                # -------------------------------------------------------------
                # PHASE 1: PREPARATION COUNTDOWN
                # -------------------------------------------------------------
                prep_start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[-] Video feed interrupted.")
                        break

                    frame = cv2.flip(frame, 1)  # Mirror feed for natural user experience
                    elapsed_prep = time.time() - prep_start_time
                    remaining_prep = max(0.0, prep_delay - elapsed_prep)

                    # Extract landmarks for live feedback during prep
                    results = extractor.process_frame(frame)
                    annotated_frame = extractor.draw_styled_landmarks(
                        frame, results, draw_face_mesh=draw_mesh
                    )

                    hud_frame = draw_hud(
                        annotated_frame,
                        action=action,
                        action_idx=action_idx,
                        total_actions=total_actions,
                        seq_idx=seq_idx,
                        total_seqs=num_sequences,
                        frame_idx=0,
                        total_frames=sequence_length,
                        state="PAUSED" if is_paused else "PREPARING",
                        countdown=remaining_prep,
                    )

                    cv2.imshow("Vani - Sign Language Data Collection", hud_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n[!] User terminated data collection.")
                        return
                    elif key == ord(" "):
                        is_paused = not is_paused
                        if not is_paused:
                            prep_start_time = time.time()
                    elif key == ord("l"):
                        draw_mesh = not draw_mesh
                    elif key == ord("s"):
                        # Skip this sequence
                        break

                    if not is_paused and remaining_prep <= 0.0:
                        break

                # -------------------------------------------------------------
                # PHASE 2: RECORDING 30-FRAME TEMPORAL SEQUENCE
                # -------------------------------------------------------------
                sequence_frames = []
                frame_idx = 0

                while frame_idx < sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)

                    # 1. Process with MediaPipe Holistic
                    results = extractor.process_frame(frame)

                    # 2. Extract 1662-dimensional keypoint vector (with zero padding)
                    # Shape: (1662,), dtype: np.float32
                    keypoints = extractor.extract_keypoints(results)
                    sequence_frames.append(keypoints)

                    # 3. Draw styled landmarks & HUD
                    annotated_frame = extractor.draw_styled_landmarks(
                        frame, results, draw_face_mesh=draw_mesh
                    )

                    hud_frame = draw_hud(
                        annotated_frame,
                        action=action,
                        action_idx=action_idx,
                        total_actions=total_actions,
                        seq_idx=seq_idx,
                        total_seqs=num_sequences,
                        frame_idx=frame_idx,
                        total_frames=sequence_length,
                        state="RECORDING",
                    )

                    cv2.imshow("Vani - Sign Language Data Collection", hud_frame)
                    frame_idx += 1

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n[!] User terminated data collection.")
                        return
                    elif key == ord("l"):
                        draw_mesh = not draw_mesh

                # -------------------------------------------------------------
                # PHASE 3: EXPORT SEQUENCE AS NUMPY ARRAY (.npy)
                # -------------------------------------------------------------
                if len(sequence_frames) == sequence_length:
                    # Convert list of 30 vectors into matrix of shape (30, 1662)
                    sequence_array = np.array(sequence_frames, dtype=np.float32)

                    # Ensure shape invariant
                    assert sequence_array.shape == (sequence_length, TOTAL_KEYPOINTS_PER_FRAME), (
                        f"Invalid sequence matrix shape {sequence_array.shape}, "
                        f"expected ({sequence_length}, {TOTAL_KEYPOINTS_PER_FRAME})"
                    )

                    # File naming: seq_000.npy, seq_001.npy, ...
                    seq_filename = action_dir / f"seq_{seq_idx:03d}.npy"
                    np.save(str(seq_filename), sequence_array)
                    print(f"  [+] Saved {action}/seq_{seq_idx:03d}.npy -> Shape {sequence_array.shape}")
                    seq_idx += 1

        print("\n" + "=" * 70)
        print(" [OK] DATASET COLLECTION COMPLETE!")
        print(f" Dataset successfully saved to: {data_path.resolve()}")
        print("=" * 70)

    finally:
        cap.release()
        extractor.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Vani Sign Language Data Extraction & Collection Pipeline"
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=ACTIONS,
        help="List of gesture actions to record (e.g. hello thank_you yes no)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=NUM_SEQUENCES,
        help=f"Number of sequences to record per action (default: {NUM_SEQUENCES})",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=f"Number of frames per sequence (default: {SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"Root directory to store NumPy sequences (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"OpenCV video capture device index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--prep-delay",
        type=float,
        default=PREPARATION_DELAY_SECONDS,
        help=f"Preparation countdown delay in seconds (default: {PREPARATION_DELAY_SECONDS})",
    )

    args = parser.parse_args()

    run_data_collection(
        actions=args.actions,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        data_path=args.data_path,
        camera_index=args.camera,
        prep_delay=args.prep_delay,
    )


if __name__ == "__main__":
    main()
