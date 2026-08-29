"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
Pipeline Unit & Integration Verification Suite
=============================================================================
Validates:
1. MediaPipe Holistic initialization and extraction on synthetic image frames.
2. Zero-padding logic ensuring strict (1662,) 1D feature vector shape.
3. TemporalSequenceBuffer sliding window queue and (30, 1662) batch shape.
4. Dataset filesystem serialization and deserialization integrity (.npy).
=============================================================================
"""

import os
import sys
import shutil
import tempfile
import numpy as np
from pathlib import Path

# Ensure UTF-8 output if supported
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import (
    TOTAL_KEYPOINTS_PER_FRAME,
    SEQUENCE_LENGTH,
    POSE_FEATURES_TOTAL,
    FACE_FEATURES_TOTAL,
    HAND_FEATURES_TOTAL,
)
from landmark_extractor import MediaPipeHolisticExtractor
from sequence_buffer import TemporalSequenceBuffer


def test_landmark_extractor_shapes():
    print("[*] Testing MediaPipe Holistic Landmark Extractor...")
    extractor = MediaPipeHolisticExtractor()

    # 1. Test with synthetic blank frame (simulates no landmarks detected)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = extractor.process_frame(blank_frame)
    keypoints = extractor.extract_keypoints(results)

    assert isinstance(keypoints, np.ndarray), "Keypoints must be a numpy ndarray"
    assert keypoints.dtype == np.float32, f"Expected float32, got {keypoints.dtype}"
    assert keypoints.shape == (TOTAL_KEYPOINTS_PER_FRAME,), (
        f"Expected shape ({TOTAL_KEYPOINTS_PER_FRAME},), got {keypoints.shape}"
    )

    # Verify zero-padding works when no human is detected
    assert np.all(keypoints == 0.0), "Synthetic blank image should produce all zeros"
    print(f"  [+] Zero-padded fallback passed: Shape {keypoints.shape} with all zeros")

    # 2. Test styled drawing utility
    annotated = extractor.draw_styled_landmarks(blank_frame, results)
    assert annotated.shape == blank_frame.shape, "Annotated frame shape mismatch"
    print("  [+] Styled drawing utility passed")

    extractor.release()


def test_sequence_buffer():
    print("\n[*] Testing Temporal Sequence Buffer...")
    seq_len = 30
    buffer = TemporalSequenceBuffer(sequence_length=seq_len, feature_dim=TOTAL_KEYPOINTS_PER_FRAME)

    assert not buffer.is_ready(), "Buffer should not be ready initially"
    assert len(buffer) == 0, "Buffer length should be 0 initially"
    assert buffer.get_sequence() is None, "get_sequence() should return None when buffer is incomplete"

    # Test padded retrieval on incomplete buffer
    padded_seq = buffer.get_sequence(pad_if_incomplete=True)
    assert padded_seq.shape == (seq_len, TOTAL_KEYPOINTS_PER_FRAME)
    print(f"  [+] Incomplete buffer zero-padding passed: Shape {padded_seq.shape}")

    # Push 30 frames
    for i in range(seq_len):
        dummy_frame = np.full((TOTAL_KEYPOINTS_PER_FRAME,), fill_value=i, dtype=np.float32)
        buffer.append(dummy_frame)

    assert buffer.is_ready(), "Buffer should be ready after 30 frames"
    assert len(buffer) == seq_len, f"Buffer length should be {seq_len}"

    seq = buffer.get_sequence()
    assert seq.shape == (seq_len, TOTAL_KEYPOINTS_PER_FRAME), f"Expected ({seq_len}, {TOTAL_KEYPOINTS_PER_FRAME}), got {seq.shape}"
    assert seq.dtype == np.float32

    # Check temporal ordering: first frame value should be 0, last frame value should be 29
    assert np.all(seq[0] == 0.0), "First frame in sequence should have value 0.0"
    assert np.all(seq[-1] == 29.0), "Last frame in sequence should have value 29.0"

    # Test batch sequence shape for LSTM model ingestion
    batch_seq = buffer.get_batch_sequence()
    assert batch_seq.shape == (1, seq_len, TOTAL_KEYPOINTS_PER_FRAME), (
        f"Expected batch shape (1, {seq_len}, {TOTAL_KEYPOINTS_PER_FRAME}), got {batch_seq.shape}"
    )
    print(f"  [+] Sequence buffer sliding window & batch shape passed: {batch_seq.shape}")

    # Push one more frame and verify sliding window advances
    new_frame = np.full((TOTAL_KEYPOINTS_PER_FRAME,), fill_value=30, dtype=np.float32)
    buffer.append(new_frame)
    adv_seq = buffer.get_sequence()
    assert np.all(adv_seq[0] == 1.0), "Sliding window should have dropped frame 0"
    assert np.all(adv_seq[-1] == 30.0), "Sliding window should have frame 30 at end"
    print("  [+] Sliding window FIFO behavior verified")

    buffer.reset()
    assert len(buffer) == 0, "Buffer reset failed"
    print("  [+] Buffer reset passed")


def test_npy_export_and_load():
    print("\n[*] Testing NumPy (.npy) Dataset Export and Load...")
    temp_dir = Path(tempfile.mkdtemp(prefix="vani_test_"))
    try:
        action_name = "hello"
        action_dir = temp_dir / action_name
        action_dir.mkdir(parents=True, exist_ok=True)

        # Create mock 30-frame sequence
        mock_sequence = np.random.randn(SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME).astype(np.float32)
        save_path = action_dir / "seq_000.npy"
        np.save(str(save_path), mock_sequence)

        assert save_path.exists(), f"File {save_path} was not created"

        # Load back and verify integrity
        loaded_array = np.load(str(save_path))
        assert loaded_array.shape == (SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME), (
            f"Loaded shape mismatch: {loaded_array.shape}"
        )
        assert loaded_array.dtype == np.float32, f"Loaded dtype mismatch: {loaded_array.dtype}"
        assert np.allclose(mock_sequence, loaded_array), "Data corruption during save/load"

        print(f"  [+] Successfully exported and verified: {save_path.name} -> Shape {loaded_array.shape}")
    finally:
        shutil.rmtree(temp_dir)
        print("  [+] Temporary test artifacts cleaned up")


def main():
    print("=" * 70)
    print(" RUNNING PROJECT VANI PIPELINE VERIFICATION SUITE")
    print("=" * 70)
    test_landmark_extractor_shapes()
    test_sequence_buffer()
    test_npy_export_and_load()
    print("\n" + "=" * 70)
    print(" [OK] ALL PIPELINE UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
