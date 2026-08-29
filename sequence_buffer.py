"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
Temporal Sequence Window Buffer Module
=============================================================================
This module provides a high-performance circular sequence window buffer
to maintain a temporal sliding window of N consecutive landmark frames.

Matrix Dimensions:
------------------
- Input frame vector: Shape (1662,) -> np.float32
- Buffered sequence:  Shape (SEQUENCE_LENGTH, 1662) = (30, 1662) -> np.float32
- Batch representation for LSTM input: (1, 30, 1662)
=============================================================================
"""

import collections
import numpy as np
from typing import Optional

from config import SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME


class TemporalSequenceBuffer:
    """
    Circular sliding window buffer for temporal sequence aggregation.
    Maintains a FIFO queue of landmark vectors for real-time inference and training.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        feature_dim: int = TOTAL_KEYPOINTS_PER_FRAME,
    ) -> None:
        """
        Initializes the temporal sequence buffer.

        Args:
            sequence_length: Number of consecutive frames to hold (default: 30).
            feature_dim: Dimension of the flattened landmark vector per frame (default: 1662).
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.buffer = collections.deque(maxlen=sequence_length)

    def append(self, keypoints: np.ndarray) -> None:
        """
        Appends a new frame of keypoints to the sliding window buffer.

        Args:
            keypoints: 1D feature vector of shape (feature_dim,) -> (1662,)
        """
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.asarray(keypoints, dtype=np.float32)

        if keypoints.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected keypoints shape ({self.feature_dim},), "
                f"but received {keypoints.shape}"
            )

        self.buffer.append(keypoints.astype(np.float32))

    def is_ready(self) -> bool:
        """
        Checks if the buffer has accumulated a complete window of frames.

        Returns:
            bool: True if buffer holds exactly `sequence_length` frames.
        """
        return len(self.buffer) == self.sequence_length

    def get_sequence(self, pad_if_incomplete: bool = False) -> Optional[np.ndarray]:
        """
        Retrieves the current temporal sequence window as a 2D NumPy array.

        Args:
            pad_if_incomplete: If True and buffer is not full, prepends zero-frames
                               to return a complete (sequence_length, feature_dim) array.

        Returns:
            np.ndarray of shape (sequence_length, feature_dim) -> (30, 1662),
            or None if buffer is not full and pad_if_incomplete is False.
        """
        if not self.is_ready():
            if not pad_if_incomplete:
                return None
            
            # Left-pad with zeros if requested
            current_len = len(self.buffer)
            missing = self.sequence_length - current_len
            pad_frames = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(missing)]
            full_list = pad_frames + list(self.buffer)
            return np.array(full_list, dtype=np.float32)

        return np.array(self.buffer, dtype=np.float32)

    def get_batch_sequence(self) -> Optional[np.ndarray]:
        """
        Convenience method to retrieve sequence with an added batch dimension
        ready for direct model ingestion.

        Returns:
            np.ndarray of shape (1, sequence_length, feature_dim) -> (1, 30, 1662),
            or None if buffer is not full.
        """
        seq = self.get_sequence()
        if seq is None:
            return None
        return np.expand_dims(seq, axis=0)

    def reset(self) -> None:
        """Clears all accumulated frames from the buffer."""
        self.buffer.clear()

    @property
    def current_length(self) -> int:
        """Returns the number of frames currently in the buffer."""
        return len(self.buffer)

    def __len__(self) -> int:
        return self.current_length
