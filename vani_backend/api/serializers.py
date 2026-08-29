"""
=============================================================================
Project Vani: API Request Serializers & Validation Schemas
=============================================================================
Validates incoming time-series landmark coordinate arrays from Flutter / Web clients.
=============================================================================
"""

from rest_framework import serializers
from api.apps import ApiConfig


class SequencePredictionSerializer(serializers.Serializer):
    """
    Validates that the incoming request contains a valid (30, 1662) sequence matrix.
    """
    sequence = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField(),
            allow_empty=False,
        ),
        allow_empty=False,
        help_text="Nested 2D array of landmark coordinates with shape [30 frames, 1662 keypoints]",
    )

    def validate_sequence(self, value):
        expected_frames = ApiConfig.sequence_length or 30
        expected_features = ApiConfig.feature_dimension or 1662

        # 1. Validate temporal length
        if len(value) != expected_frames:
            raise serializers.ValidationError(
                f"Invalid temporal sequence length: Received {len(value)} frames, expected exactly {expected_frames}."
            )

        # 2. Validate feature dimension for each frame
        for frame_idx, frame in enumerate(value):
            if len(frame) != expected_features:
                raise serializers.ValidationError(
                    f"Invalid keypoint dimension at frame {frame_idx}: Received {len(frame)} values, expected exactly {expected_features}."
                )

        return value
