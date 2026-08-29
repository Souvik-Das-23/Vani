"""
=============================================================================
Project Vani: API Views & Inference Endpoints
=============================================================================
Provides REST endpoints for:
1. POST /api/translate/ -> Evaluates 30-frame sequence & returns predicted sign
2. GET  /api/health/    -> Server & model readiness status
3. GET  /api/actions/   -> List of supported sign language gesture actions
=============================================================================
"""

import time
import numpy as np
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from api.apps import ApiConfig
from api.serializers import SequencePredictionSerializer


@api_view(["POST"])
@permission_classes([AllowAny])
def translate_gesture(request):
    """
    Inference Endpoint: Classifies a 30-frame spatial coordinate sequence.

    Expected JSON Payload:
    {
        "sequence": [[... 1662 floats ...], ... 30 frames ...]
    }

    Response (200 OK):
    {
        "status": "success",
        "action": "hello",
        "confidence": 0.9842,
        "probabilities": {
            "hello": 0.9842,
            "thank_you": 0.0051,
            ...
        },
        "latency_ms": 12.4
    }
    """
    start_time = time.time()

    # 1. Validate JSON payload structure and matrix shapes
    serializer = SequencePredictionSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {
                "status": "error",
                "message": "Invalid input matrix dimensions or malformed payload.",
                "errors": serializer.errors,
                "expected_shape": f"({ApiConfig.sequence_length}, {ApiConfig.feature_dimension})",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # 2. Check if the pre-trained LSTM model is loaded in memory
    if not ApiConfig.is_model_loaded or ApiConfig.model is None:
        # Attempt fallback reload if model was trained after server started
        ApiConfig._load_trained_model()
        if ApiConfig.model is None:
            return Response(
                {
                    "status": "error",
                    "message": "Model not loaded in server memory. Please run 'train_model.py' to generate model weights.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    try:
        # 3. Extract and convert coordinate sequence to NumPy array
        raw_sequence = serializer.validated_data["sequence"]
        sequence_array = np.array(raw_sequence, dtype=np.float32)

        # 4. Expand dimensions for batch inference: (1, 30, 1662)
        batch_input = np.expand_dims(sequence_array, axis=0)

        # 5. Run Model Inference
        probabilities = ApiConfig.model.predict(batch_input, verbose=0)[0]

        # 6. Extract predicted class and confidence
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        predicted_action = (
            ApiConfig.actions[best_idx]
            if best_idx < len(ApiConfig.actions)
            else f"class_{best_idx}"
        )

        # 7. Build probability map across all classes
        prob_dict = {
            action: round(float(prob), 4)
            for action, prob in zip(ApiConfig.actions, probabilities)
        }

        latency = round((time.time() - start_time) * 1000, 2)

        return Response(
            {
                "status": "success",
                "action": predicted_action,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "sequence_length": len(raw_sequence),
                "latency_ms": latency,
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {
                "status": "error",
                "message": f"Inference execution failed: {str(e)}",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    """
    Health Check Endpoint: Returns server readiness, model status, and metadata.
    """
    return Response(
        {
            "status": "healthy",
            "service": "Vani Sign Language Recognition Backend API",
            "version": "1.0.0",
            "model_loaded": ApiConfig.is_model_loaded,
            "actions_count": len(ApiConfig.actions),
            "actions": ApiConfig.actions,
            "input_shape": {
                "sequence_length": ApiConfig.sequence_length,
                "feature_dimension": ApiConfig.feature_dimension,
            },
        },
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def list_actions(request):
    """
    Returns the complete list of supported sign language actions.
    """
    return Response(
        {
            "status": "success",
            "total_classes": len(ApiConfig.actions),
            "actions": ApiConfig.actions,
        },
        status=status.HTTP_200_OK,
    )
