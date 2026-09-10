"""
=============================================================================
Project Vani: Django API Automated Test Suite
=============================================================================
Tests:
1. Health Check Endpoint (GET /api/health/)
2. Supported Actions Endpoint (GET /api/actions/)
3. Gesture Translation with Valid Input (POST /api/translate/)
4. Gesture Translation with Malformed / Invalid Input (400 Bad Request)
=============================================================================
"""

from unittest.mock import patch

import numpy as np
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status

from api.apps import ApiConfig


class VaniApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.translate_url = reverse("translate_gesture")
        self.health_url = reverse("health_check")
        self.actions_url = reverse("list_actions")

        # Ensure model is initialized
        if not ApiConfig.is_model_loaded:
            ApiConfig._load_trained_model()

    def test_health_check_endpoint(self):
        """Validates that /api/health/ returns 200 OK with service metadata."""
        response = self.client.get(self.health_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "healthy")
        self.assertIn("actions", response.data)
        self.assertIn("model_loaded", response.data)

    def test_list_actions_endpoint(self):
        """Validates that /api/actions/ returns 200 OK with action list."""
        response = self.client.get(self.actions_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "success")
        self.assertIsInstance(response.data["actions"], list)
        self.assertGreater(len(response.data["actions"]), 0)

    def test_translate_with_valid_sequence(self):
        """Validates that /api/translate/ accepts a (30, 1662) matrix and returns predicted action."""
        # Create valid mock 30-frame sequence
        valid_matrix = np.random.randn(30, 1662).tolist()
        payload = {"sequence": valid_matrix}

        response = self.client.post(self.translate_url, payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "success")
        self.assertIn("action", response.data)
        self.assertIn("confidence", response.data)
        self.assertIn("probabilities", response.data)
        self.assertIsInstance(response.data["confidence"], float)

    def test_translate_with_invalid_temporal_length(self):
        """Validates that sequences with != 30 frames are rejected with 400 Bad Request."""
        invalid_temporal_matrix = np.random.randn(15, 1662).tolist()  # 15 frames instead of 30
        payload = {"sequence": invalid_temporal_matrix}

        response = self.client.post(self.translate_url, payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["status"], "error")

    def test_translate_with_invalid_feature_dimension(self):
        """Validates that frames with != 1662 keypoints are rejected with 400 Bad Request."""
        invalid_feature_matrix = np.random.randn(30, 50).tolist()  # 50 features instead of 1662
        payload = {"sequence": invalid_feature_matrix}

        response = self.client.post(self.translate_url, payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["status"], "error")

    def test_translate_with_empty_payload(self):
        """Validates that empty payloads return 400 Bad Request."""
        response = self.client.post(self.translate_url, {}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_translate_exception_does_not_leak_internal_details(self):
        """
        Validates that if model inference raises an exception, /api/translate/
        returns a generic 500 error without exposing the raw exception message
        (file paths, library internals, etc.) to the client.
        """
        class _RaisingModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError("internal failure: /srv/models/secret_path.keras corrupted")

        valid_matrix = np.random.randn(30, 1662).tolist()
        payload = {"sequence": valid_matrix}

        with patch.object(ApiConfig, "model", _RaisingModel()), \
             patch.object(ApiConfig, "is_model_loaded", True):
            response = self.client.post(self.translate_url, payload, format="json")

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(response.data["status"], "error")
        self.assertNotIn("secret_path", response.data["message"])
        self.assertNotIn("internal failure", response.data["message"])
