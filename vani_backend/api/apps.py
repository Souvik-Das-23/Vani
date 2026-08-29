"""
=============================================================================
Project Vani: Django AppConfig & Model Singleton Loader
=============================================================================
This module initializes the API app and loads the trained TensorFlow / Keras
LSTM model (vani_gesture_model.h5 / vani_gesture_model.keras) and action label
mappings into memory ONCE during server startup.
=============================================================================
"""

import os
import sys
import json
from pathlib import Path
from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "Vani Sign Language Recognition API"

    # Static Singleton Attributes (Persisted in server memory)
    model = None
    actions = []
    sequence_length = 30
    feature_dimension = 1662
    is_model_loaded = False

    def ready(self) -> None:
        """
        Lifecycle hook invoked once when the Django process finishes initializing.
        Loads the pre-trained model into memory to ensure O(1) request-time latency.
        """
        # Avoid redundant loading during certain manage.py commands (e.g. makemigrations)
        if any(cmd in sys.argv for cmd in ["makemigrations", "migrate", "collectstatic"]):
            return

        self._load_trained_model()

    @classmethod
    def _load_trained_model(cls) -> None:
        """Loads the Keras / H5 model and label metadata from disk into memory."""
        try:
            import tensorflow as tf
            # Suppress internal TensorFlow C++ logs
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

            models_dir = getattr(settings, "MODELS_DIR", Path(__file__).resolve().parent.parent.parent / "models")
            keras_path = models_dir / "vani_gesture_model.keras"
            h5_path = models_dir / "vani_gesture_model.h5"
            actions_path = models_dir / "actions.json"

            # 1. Load Action Labels Metadata
            if actions_path.exists():
                with open(actions_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    cls.actions = metadata.get("actions", [])
                    cls.sequence_length = metadata.get("sequence_length", 30)
                    cls.feature_dimension = metadata.get("feature_dimension", 1662)
            else:
                cls.actions = ["hello", "thank_you", "yes", "no", "help", "please", "i_love_you"]

            # 2. Load Model Weights
            model_file = None
            if keras_path.exists():
                model_file = keras_path
            elif h5_path.exists():
                model_file = h5_path

            if model_file and model_file.exists():
                print(f"\n[*] [ApiConfig] Loading Vani LSTM model into server memory from: {model_file}...")
                cls.model = tf.keras.models.load_model(str(model_file))
                cls.is_model_loaded = True
                print(f"[+] [ApiConfig] Model loaded successfully! Registered classes ({len(cls.actions)}): {cls.actions}\n")
            else:
                print(f"[-] [ApiConfig] Warning: Model file not found in {models_dir}. Endpoints will operate in mock mode until model is trained.")

        except Exception as e:
            print(f"[-] [ApiConfig] Error loading model during startup: {e}")
            cls.is_model_loaded = False
