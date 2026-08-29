"""
=============================================================================
Project Vani: Real-Time Sign Language Recognition System
LSTM Sequence Model Architecture, Training & Export Engine
=============================================================================
This module builds, compiles, trains, and exports a deep temporal LSTM neural
network for multi-class sign language gesture recognition using sequential
MediaPipe landmark matrices of shape [30, 1662].

Tensor Dimensions Across Architecture:
--------------------------------------
1. Input Layer:        (None, 30, 1662)  - 30 temporal frames, 1662 spatial features
2. LSTM Layer 1:       (None, 30, 64)    - return_sequences=True
3. Dropout (0.2):      (None, 30, 64)
4. LSTM Layer 2:       (None, 30, 128)   - return_sequences=True
5. Dropout (0.2):      (None, 30, 128)
6. LSTM Layer 3:       (None, 64)        - return_sequences=False
7. Dense Layer 1:      (None, 64)        - ReLU activation
8. Dropout (0.2):      (None, 64)
9. Dense Layer 2:      (None, 32)        - ReLU activation
10. Output Softmax:    (None, num_classes)- Probability distribution
=============================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Suppress verbose TensorFlow C++ logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.utils import to_categorical

from config import (
    DATA_PATH,
    ACTIONS,
    SEQUENCE_LENGTH,
    TOTAL_KEYPOINTS_PER_FRAME,
)


def load_dataset(
    data_path: Path,
    actions: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Dynamically loads all .npy sequence files from categorized action directories.

    Args:
        data_path: Root dataset directory path (e.g., Path("dataset")).
        actions: List of target action class names.

    Returns:
        X: Sequence feature tensor of shape (N_samples, 30, 1662)
        y: One-hot encoded target tensor of shape (N_samples, num_classes)
        label_map: Dictionary mapping class name to integer index {action: idx}
    """
    sequences = []
    labels = []
    label_map = {action: idx for idx, action in enumerate(actions)}

    print("\n" + "=" * 70)
    print(" [*] LOADING DATASET FROM DISK")
    print("=" * 70)
    print(f" Dataset Path        : {data_path.resolve()}")
    print(f" Target Classes ({len(actions)}): {actions}")

    valid_actions = []

    for action in actions:
        action_dir = data_path / action
        if not action_dir.exists():
            print(f"  [-] Warning: Directory '{action_dir}' not found. Skipping...")
            continue

        npy_files = sorted(list(action_dir.glob("*.npy")))
        if not npy_files:
            print(f"  [-] Warning: No .npy files found in '{action_dir}'. Skipping...")
            continue

        valid_actions.append(action)
        action_seq_count = 0

        for npy_file in npy_files:
            try:
                # Load sequence array of shape (30, 1662)
                res = np.load(str(npy_file))

                # Validate matrix dimensions
                if res.shape == (SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME):
                    sequences.append(res)
                    labels.append(label_map[action])
                    action_seq_count += 1
                else:
                    print(
                        f"  [-] Invalid shape {res.shape} in {npy_file.name}, "
                        f"expected ({SEQUENCE_LENGTH}, {TOTAL_KEYPOINTS_PER_FRAME})"
                    )
            except Exception as e:
                print(f"  [-] Error reading {npy_file}: {e}")

        print(f"  [+] Action '{action.upper():<12}' : Loaded {action_seq_count:>4} sequences")

    if not sequences:
        raise ValueError(
            f"No valid sequence files found in {data_path}. "
            "Please record sequences using 'collect_data.py' or run with '--generate-mock-data'."
        )

    # Convert to NumPy arrays
    X = np.array(sequences, dtype=np.float32)
    y_raw = np.array(labels, dtype=np.int32)

    # One-hot encode targets: shape (N, num_classes)
    y = to_categorical(y_raw, num_classes=len(actions)).astype(np.float32)

    print("-" * 70)
    print(f" [+] Total Sequences Loaded  : {X.shape[0]}")
    print(f" [+] Input Feature Tensor X  : Shape {X.shape} (dtype: {X.dtype})")
    print(f" [+] Output Target Tensor y  : Shape {y.shape} (dtype: {y.dtype})")
    print("=" * 70)

    return X, y, label_map


def generate_mock_dataset(
    data_path: Path,
    actions: List[str],
    samples_per_action: int = 30,
) -> None:
    """
    Generates synthetic landmark sequence files for pipeline validation & testing.
    """
    print(f"\n[*] Generating synthetic mock dataset for {len(actions)} actions...")
    data_path.mkdir(parents=True, exist_ok=True)

    for action_idx, action in enumerate(actions):
        action_dir = data_path / action
        action_dir.mkdir(parents=True, exist_ok=True)

        for seq_idx in range(samples_per_action):
            # Create synthetic temporal curve with class-specific signature
            t = np.linspace(0, np.pi * 2, SEQUENCE_LENGTH)[:, np.newaxis]
            base_pattern = np.sin(t + action_idx * 0.5) * 0.5 + 0.5
            noise = np.random.normal(0, 0.05, (SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME))
            mock_sequence = (base_pattern + noise).astype(np.float32)

            file_path = action_dir / f"seq_{seq_idx:03d}.npy"
            np.save(str(file_path), mock_sequence)

    print(f"[+] Successfully generated {len(actions) * samples_per_action} mock sequences in {data_path}.")


def build_lstm_model(
    input_shape: Tuple[int, int] = (SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME),
    num_classes: int = len(ACTIONS),
) -> Sequential:
    """
    Builds the Sequential Keras LSTM Architecture for Temporal Gesture Classification.

    Args:
        input_shape: Temporal input dimensions (30 frames, 1662 keypoints).
        num_classes: Number of discrete gesture action classes.

    Returns:
        Compiled Keras Sequential Model.
    """
    model = Sequential(
        [
            # 1. Explicit Input layer
            Input(shape=input_shape, name="sequence_input"),
            
            # 2. First LSTM layer (Extracts low-level temporal dynamics)
            # Input: (Batch, 30, 1662) -> Output: (Batch, 30, 64)
            LSTM(
                64,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                name="lstm_layer_1",
            ),
            Dropout(0.2, name="dropout_1"),
            
            # 3. Second LSTM layer (Extracts hierarchical temporal representations)
            # Input: (Batch, 30, 64) -> Output: (Batch, 30, 128)
            LSTM(
                128,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                name="lstm_layer_2",
            ),
            Dropout(0.2, name="dropout_2"),
            
            # 4. Third LSTM layer (Condenses sequence into final temporal state)
            # Input: (Batch, 30, 128) -> Output: (Batch, 64)
            LSTM(
                64,
                return_sequences=False,
                activation="tanh",
                recurrent_activation="sigmoid",
                name="lstm_layer_3",
            ),
            
            # 5. Fully Connected Representation Layers
            # Input: (Batch, 64) -> Output: (Batch, 64)
            Dense(64, activation="relu", name="dense_features_1"),
            Dropout(0.2, name="dropout_3"),
            
            # Input: (Batch, 64) -> Output: (Batch, 32)
            Dense(32, activation="relu", name="dense_features_2"),
            
            # 6. Classification Output Layer
            # Input: (Batch, 32) -> Output: (Batch, num_classes)
            Dense(num_classes, activation="softmax", name="gesture_probabilities"),
        ],
        name="Vani_LSTM_Gesture_Classifier",
    )

    return model


def train_pipeline(
    data_path: Path = DATA_PATH,
    actions: List[str] = ACTIONS,
    epochs: int = 150,
    batch_size: int = 16,
    test_size: float = 0.20,
    learning_rate: float = 0.001,
    models_dir: Path = Path("models"),
    generate_mock_data: bool = False,
) -> Sequential:
    """
    Executes the end-to-end training, evaluation, and model export pipeline.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = models_dir / "logs" / f"run_{int(time.time())}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Check if mock data generation requested or if data folder is empty
    if generate_mock_data or not any(data_path.glob("*/*.npy")):
        generate_mock_dataset(data_path, actions, samples_per_action=35)

    # 1. Load and pre-process dataset
    X, y, label_map = load_dataset(data_path, actions)

    # 2. Train-Test Split (80% Train, 20% Test with Stratification)
    # y_raw indices for stratification
    y_indices = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y_indices,
    )

    print(f"\n[*] Training Set Partition   : {X_train.shape[0]} samples (Shape: {X_train.shape})")
    print(f"[*] Testing Set Partition    : {X_test.shape[0]} samples (Shape: {X_test.shape})")

    # 3. Build & Compile Model
    model = build_lstm_model(
        input_shape=(SEQUENCE_LENGTH, TOTAL_KEYPOINTS_PER_FRAME),
        num_classes=len(actions),
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # Print model architecture summary
    print("\n" + "=" * 70)
    print(" VANI LSTM GESTURE MODEL SUMMARY")
    print("=" * 70)
    model.summary()
    print("=" * 70 + "\n")

    # 4. Configure Training Callbacks
    checkpoint_path = models_dir / "best_vani_gesture_model.keras"
    callbacks = [
        # TensorBoard for visualization
        TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
        # ModelCheckpoint to save optimal weights
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # EarlyStopping to avoid overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when plateauing
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Fit Model
    print(f"[*] Commencing Model Training for up to {epochs} Epochs...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Model Evaluation on Test Split
    print("\n" + "=" * 70)
    print(" [*] MODEL EVALUATION ON UNSEEN TEST PARTITION")
    print("=" * 70)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f" [+] Final Test Loss         : {test_loss:.4f}")
    print(f" [+] Final Test Accuracy     : {test_acc * 100:.2f}%")

    # Detailed Classification Report & Confusion Matrix
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n" + "-" * 70)
    print(" CLASSIFICATION REPORT:")
    print("-" * 70)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=actions,
            zero_division=0,
        )
    )

    # 7. Model & Metadata Export
    # Save standard .keras format
    keras_save_path = models_dir / "vani_gesture_model.keras"
    model.save(str(keras_save_path))
    print(f"[+] Saved Keras model to: {keras_save_path.resolve()}")

    # Save legacy .h5 format for backward compatibility
    h5_save_path = models_dir / "vani_gesture_model.h5"
    try:
        model.save(str(h5_save_path))
        print(f"[+] Saved H5 model to: {h5_save_path.resolve()}")
    except Exception as e:
        print(f"[-] Note on H5 export: {e}")

    # Save class label mapping metadata for Django & Mobile inference
    labels_metadata_path = models_dir / "actions.json"
    with open(labels_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "actions": actions,
                "label_map": label_map,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_dimension": TOTAL_KEYPOINTS_PER_FRAME,
                "test_accuracy": float(test_acc),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=4,
        )
    print(f"[+] Saved action metadata to: {labels_metadata_path.resolve()}")
    print("=" * 70)
    print(" [OK] TRAINING & EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Vani Sign Language LSTM Classification Model"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"Dataset root folder (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=ACTIONS,
        help="List of gesture classes to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum training epochs (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Adam optimizer learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--generate-mock-data",
        action="store_true",
        help="Generate synthetic dataset for testing if real dataset is empty",
    )

    args = parser.parse_args()

    train_pipeline(
        data_path=args.data_path,
        actions=args.actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        generate_mock_data=args.generate_mock_data,
    )


if __name__ == "__main__":
    main()
