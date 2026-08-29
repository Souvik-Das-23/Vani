# Vani (वाणी) - Real-Time Sign Language Recognition & Translation System

<p align="center">
  <img src="./assets/vani_preview.png" alt="Vani AI Interface Preview" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/MediaPipe-Holistic%20%2F%20Tasks-00C4B4?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 🌟 Overview

**Vani** is an end-to-end, bi-directional sign language recognition and translation platform designed to empower seamless communication for the deaf and hard-of-hearing community. 

This repository contains:
1. **Task 1: Spatial-Temporal Feature Extraction Pipeline**: Captures webcam streams, extracts 543 multi-modal landmarks (Pose, Face, Hands), pads missing keypoints to shape `(1662,)`, and windows them into 30-frame temporal blocks `(30, 1662)`.
2. **Task 2: Deep Temporal LSTM Gesture Model**: Trains a multi-layer LSTM neural network on sequential landmark matrices, saves best weights via callbacks (`.keras` & `.h5`), and provides real-time webcam inference with dynamic sentence translation.

---

## 📐 Matrix Dimensions & Geometric Breakdown

Each video frame is processed through **MediaPipe Holistic** to extract 3D Cartesian coordinates and visibility metrics across 4 body modalities:

| Landmark Modality | Landmark Count | Values per Landmark | Feature Vector Shape | Zero-Padding Fallback | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pose** | 33 | 4 $(x, y, z, \text{visibility})$ | `(132,)` | `np.zeros(132)` | Torso, shoulders, elbows, wrists |
| **Face Mesh** | 468 | 3 $(x, y, z)$ | `(1404,)` | `np.zeros(1404)` | Facial expressions & lip contours |
| **Left Hand** | 21 | 3 $(x, y, z)$ | `(63,)` | `np.zeros(63)` | Left finger joints & palm |
| **Right Hand** | 21 | 3 $(x, y, z)$ | `(63,)` | `np.zeros(63)` | Right finger joints & palm |
| **Total Frame Vector** | **543** | — | **`(1662,)`** | `np.zeros(1662)` | Concatenated 1D `float32` array |

### ⏱️ Temporal Sliding Window
- **Sequence Length**: 30 consecutive frames ($\approx 1$ second of continuous motion at 30 FPS).
- **Sequence Tensor Shape**: **`(30, 1662)`**
- **Batch Shape for LSTM Input**: **`(Batch_Size, 30, 1662)`**

---

## 🧠 Model Architecture (TensorFlow / Keras)

```
=============================================================================
Layer (type)                     Output Shape          Param #   Description
=============================================================================
sequence_input (InputLayer)      (None, 30, 1662)      0         30 temporal frames × 1662 keypoints
lstm_layer_1 (LSTM)              (None, 30, 64)        442,112   return_sequences=True
dropout_1 (Dropout)              (None, 30, 64)        0         p = 0.2
lstm_layer_2 (LSTM)              (None, 30, 128)       98,816    return_sequences=True
dropout_2 (Dropout)              (None, 30, 128)       0         p = 0.2
lstm_layer_3 (LSTM)              (None, 64)            49,408    return_sequences=False
dense_features_1 (Dense)         (None, 64)            4,160     ReLU activation
dropout_3 (Dropout)              (None, 64)            0         p = 0.2
dense_features_2 (Dense)         (None, 32)            2,080     ReLU activation
gesture_probabilities (Dense)    (None, num_classes)   231       Softmax classification
=============================================================================
Total params: 596,807 (2.28 MB)
Trainable params: 596,807 (2.28 MB)
```

---

## 📁 Repository Structure

```
vani/
├── assets/
│   └── vani_preview.png              # Interface screenshot & preview banner
├── models/
│   ├── vani_gesture_model.keras      # Trained model weights (Keras 3 native format)
│   ├── vani_gesture_model.h5         # Trained model weights (H5 format for Django)
│   ├── actions.json                  # Class index-to-label metadata mapping
│   └── logs/                         # TensorBoard training metrics
├── requirements.txt                  # Python dependencies
├── config.py                         # Pipeline constants, actions, tensor shapes, camera setup
├── landmark_extractor.py             # Dual-mode MediaPipe extractor (Tasks Vision API + Solutions)
├── sequence_buffer.py                # Circular temporal sliding window buffer (30, 1662)
├── collect_data.py                   # Interactive dataset recorder CLI with real-time HUD
├── train_model.py                    # LSTM model construction, callbacks, training & evaluation
├── realtime_inference.py             # Live webcam gesture inference & sentence builder
├── test_pipeline.py                  # Automated unit and integration test suite
└── dataset/                          # Categorized dataset directory (.npy sequences)
    ├── hello/
    │   ├── seq_000.npy               # Shape: (30, 1662)
    │   └── ...
    ├── thank_you/
    └── ...
```

---

## 🚀 Quick Start

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/Souvik-Das-23/Vani.git
cd Vani

# Optional: Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Record Custom Gestures (Optional)

```bash
python collect_data.py
```

*Shortcuts: `[SPACE]` Pause/Resume, `[S]` Skip, `[L]` Face mesh, `[Q]` Exit.*

### 4. Train the LSTM Sequence Model

Train the model on your recorded dataset (or generate synthetic validation data with `--generate-mock-data`):

```bash
python train_model.py --epochs 100 --batch-size 16
```

#### Training Callbacks Configured:
- **`ModelCheckpoint`**: Saves best weights to `models/best_vani_gesture_model.keras`.
- **`EarlyStopping`**: Restores best weights when validation loss stops improving (patience: 25).
- **`ReduceLROnPlateau`**: Dynamically scales learning rate on loss plateaus.
- **`TensorBoard`**: Logs scalar curves and weight distributions in `models/logs/`.

### 5. Run Real-Time Webcam Translation

Launch the real-time inference loop to test your trained model with live camera feed:

```bash
python realtime_inference.py
```

*Shortcuts: `[C]` Clear translated sentence history, `[L]` Toggle face mesh, `[Q]` Quit.*

---

## 🛣️ Project Roadmap

- [x] **Task 1: Data Extraction & Preprocessing Pipeline** (MediaPipe Holistic + 30-Frame Sequence Buffer)
- [x] **Task 2: Temporal Model Training & Real-Time Inference** (TensorFlow / Keras LSTM)
- [ ] **Task 3: Backend API** (Django REST Framework for stream classification & inference)
- [ ] **Task 4: Cross-Platform Mobile App** (Flutter & Dart with real-time camera overlay)
- [ ] **Task 5: 3D Avatar Rendering** (Three.js text-to-sign reverse translation)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
