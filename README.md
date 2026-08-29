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

This repository contains **Task 1: The Core Data Extraction and Sequence Windowing Engine**, responsible for capturing real-time video streams, tracking multi-modal spatial landmarks via Google MediaPipe, applying zero-padding to missing keypoints, buffering into temporal 30-frame sliding windows, and exporting structured NumPy datasets (`.npy`) ready for LSTM model training.

---

## 📐 Matrix Dimensions & Geometric Breakdown

Each video frame is processed through **MediaPipe Holistic (Tasks Vision API)** to extract 3D Cartesian coordinates and visibility metrics across 4 body modalities:

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

If a hand or body part is occluded or outside the camera frame, the extractor automatically pads the corresponding slice with zeros to guarantee deterministic tensor dimensions.

---

## 📁 Repository Structure

```
vani/
├── assets/
│   └── vani_preview.png       # Interface screenshot & preview banner
├── requirements.txt           # Python dependencies
├── config.py                  # Pipeline constants, actions, tensor shapes, camera setup
├── landmark_extractor.py      # Dual-mode MediaPipe extractor (Tasks Vision API + Solutions)
├── sequence_buffer.py         # Circular temporal sliding window buffer (30, 1662)
├── collect_data.py            # Interactive dataset recorder CLI with real-time HUD
├── test_pipeline.py           # Automated unit and integration test suite
└── dataset/                   # Categorized dataset directory (created upon recording)
    ├── hello/
    │   ├── seq_000.npy        # Shape: (30, 1662)
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

### 3. Run Pipeline Verification Tests

Verify MediaPipe initialization, zero-padding, sliding window buffering, and array serialization:

```bash
python test_pipeline.py
```

---

## 🎥 Interactive Dataset Collection

Launch the OpenCV GUI recorder with live Heads-Up Display (HUD) and preparation countdown:

```bash
python collect_data.py
```

### Custom Arguments:
```bash
# Record specific actions with custom sequence count
python collect_data.py --actions hello thank_you yes no help --num-sequences 40

# Custom sequence length and preparation delay
python collect_data.py --sequence-length 30 --prep-delay 3.0 --camera 0
```

### 🎮 Keyboard Shortcuts During Recording:
- `[SPACE]` : **Pause / Resume** recording.
- `[S]` : **Skip** to the next sequence.
- `[L]` : **Toggle** fine-grain facial mesh points.
- `[Q]` : **Exit** and safely save recorded sequences.

---

## 🛣️ Project Roadmap

- [x] **Task 1: Data Extraction & Preprocessing Pipeline** (MediaPipe Holistic + 30-Frame Sequence Buffer)
- [ ] **Task 2: Temporal Model Training** (TensorFlow / Keras LSTM / Bi-LSTM Architecture)
- [ ] **Task 3: Backend API** (Django REST Framework for stream classification & inference)
- [ ] **Task 4: Cross-Platform Mobile App** (Flutter & Dart with real-time camera overlay)
- [ ] **Task 5: 3D Avatar Rendering** (Three.js text-to-sign reverse translation)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
