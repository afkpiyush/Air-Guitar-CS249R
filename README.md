# Air Guitar CS249R (Software-Only ML Air Guitar System)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe%20Hands-orange.svg)](https://developers.google.com/mediapipe)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%20%2F%20Keras-red.svg)](https://tensorflow.org/)
[![Pygame](https://img.shields.io/badge/Audio-Pygame%20Mixer-green.svg)](https://www.pygame.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)]()

A 100% software-based, webcam-controlled machine learning Air Guitar system. This project replaces physical sensor gloves, Teensy microcontrollers, and IMU hardware with computer vision hand tracking via **MediaPipe Hands**, invariant geometric and temporal feature engineering, custom **TensorFlow/Keras** neural networks, and low-latency acoustic audio playback via **Pygame**.

---

## 1. Project Overview & Architecture

### High-Level Concept
The system tracks both of the player's hands using a standard laptop or desktop RGB webcam:
1. **Left Hand (Chord Selection):** Static hand configuration is detected, normalized, and classified into one of 13 supported guitar chords using a Deep Multi-Layer Perceptron (MLP) with confidence thresholding and temporal smoothing.
2. **Right Hand (Strumming Recognition):** Continuous hand motion is tracked across consecutive frames in a 15-frame sliding window, extracting position, displacement, and velocity dynamics to classify strumming strokes (`downstroke`, `upstroke`, `idle`) using an LSTM network with cooldown debouncing.
3. **Audio Synthesis:** When a valid strum is detected, the audio engine plays the currently selected chord using pre-loaded high-fidelity acoustic guitar audio samples via Pygame.

> **Clarification:** MediaPipe is solely responsible for computer vision feature extraction (providing 21 3D spatial landmarks per hand). The custom trained machine learning models (MLP and LSTM) perform chord gesture classification and temporal strumming recognition.

```
                    +---------------------------+
                    |    Webcam Feed (30 FPS)   |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |      MediaPipe Hands      |
                    |  (21 landmarks per hand)  |
                    +-------------+-------------+
                                  |
                +-----------------+-----------------+
                |                                   |
                v (Left Hand / Chords)              v (Right Hand / Strumming)
    +-----------------------+           +-----------------------+
    | Coordinate & Scale    |           | Temporal Sliding      |
    | Normalization         |           | Window (15 frames)    |
    | (98 features)         |           | (18 features / frame) |
    +-----------+-----------+           +-----------+-----------+
                |                                   |
                v                                   v
    +-----------------------+           +-----------------------+
    | Keras Deep MLP        |           | Keras Temporal LSTM   |
    | Chord Classifier      |           | Strumming Classifier  |
    +-----------+-----------+           +-----------+-----------+
                |                                   |
                v                                   v
          Active Chord                       Strum Trigger
        (Confidence >= 0.60)               (Cooldown >= 250 ms)
                \                                   /
                 \                                 /
                  v                               v
    +-----------------------------------------------------------+
    |                    Pygame Audio Engine                    |
    |           Asynchronous Low-Latency Playback               |
    |              sounds/chord_recordings/<chord>.wav          |
    +-----------------------------------------------------------+
```

---

## 2. Supported Chords & Actions

### 2.1 Supported Chords (13 Classes)
The system supports 13 acoustic guitar chords (configurable in [src/config.py](file:///c:/Users/piyus/Desktop/ds%20cp%20airguitar/src/config.py)):
- **Major Chords:** `A`, `B`, `C`, `D`, `E`, `F`, `G`
- **Minor Chords:** `Am`, `Bm`, `Cm`, `Dm`, `Em`, `Gm`
- **Neutral / Uncertain:** When confidence is below threshold or hand is resting, chord playback is held in neutral to prevent false audio triggers.

### 2.2 Strumming Actions (3 Classes)
- `idle`: Hand hovering or resting.
- `downstroke`: Downward sweeping motion across strings.
- `upstroke`: Upward return stroke.

---

## 3. Repository Structure

```
Air-Guitar-CS249R/
├── notebooks/                       # End-to-end JupyterLab ML Pipeline
│   ├── 01_data_collection.ipynb     # Interactive webcam dataset collection
│   ├── 02_preprocessing.ipynb       # Feature engineering & session splitting
│   ├── 03_chord_classification.ipynb# MLP training, evaluation & metrics
│   ├── 04_strumming_lstm.ipynb      # LSTM temporal sequence training
│   ├── 05_evaluation_and_export.ipynb# Latency benchmarking & export
│   └── legacy/                      # Archived CS249R hardware notebooks
├── src/                             # Core Python Package
│   ├── air_guitar.py                # Real-time webcam air guitar application
│   ├── hand_tracking.py             # MediaPipe Hands wrapper & skeleton drawing
│   ├── feature_extraction.py        # Invariant geometric & velocity feature engineering
│   ├── inference.py                 # ChordPredictor & StrummingDetector debouncer
│   ├── audio_engine.py              # Pygame audio engine & asset verification
│   ├── config.py                    # Global constants, paths, and thresholds
│   └── generate_smoke_fixture.py    # Synthetic fixture & baseline model generator
├── dataset/
│   ├── raw/                         # Raw CSV landmark recordings
│   ├── processed/                   # Processed feature matrices and metadata
│   └── smoke_test/                  # Synthetic fixture for offline testing
├── models/
│   ├── chord_classifier/            # Keras chord model & label encoder
│   └── strumming_lstm/              # Keras LSTM strum model & config
├── sounds/
│   ├── chord_recordings/            # 13 high quality acoustic guitar WAV files
│   └── README.md                    # Audio licensing and asset documentation
├── reports/
│   └── model_evaluation_report.md   # Benchmarking and model metrics report
├── docs/
│   ├── REPOSITORY_AUDIT.md          # Complete audit of legacy hardware removal
│   ├── DATASET_SCHEMA.md            # Detailed schema and non-leakage rules
│   └── MODEL_TRAINING.md            # Model specifications and architectures
├── tests/                           # Unit tests and test suite
│   ├── test_features.py             # Normalization, angles, scale invariance
│   ├── test_preprocessing.py        # Schemas, session splitting, missing inputs
│   ├── test_inference.py            # Latency, smoothing, debouncing
│   └── test_audio.py                # Asset verification, volume, muting
├── requirements.txt                 # Clean Python dependencies
├── .gitignore                       # Ignored caches, videos, and checkpoints
└── README.md                        # Project documentation
```

---

## 4. Hardware-Free Installation & Setup

No microcontrollers, Teensy boards, gloves, flex sensors, or physical guitars are needed. A desktop or laptop with a built-in or USB webcam and speakers/headphones is all that is required.

### 4.1 Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit).
- Windows 10/11, macOS, or Linux.

### 4.2 Windows Setup Guide

1. **Open PowerShell or Command Prompt** and navigate to the project directory:
   ```powershell
   cd "c:\Users\piyus\Desktop\ds cp airguitar"
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify the Installation with the Test Suite:**
   ```powershell
   pytest -v tests/
   ```
   *(All 17 tests will run and pass using the built-in synthetic smoke fixture without requiring a webcam)*

---

## 5. End-to-End JupyterLab ML Pipeline

Launch JupyterLab:
```powershell
jupyter lab
```

Execute the five notebooks under `notebooks/` in the following sequence:

| Step | Notebook | Purpose |
| :---: | :--- | :--- |
| **1** | [01_data_collection.ipynb](file:///c:/Users/piyus/Desktop/ds%20cp%20airguitar/notebooks/01_data_collection.ipynb) | Open webcam, record Left hand static chord gestures and Right hand strumming sequences. Saves incrementally to `dataset/raw/`. |
| **2** | [02_preprocessing.ipynb](file:///c:/Users/piyus/Desktop/ds%20cp%20airguitar/notebooks/02_preprocessing.ipynb) | Performs wrist translation, palm-scale normalization, 98-feature engineering, sliding 15-frame sequence generation, and session-based split (zero data leakage). |
| **3** | [03_chord_classification.ipynb](file:///c:/Users/piyus/Desktop/ds%20cp%20airguitar/notebooks/03_chord_classification.ipynb) | Trains Random Forest baseline and Keras Deep MLP. Generates confusion matrix and loss/accuracy curves. Exports to `models/chord_classifier/`. |
| **4** | [04_strumming_lstm.ipynb](file:///c:/Users/piyus/Desktop/ds%20cp%20airguitar/notebooks/04_strumming_lstm.ipynb) | Visualizes motion dynamics ($v_y$) and trains Bidirectional LSTM for sequence recognition. Exports to `models/strumming_lstm/`. |
| **5** | [05_evaluation_and_export.ipynb](file:///c:/Users/piyus/Desktop/ds%20cp%20airguitar/notebooks/05_evaluation_and_export.ipynb) | Benchmarks inference latency on CPU, verifies audio asset mapping, and generates the final deployment report. |

---

## 6. Running the Real-Time Air Guitar Application

Once models are trained (or using the built-in verifiable smoke checkpoint), launch the live webcam application:

```powershell
python src/air_guitar.py
```

### Runtime Keyboard Shortcuts:
- **`q`** or **`ESC`**: Quit cleanly and release camera/audio resources.
- **`m`**: Toggle audio mute on/off.
- **`+` / `-`**: Increase or decrease guitar playback volume.
- **`s`**: Swap handedness (for left-handed guitarists).
- **`--camera <index>`**: Select external webcam (default: 0).
- **`--no-mirror`**: Disable selfie mirroring if using an audience-facing camera.

---

## 7. Troubleshooting & FAQs

- **"Could not open webcam at index 0"**:
  Make sure your webcam is not in use by another application (Zoom, Teams, Browser). Try `--camera 1`.
- **"Audio is not playing"**:
  Verify your speakers are unmuted. Press `m` in the application window to ensure the software engine is not muted. Run `python -c "from src.audio_engine import AudioEngine; print(AudioEngine().verify_assets())"` to confirm all 13 `.wav` files are loaded.
- **"Chord model not found"**:
  Run `notebooks/03_chord_classification.ipynb` to train and save the chord model, or run `python src/generate_smoke_fixture.py` to regenerate the baseline smoke checkpoint.
- **"Hand recognition is jittery"**:
  Ensure adequate room lighting and position both hands fully inside the camera frame. The application uses a 5-frame majority vote to smooth out minor landmark flickers.

---

## 8. License & Attribution

- MediaPipe Hands is licensed under Apache 2.0 (Google).
- Audio assets preserved from the Harvard CS249R Air Guitar project repository under educational open-source terms.
- Software implementation provided under the MIT License.
