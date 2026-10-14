# Air-Guitar-CS249R Repository Audit

**Date:** 2026-10-14  
**Author:** Senior Machine Learning & Computer Vision Engineer  
**Project:** Transition from Microcontroller-Sensor Air Guitar to Pure Software ML Air Guitar

---

## 1. Executive Summary

The original repository **Air-Guitar-CS249R** was developed as a tinyML course project at Harvard (CS249R). It relied entirely on custom physical hardware:
1. **Left Hand (Chord Selection):** A glove fitted with 4 analog flex sensors (index, middle, ring, pinky fingers) wired to ADC pins on a Teensy 4.1 microcontroller.
2. **Right Hand (Strumming Detection):** A glove fitted with an MPU/IMU 3-axis accelerometer (`ax, ay, az`) wired to the Teensy board.
3. **Firmware & Deployment:** Platform.IO C++ projects compiling quantized TensorFlow Lite Micro models into static C header files (`model.h`), running onboard the microcontroller, communicating inference triggers via USB Serial (COM port, 9600 baud) to a host Python script (`deployment.py`) that invoked `playsound`.

This audit documents the transition of this project into a **100% software-based, webcam-driven, machine learning Air Guitar system**. The entire hardware layer (microcontrollers, gloves, flex sensors, accelerometers, serial communication) is decommissioned and replaced by computer vision hand tracking via **MediaPipe Hands**, normalized geometric & temporal feature engineering, TensorFlow/Keras neural network classifiers (MLP for chords and LSTM for temporal strumming), and low-latency audio synthesis using **Pygame**.

---

## 2. Inventory of Existing Repository Assets

| Path | Description | Hardware Dependency | Usability for Software System |
| :--- | :--- | :--- | :--- |
| `README.md` | Hardware schematic, glove photos, 13 chord gestures (binary bitmask) | Teensy 4.1, flex gloves, IMU | Reusable for conceptual chord definitions and music logic; superseded by new README |
| `audio_files/chord_recordings/` | 13 `.wav` audio files (`a.wav` through `gm.wav`) | None | **100% Reusable**. High quality clean chord recordings |
| `audio_files/test_files/` | Sample audio sweeps and backing track | None | Reusable for testing and audio verification |
| `data/flex/` | 13 `.csv` files containing 4-channel ADC flex sensor readings | Flex sensor hardware | **Cannot train webcam model**. Preserved in legacy archive |
| `data/imu/` | `strum.csv`, `no_strum.csv` with 3-axis accelerometer readings | IMU sensor hardware | **Cannot train webcam model**. Preserved in legacy archive |
| `data_collection/` | Python serial scripts (`flex_data_collection.py`, `imu_data_collection.py`) and Platform.IO projects | Teensy 4.1, Arduino serial | **Obsolete**. Replaced by interactive webcam notebook `01_data_collection.ipynb` |
| `deployment/` | Platform.IO Teensy project and `deployment.py` listening on COM7 | Teensy 4.1, USB Serial, `playsound` | **Obsolete**. Replaced by `src/air_guitar.py` and `src/audio_engine.py` |
| `experimentation/` | Arduino test sketches for audio amp, sweep, flex test, IMU test, SD test | Teensy / Arduino hardware | **Obsolete**. Preserved for historical reference |
| `notebooks/` | 4 Jupyter notebooks (`flex_fc_...`, `flex_cnn_...`, `strum_fc_...`, `strum_cnn_...`) | Flex & IMU CSV inputs, tinyML C header export | Replaced by 5 new modular JupyterLab notebooks in `notebooks/` (legacy moved to `notebooks/legacy/`) |
| `results/` | Saved TFLite models, confusion matrix PNGs, and C++ header files (`model.h`) | Teensy TFLite Micro | Model architecture insights can be referenced; weights incompatible with vision landmarks |

---

## 3. Hardware Dependencies Identified & Removed

### Hardware Components Identified:
1. **Teensy 4.1 Microcontroller:** Required Platform.IO toolchains, USB serial port configuration (`COM7`, 9600 baud rate).
2. **Flex Sensors (Spectra Symbol):** 4 resistive bend sensors measuring finger flexion via voltage divider ADC circuits.
3. **IMU Accelerometer (MPU-6050 / LSM9DS1):** 3-axis acceleration (`ax, ay, az`) at ~119 Hz.
4. **Sensor Gloves & Breadboards:** Physical wiring, pull-down resistors, solder breadboards.
5. **Serial Host Communication:** Python `serial.Serial` polling loop in `deployment/deployment.py` and data collection scripts.
6. **TinyML Microcontroller Deployment:** `xxd` hex dumps converting `.tflite` to C array headers (`model.h`).

### Software Equivalents Implemented:
1. **Input Sensor:** Standard laptop/desktop RGB webcam via OpenCV (`cv2.VideoCapture`).
2. **Hand Tracking:** MediaPipe Hands extracting 21 3D hand landmarks per hand at 30+ FPS.
3. **Left Hand (Chord Selection):** 
   - Translation-invariant, scale-normalized 3D coordinates relative to wrist.
   - 20 pairwise finger-to-palm and inter-joint Euclidean distances.
   - 15 inter-joint flexion and knuckle planar angles.
   - Classification via Deep MLP neural network (with confidence thresholding and neutral class).
4. **Right Hand (Strumming Motion):**
   - Temporal sequence window (e.g. 15-frame rolling window).
   - Wrist and knuckle displacement ($\Delta x, \Delta y, \Delta z$) and velocity vectors.
   - Temporal classification via Bidirectional LSTM / GRU network distinguishing `downstroke`, `upstroke`, and `idle`.
   - Debouncing state machine with cooldown timer to prevent repeat audio re-triggers.
5. **Audio Engine:** Non-blocking asynchronous audio playback using `pygame.mixer` with dedicated volume and mute controls.

---

## 4. Dataset Compatibility Analysis

The original datasets in `data/flex/*.csv` and `data/imu/*.csv` **cannot** be used to train the software webcam model:
- **Dimension mismatch:** Flex data has 4 scalar features (`index, middle, ring, pinky`), while a single hand in MediaPipe provides 21 landmarks $\times$ 3 coordinates ($X, Y, Z = 63$ values) plus derived geometric features (angles, distances).
- **Physical sensor characteristics:** Flex sensors measure physical resistance change dependent on glove tension and curvature, whereas MediaPipe provides camera-space spatial coordinates subject to perspective and scale.
- **Action space mismatch:** The legacy IMU dataset captured only binary `strum` vs `no_strum` without distinguishing directional strumming (`downstroke` vs `upstroke`).

**Solution:**
A dedicated, reproducible data acquisition and preprocessing pipeline is created under `dataset/raw/` and `dataset/processed/`.
An automated synthetic smoke-test fixture is provided under `dataset/smoke_test/` so that tests, model architectures, and inference loops can be verified immediately without requiring physical webcam presence.

---

## 5. Proposed Replacement Software Architecture

```
                 +--------------------------------+
                 |          Webcam Feed           |
                 +---------------+----------------+
                                 |
                                 v
                 +---------------+----------------+
                 |    MediaPipe Hands Module      |
                 |  (21 landmarks per detected hand)
                 +---------------+----------------+
                                 |
               +-----------------+-----------------+
               |                                   |
               v (Left Hand / Chords)              v (Right Hand / Strumming)
   +---------------------------+       +---------------------------+
   |   Spatial Normalization   |       |    Temporal Sequencing    |
   | Wrist origin, scale norm  |       | 15-frame sliding window   |
   | Angles & Finger Distances |       | Wrist displacement & vel  |
   +-------------+-------------+       +-------------+-------------+
                 |                                   |
                 v                                   v
   +-------------+-------------+       +-------------+-------------+
   |   Keras MLP Chord Model   |       |   Keras LSTM Strum Model  |
   |  Dense(128) -> Dense(64)  |       | LSTM(64) -> Dense(32)     |
   | Softmax over 13 chords    |       | Softmax: down/up/idle     |
   +-------------+-------------+       +-------------+-------------+
                 |                                   |
                 v                                   v
          [Predicted Chord]                   [Strum Trigger]
          Confidence >= 0.70                  State machine debounce
                 \                                   /
                  \                                 /
                   v                               v
                 +-----------------------------------+
                 |        Pygame Audio Engine        |
                 | Non-blocking low-latency playback |
                 | sounds/chord_recordings/<chord>.wav
                 +-----------------------------------+
```

---

## 6. Execution and Training Order

Users and automated pipelines should execute the components in the following strict order:

1. **Environment Setup:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Notebook 01 — Data Collection (`notebooks/01_data_collection.ipynb`):**
   Run to collect real webcam samples for left-hand chords and right-hand strumming sessions. Data saved incrementally to `dataset/raw/`.
3. **Notebook 02 — Data Cleaning and Preprocessing (`notebooks/02_preprocessing.ipynb`):**
   Validates landmarks, normalizes coordinates, extracts angle/distance features, generates temporal strumming sequences, enforces session-based train/val/test splits, and saves artifacts to `dataset/processed/`.
4. **Notebook 03 — Chord Classification (`notebooks/03_chord_classification.ipynb`):**
   Trains baseline models and Keras MLP on chord features. Evaluates with confusion matrix and saves model to `models/chord_classifier/`.
5. **Notebook 04 — Temporal Strumming Recognition (`notebooks/04_strumming_lstm.ipynb`):**
   Trains baseline and Keras LSTM on temporal strumming sequences. Evaluates per-class metrics and saves model to `models/strumming_lstm/`.
6. **Notebook 05 — Model Evaluation and Export (`notebooks/05_evaluation_and_export.ipynb`):**
   Benchmarking, latency measurement, export verification, configuration generation, and summary report creation in `reports/`.
7. **Real-Time Air Guitar Application:**
   ```bash
   python src/air_guitar.py
   ```
