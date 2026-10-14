# Air Guitar CS249R — Machine Learning Model Training Guide

**Version:** 2.0  
**Location:** `docs/MODEL_TRAINING.md`

---

## 1. Machine Learning Architecture Overview

The software-only Air Guitar employs two specialized ML models operating in real-time on synchronized webcam feeds:

```
                    +---------------------------+
                    |  Webcam Input (30 FPS)    |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |      MediaPipe Hands      |
                    +-------------+-------------+
                                  |
                +-----------------+-----------------+
                |                                   |
                v (Left Hand)                       v (Right Hand)
    +-----------------------+           +-----------------------+
    |   Chord Classifier    |           |    Strumming LSTM     |
    |  (Static Spatial MLP) |           |  (Temporal Sequence)  |
    +-----------------------+           +-----------------------+
```

---

## 2. Model 1: Chord Classifier (Left Hand)

### 2.1 Problem Formulation
Multi-class classification mapping normalized static hand landmark geometry into 13 discrete guitar chords + 1 optional neutral state:
- Chords: `a`, `am`, `b`, `bm`, `c`, `cm`, `d`, `dm`, `e`, `em`, `f`, `g`, `gm` (and optional `neutral`).
- Input dimension: 63 normalized coordinates + 20 distances + 15 joint angles = **98 engineered features**.

### 2.2 Baseline Model
- **Algorithm:** Random Forest Classifier (`n_estimators=100`) or Logistic Regression with standard scaling.
- Evaluated as benchmark against deep learning model.

### 2.3 Deep Learning Architecture (MLP)
Implemented in TensorFlow/Keras:
- **Input Layer:** Shape `(98,)`
- **Dense Layer 1:** 128 units, ReLU activation, BatchNormalization, Dropout(0.2)
- **Dense Layer 2:** 64 units, ReLU activation, Dropout(0.2)
- **Dense Layer 3:** 32 units, ReLU activation
- **Output Layer:** Dense with `softmax` activation over $K$ chord classes
- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam(learning_rate=1e-3)` with decay
- **Callbacks:** `EarlyStopping(patience=15, restore_best_weights=True)`, `ModelCheckpoint`

### 2.4 Artifacts Saved
Stored in `models/chord_classifier/`:
- `model.keras` (or `best_chord_model.keras`): Trained neural network weights and architecture.
- `label_encoder.joblib`: Scikit-learn LabelEncoder instance.
- `feature_schema.json`: Ordered list of expected input features and normalization parameters.

---

## 3. Model 2: Temporal Strumming Classifier (Right Hand)

### 3.1 Problem Formulation
Temporal sequence classification predicting right-hand strumming dynamics over a sliding window:
- Classes:
  - `0`: `idle` (hand stationary or hovering)
  - `1`: `downstroke` (rapid downward movement across guitar strings)
  - `2`: `upstroke` (upward movement returning from strum)
- Input shape: `(batch_size, 15, 18)` (15 frames $\approx$ 0.5 sec, 18 velocity/displacement features per frame).

### 3.2 Baseline Model
- Rule-based velocity thresholding and/or Flattened Gradient Boosting classifier.

### 3.3 Deep Learning Architecture (LSTM)
Implemented in TensorFlow/Keras:
- **Input Layer:** Shape `(15, 18)`
- **Bidirectional LSTM Layer 1:** 64 units, return_sequences=True, Dropout(0.2)
- **LSTM Layer 2:** 32 units, return_sequences=False, Dropout(0.2)
- **Dense Layer 1:** 32 units, ReLU activation
- **Output Layer:** Dense(3, activation='softmax')
- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam(learning_rate=1e-3)`
- **Callbacks:** `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5)`

### 3.4 Temporal Debouncing and Trigger Logic
In real-time inference, raw frame-by-frame predictions can produce multiple strum triggers during a single physical gesture.
The engine employs:
1. **Confidence Threshold:** Require $P(\text{class}) \ge 0.70$.
2. **State Machine:** Transition to active strum state only on `idle` $\to$ `downstroke` / `upstroke`.
3. **Cooldown Window:** Enforce a minimum interval (e.g. 250 ms) before the next strum sound can trigger.

### 3.5 Artifacts Saved
Stored in `models/strumming_lstm/`:
- `strumming_lstm.keras`: Trained sequence network.
- `strum_config.json`: Sliding window length, feature list, cooldown parameters.

---

## 4. Model Training and Notebook Pipeline

Run the notebooks sequentially in JupyterLab:
1. `notebooks/01_data_collection.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_chord_classification.ipynb`
4. `notebooks/04_strumming_lstm.ipynb`
5. `notebooks/05_evaluation_and_export.ipynb`
