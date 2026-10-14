# Air Guitar CS249R — Dataset Schema Specification

**Version:** 2.0 (Software-Only MediaPipe Architecture)  
**Location:** `dataset/`

---

## 1. Overview

The software-only ML Air Guitar pipeline replaces analog flex sensors and physical IMUs with vision-based 3D hand landmarks extracted from standard webcam video frames using **MediaPipe Hands**.

The dataset directory is organized as follows:
```
dataset/
├── raw/
│   ├── chord_landmarks.csv
│   ├── strumming_frames.csv
│   └── sequences/
├── processed/
│   ├── chord_features.csv
│   ├── strumming_sequences.npz
│   └── metadata.json
└── smoke_test/
    ├── smoke_chord_features.csv
    ├── smoke_strumming_sequences.npz
    └── smoke_metadata.json
```

---

## 2. MediaPipe Hand Landmark Landmark Definition

MediaPipe Hands outputs 21 3D landmarks for each detected hand:
- `0`: WRIST
- `1-4`: THUMB (CMC, MCP, IP, TIP)
- `5-8`: INDEX FINGER (MCP, PIP, DIP, TIP)
- `9-12`: MIDDLE FINGER (MCP, PIP, DIP, TIP)
- `13-16`: RING FINGER (MCP, PIP, DIP, TIP)
- `17-20`: PINKY (MCP, PIP, DIP, TIP)

Coordinates:
- `x`: Horizontal screen coordinate, normalized to $[0.0, 1.0]$.
- `y`: Vertical screen coordinate, normalized to $[0.0, 1.0]$ (0 at top, 1 at bottom).
- `z`: Relative depth roughly proportional to palm scale (depth relative to wrist).

---

## 3. Raw Datasets Schema

### 3.1 Raw Chord Landmarks (`dataset/raw/chord_landmarks.csv`)

Stores static hand landmark frames captured during chord gesture collection.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `timestamp` | Float | POSIX epoch timestamp (seconds) when frame was captured |
| `session_id` | String | Unique identifier of collection session (e.g. `sess_20261014_091522`) |
| `sample_id` | Integer | Incremental counter within the session |
| `handedness` | String | `Left` or `Right` (as detected by MediaPipe) |
| `label` | String | Target chord label (`a`, `am`, `b`, `bm`, `c`, `cm`, `d`, `dm`, `e`, `em`, `f`, `g`, `gm`, or `neutral`) |
| `confidence` | Float | Hand landmark tracking confidence score ($0.0 - 1.0$) |
| `x_0` to `z_20` | Float (63 cols) | Raw $(x, y, z)$ coordinates for all 21 landmarks |

### 3.2 Raw Strumming Frames (`dataset/raw/strumming_frames.csv`)

Stores continuous temporal sequence frames captured during right-hand strumming motions.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `timestamp` | Float | POSIX epoch timestamp (seconds) |
| `session_id` | String | Collection session ID |
| `sequence_id` | Integer | Identifier grouping contiguous gesture sequences |
| `frame_idx` | Integer | Frame index within the current sequence ($0, 1, 2, \dots$) |
| `handedness` | String | `Right` or `Left` |
| `label` | String | Motion class: `downstroke`, `upstroke`, or `idle` |
| `x_0` to `z_20` | Float (63 cols) | Raw $(x, y, z)$ coordinates for all 21 landmarks |

---

## 4. Processed Datasets Schema

### 4.1 Processed Chord Features (`dataset/processed/chord_features.csv`)

Contains normalized, invariant geometric representations designed for invariant gesture classification.

#### Normalization Strategy:
1. **Translation Invariance:** The wrist coordinate $(x_0, y_0, z_0)$ is subtracted from all 21 landmark positions, centering the wrist at $(0, 0, 0)$.
2. **Scale Invariance:** The distance between the wrist ($L_0$) and the middle MCP knuckle ($L_9$) is computed as the palm scale reference:
   $$s = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2 + (z_9 - z_0)^2}$$
   All centered coordinates are divided by $s$.
3. **Geometric Angles:** Joint flexion angles computed via dot products for all fingers (PIP, DIP, MCP joints).
4. **Normalized Distances:** Euclidean distances between each fingertip ($L_4, L_8, L_{12}, L_{16}, L_{20}$) and:
   - The wrist ($L_0$)
   - The corresponding knuckle ($L_1, L_5, L_9, L_{13}, L_{17}$)
   - Adjacent fingertips (to detect spread vs pinch)

#### Column Structure:
- `session_id` (String): Grouping key for non-leaking train/val/test splits.
- `norm_x_0` to `norm_z_20` (63 floats): Centered and scale-normalized coordinates.
- `dist_thumb_tip_wrist`, `dist_index_tip_wrist`, etc. (Floats).
- `angle_index_pip`, `angle_middle_pip`, etc. (Floats in radians).
- `label` (String): Ground truth chord name.
- `label_encoded` (Integer): Integer index corresponding to `label_classes.json`.

### 4.2 Processed Strumming Sequences (`dataset/processed/strumming_sequences.npz`)

Used for training the temporal LSTM classifier.

- **`X`** (NumPy array, `float32`, shape: `(N, 15, F)`):
  - Fixed sequence length of $T = 15$ frames (~0.5 seconds at 30 FPS).
  - Feature dimension $F = 18$, including:
    - Normalized wrist position ($x, y, z$)
    - First-order displacement ($\Delta x, \Delta y, \Delta z$) relative to previous frame
    - Instantaneous velocity ($v_x, v_y, v_z$)
    - Index knuckle position & displacement
    - Vertical motion indicator ($\Delta y$)
- **`y`** (NumPy array, `int32`, shape: `(N,)`):
  - `0`: `idle`
  - `1`: `downstroke`
  - `2`: `upstroke`
- **`session_ids`** (NumPy array, string, shape: `(N,)`):
  - Ensures sequence windows never cross recording or session boundaries.
- **`class_names`** (NumPy array, string): `['idle', 'downstroke', 'upstroke']`.

### 4.3 Metadata Artifact (`dataset/processed/metadata.json`)

Stores preprocessing parameters, feature column order, label encoding dictionary, and dataset generation timestamp to guarantee identical feature extraction during real-time webcam inference.

---

## 5. Non-Leakage Splitting Rule

**Strict Requirement:**
Random train/test splits across individual frames introduce substantial data leakage because adjacent video frames from the same gesture recording are highly correlated.
All train/test/validation partitioning **must** be grouped by `session_id` using `GroupKFold` or `GroupShuffleSplit` from `scikit-learn`.
