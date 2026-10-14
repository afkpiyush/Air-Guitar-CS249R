# Guitar Audio Assets

**Location:** `sounds/chord_recordings/`

This directory houses the audio assets used by the software-only Air Guitar application for low-latency chord playback via Pygame.

---

## 1. Supported Chords & Audio Files

The application natively supports 13 acoustic guitar chord recordings:

| Chord Label | File Name | Description |
| :--- | :--- | :--- |
| `a` | `a.wav` | A Major |
| `am` | `am.wav` | A Minor |
| `b` | `b.wav` | B Major |
| `bm` | `bm.wav` | B Minor |
| `c` | `c.wav` | C Major |
| `cm` | `cm.wav` | C Minor |
| `d` | `d.wav` | D Major |
| `dm` | `dm.wav` | D Minor |
| `e` | `e.wav` | E Major |
| `em` | `em.wav` | E Minor |
| `f` | `f.wav` | F Major |
| `g` | `g.wav` | G Major |
| `gm` | `gm.wav` | G Minor |

All files are standard uncompressed PCM WAV files sampled at 44.1 kHz / 16-bit stereo.

---

## 2. Licensing and Attribution

The chord recordings in this directory are preserved from the original Air-Guitar-CS249R repository (CS249R Machine Learning Course, Harvard University), provided under the repository's MIT / Apache 2.0 open-source educational terms.
They were recorded cleanly on acoustic guitar specifically for polyphonic trigger playback.

---

## 3. Audio File Verification

To verify that all required audio assets are present and can be loaded by the Pygame mixer:
```bash
python -c "from src.audio_engine import AudioEngine; ae = AudioEngine(); print(ae.verify_assets())"
```

If an audio file is missing or corrupted, the audio engine will report a clear warning and gracefully avoid crashing during runtime.
