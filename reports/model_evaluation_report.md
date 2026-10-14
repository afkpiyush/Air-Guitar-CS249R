# Air Guitar CS249R — Model Evaluation Report

**Generated:** 2026-10-14 05:59:18

## Model Summary

- **Left Hand Chord Model:** Keras MLP (chord_classifier.keras)
- **Right Hand Strum Model:** Keras Bidirectional LSTM (strumming_lstm.keras)
- **Inference Latency:** 314.97 ms per frame
- **Projected Throughput:** 3.2 FPS (CPU)
- **Audio Assets:** 13/13 verified

## Deployment Readiness Checklist

- [x] Hardware dependencies completely removed (no microcontrollers, gloves, or sensors)
- [x] Feature extraction matches training schema exactly (98 chord features, 18 strum features)
- [x] Debounce cooldown active (250 ms)
- [x] Ready to launch: `python src/air_guitar.py`
