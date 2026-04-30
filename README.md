# PiFocus: Context-Aware Focus Risk Estimation

## Overview

PiFocus is an edge-based system that estimates user focus by combining visual attention signals from a Raspberry Pi camera with task context information.

Instead of directly predicting "focus" from facial signals, PiFocus separates the problem into:

- **Visual Attention**: Is the user behaviorally engaged (e.g., present, facing screen)?
- **Task Relevance**: Is the user's activity related to their intended task?

The system then fuses these signals over time to estimate **focus risk**, rather than making fragile binary judgments.

---

## Motivation

Many existing approaches attempt to infer focus directly from camera data.

However, we observe a key limitation:

> A user can appear visually focused (e.g., looking at the screen) while actually being off-task (e.g., watching videos).

This reveals a fundamental gap between:
- **Visual engagement**
- **True task-related focus**

PiFocus addresses this by explicitly modeling both.

---

## System Architecture
            +----------------------+
            | Raspberry Pi Camera |
            +----------+----------+
                       |
                       v
          Visual Attention Model
                       |
                       v
            visual_attention_score

            +----------------------+
            | Laptop Activity Log |
            +----------+----------+
                       |
                       v
           Task Relevance Estimator
                       |
                       v
            task_relevance_score

                       |
                       v
             Fusion & Risk Engine
                       |
                       v
               Focus Risk State

---

## Key Components

### 1. Visual Attention (Edge - Raspberry Pi)

- Face detection
- Face stability (movement, centeredness)
- Presence (face ratio)
- Derived attention score

Model:
- Logistic Regression
- Random Forest

---

### 2. Task Context (Laptop - Consent-Based)

- Active application / window title
- Idle time
- User-defined task mode (e.g., writing, coding, lecture)

Example:

| App | Task Mode | Relevance |
|-----|----------|----------|
| VSCode | coding | on-task |
| YouTube | writing | off-task |
| PDF reader | reading | on-task |

---

### 3. Fusion Engine

Instead of binary classification:

The system outputs:

- `focused_on_task`
- `present_but_off_task`
- `absent`
- `uncertain`

---

## Results

We trained classifiers on collected camera data.

### Confusion Matrix (Random Forest)

- High accuracy (~96%)
- BUT: model relies heavily on visual stability

### Key Finding

> The model captures physical presence and posture, but cannot reliably distinguish true task focus.

This validates the need for task context integration.

---

## Demo

The demo includes:

- Real camera-based attention signals
- Simulated or real task context
- Real-time focus risk estimation
- Temporal smoothing and risk accumulation

---

## Why This Matters

PiFocus reframes focus detection as:

> Not "Is the user focused?"  
> But "Is the user engaged with the intended task over time?"

This approach is:
- More robust
- More realistic
- More aligned with real-world behavior

---

## Limitations

- Camera-only signals cannot capture cognitive intent
- Task context currently simplified (no deep semantic understanding)
- Privacy considerations for activity logging

---

## Future Work

- Lightweight on-device activity classification
- Multimodal signals (keyboard, audio)
- Temporal modeling (RNN / sequence models)
- Personalized behavior modeling

---

## Tech Stack

- Python
- OpenCV
- scikit-learn
- Raspberry Pi
- Pandas / NumPy

---

## Author

Jiayi Wang  
Columbia University  
Electrical Engineering

---

## License

MIT License
