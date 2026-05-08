# PiFocus: Context-Aware Focus Risk Estimation

PiFocus estimates focus risk by combining camera-based visual attention signals with local laptop activity context. The system is designed to run locally and update a Streamlit dashboard as new rows arrive.

The core idea is that visual engagement and task focus are not the same: a user can be looking at the screen while still being off task. PiFocus combines:

- **Visual attention**: whether the user appears present and visually engaged.
- **Task relevance**: whether the current active app/window matches the intended task.
- **Temporal focus risk**: whether off-task or uncertain behavior is sustained over time.

## Current Pipeline

The active code path is:

```text
src/pi_camera_feature_extractor.py   Raspberry Pi/OpenCV camera feature extraction
src/predict_camera_attention.py      ML + heuristic visual attention prediction
laptop/activity_logger.py            Consent-based laptop activity logger
src/fusion_engine.py                 Camera/activity fusion and risk accumulation
frontend/dashboard.py                Streamlit monitoring dashboard
run_live.py                          Local pipeline runner
```

Run the local monitoring pipeline with:

```bash
python3 run_live.py
```

This starts:

1. The laptop activity logger.
2. A camera prediction updater that reads `data/live_camera_features.csv` and writes `data/live_camera_predictions.csv`.
3. A fusion loop that updates `data/fused_focus_log.csv`.
4. The Streamlit dashboard at `frontend/dashboard.py`.

If the camera is attached to the same machine, start camera feature extraction too:

```bash
python3 run_live.py --start_camera
```

On a Raspberry Pi, the camera feature extractor can also run by itself:

```bash
python3 src/pi_camera_feature_extractor.py --output_csv data/live_camera_features.csv
```

Then the prediction step converts those features into camera attention predictions:

```bash
python3 src/predict_camera_attention.py   --input_csv data/live_camera_features.csv   --output_csv data/live_camera_predictions.csv
```

Manual laptop/dashboard commands:

```bash
python3 laptop/activity_logger.py --task_mode writing --duration 900
python3 src/fusion_engine.py --camera_csv data/live_camera_predictions.csv
streamlit run frontend/dashboard.py
```

## Data Files

The `data/` folder contains header-only CSV files so the expected schemas are visible in the repository. During a local run, PiFocus writes rows into these files on the local machine.

```text
data/live_camera_features.csv       Live camera features extracted from frames
data/live_camera_predictions.csv    Visual attention predictions
data/activity_log.csv               Laptop app/window/idle context
data/fused_focus_log.csv            Fused focus-risk output for the dashboard
data/session_features_v2.csv        Optional training dataset using the same camera feature schema
```

Camera feature columns:

```text
timestamp
elapsed_seconds
avg_brightness
avg_blur
face_ratio
missing_face_count
avg_face_area_ratio
avg_center_offset
avg_centeredness
avg_movement
movement_stability
focus_score
attention_state
label
```

Camera prediction columns:

```text
timestamp
elapsed_seconds
visual_attention_pred
visual_attention_prob
visual_attention_state
model_prob_focused
heuristic_prob
```

Activity context columns:

```text
timestamp
elapsed_seconds
task_mode
active_app
window_title
idle_seconds
```

## Raspberry Pi Camera Features

`src/pi_camera_feature_extractor.py` uses OpenCV to extract lightweight visual features from camera frames:

- brightness and blur/sharpness
- face presence ratio
- missing-face count in the rolling window
- face area ratio
- face center offset and centeredness
- frame-to-frame movement
- movement stability
- rule-based `focus_score`

Those columns match `src/predict_camera_attention.py`, which combines the trained model probability with the feature-based `focus_score` so strong visual evidence is not suppressed by an unreliable probability column.

## Privacy Notes

PiFocus is designed as a local, consent-based prototype:

- No raw video is stored by the dashboard/fusion pipeline.
- No screenshots are stored.
- No keystrokes are stored.
- No audio is stored.
- Activity context is limited to app/window metadata, task mode, and idle time.

## System Architecture

```text
Raspberry Pi camera frames
        |
        v
Live camera feature extraction
        |
        v
Visual attention prediction  +  Laptop activity context
             |                         |
             +-----------+-------------+
                         v
                  Fusion engine
                         |
                         v
            Accumulated focus risk dashboard
```

## Fusion States

The fusion engine emits these user-facing states:

- `focused_on_task`
- `present_but_off_task`
- `absent_or_disengaged`
- `uncertain`

It also computes:

- `task_relevance_score`
- `instantaneous_risk_score`
- `accumulated_risk_score`
- `risk_level`

The accumulated risk score is temporal: brief anomalies should not immediately trigger a strong warning, while sustained off-task behavior should increase risk.

## Repository Structure

```text
data/       Header-only CSV schemas and local runtime CSV targets
frontend/   Streamlit dashboard
laptop/     Local laptop activity logger
models/     Trained visual attention model
results/    Model evaluation artifacts
src/        Camera feature extraction, prediction, and fusion scripts
```

## Requirements

Main Python packages:

- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- streamlit-autorefresh optional, dashboard has a fallback refresh path
- opencv-python for camera feature extraction

Install example:

```bash
pip install pandas numpy scikit-learn joblib streamlit streamlit-autorefresh opencv-python
```

On Raspberry Pi, use the OpenCV package appropriate for the Pi OS image if `opencv-python` is not suitable.

## Results and Motivation

Earlier experiments showed that camera-only models can capture physical presence and posture, but cannot reliably infer whether a visually engaged user is actually working on the intended task. This motivates combining visual attention with task context and temporal risk accumulation.

## Limitations

- Camera-derived signals do not reveal cognitive intent.
- Activity context is rule-based in this prototype.
- The dashboard and fusion pipeline are local research tools, not production monitoring software.

## Author

Jiayi Wang                Tongxuan Li
Columbia University       Columbia University  
Electrical Engineering    Electrical Engineering

## License

MIT License
