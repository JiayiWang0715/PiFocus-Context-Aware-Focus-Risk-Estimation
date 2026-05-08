# PiFocus: Context-Aware Focus Risk Estimation

PiFocus estimates focus risk by combining visual attention signals from a Raspberry Pi camera pipeline with laptop activity context. The core idea is that visual engagement and task focus are not the same: a user can be looking at the screen while still being off task.

The system separates the problem into:

- **Visual attention**: whether the user appears present and visually engaged.
- **Task relevance**: whether the current activity matches the intended task.
- **Temporal focus risk**: whether off-task or uncertain behavior is sustained over time.

## Current Demo Pipeline

The current runnable demo is in the structured project folders:

```text
frontend/dashboard.py          Streamlit monitoring dashboard
laptop/activity_logger.py      Consent-based laptop activity logger
src/predict_camera_attention.py
src/replay_camera_stream.py
src/create_demo_activity_log.py
src/fusion_engine.py
run_live_demo.py               One-command local demo runner
```

Run the local demo with:

```bash
python3 run_live_demo.py
```

This starts:

1. A replay stream of pre-collected Raspberry Pi camera predictions.
2. The laptop activity logger.
3. A fusion loop that updates `data/fused_focus_log.csv`.
4. The Streamlit dashboard at `frontend/dashboard.py`.

You can also run the pieces manually:

```bash
python3 src/replay_camera_stream.py
python3 laptop/activity_logger.py --task_mode writing --duration 900
python3 src/fusion_engine.py
streamlit run frontend/dashboard.py
```

## Demo Replay Mode

For a stable classroom/demo setting, PiFocus includes a replay mode. The Raspberry Pi camera feature logger collects real camera-derived features locally, and `src/replay_camera_stream.py` replays pre-collected camera predictions over time.

This replay mode does **not** fabricate camera predictions and does **not** claim that the Raspberry Pi camera is live during replay. It is a controlled replay of collected camera prediction rows so the rest of the local pipeline can be demonstrated reliably.

The laptop context side can be run live with `laptop/activity_logger.py`. For privacy-safe controlled demos, `src/create_demo_activity_log.py` can generate `data/demo_activity_log.csv` using neutral inferred activity labels.

The activity labels represent inferred context from observable signals. They do not imply direct detection of external devices such as phones.

## Privacy Notes

PiFocus is designed as a local, consent-based prototype:

- No raw video is stored by the dashboard/fusion pipeline.
- No screenshots are stored.
- No keystrokes are stored.
- No audio is stored.
- Activity context is limited to app/window metadata, task mode, and idle time.

Generated or local runtime files such as `data/activity_log.csv`, `data/live_camera_predictions.csv`, and `data/fused_focus_log.csv` should not be committed unless intentionally shared as sample data.

## System Architecture

```text
Raspberry Pi camera features
        |
        v
Visual attention prediction
        |
        v
Camera prediction stream  +  Laptop activity context
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

The accumulated risk score is intentionally temporal: brief anomalies should not immediately trigger a strong warning, while sustained off-task behavior should increase risk.

## Repository Structure

```text
data/       Demo input CSVs and sample prediction data
frontend/   Streamlit dashboard
laptop/     Local laptop activity logger
models/     Trained visual attention model
results/    Earlier model evaluation artifacts
src/        Prediction, replay, demo data, and fusion scripts
```

Some older root-level scripts from the initial prototype are kept for project history. The current demo entrypoint is `run_live_demo.py`.

## Requirements

Main Python packages:

- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- streamlit-autorefresh optional, dashboard has a fallback refresh path

Install example:

```bash
pip install pandas numpy scikit-learn joblib streamlit streamlit-autorefresh
```

## Results and Motivation

Earlier experiments showed that camera-only models can capture physical presence and posture, but cannot reliably infer whether a visually engaged user is actually working on the intended task. This motivates combining visual attention with task context and temporal risk accumulation.

## Limitations

- Camera-derived signals do not reveal cognitive intent.
- Activity context is rule-based in this prototype.
- Replay mode is for stable demonstration, not a claim of live camera capture.
- The dashboard and fusion pipeline are local research/demo tools, not production monitoring software.

## Author

Jiayi Wang  
Columbia University  
Electrical Engineering

## License

MIT License
