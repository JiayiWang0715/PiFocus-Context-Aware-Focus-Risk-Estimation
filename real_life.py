from pathlib import Path
from collections import deque
import threading
import time

import av
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="PiFocus Camera Monitor",
    page_icon="🎯",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = ROOT_DIR / "session_features_v2(1).csv"


# --------------------------------------------------
# Global configuration
# --------------------------------------------------
FEATURE_COLUMNS = [
    "avg_brightness",
    "avg_blur",
    "face_ratio",
    "missing_face_count",
    "avg_face_area_ratio",
    "avg_center_offset",
    "avg_centeredness",
    "avg_movement",
    "movement_stability",
    "focus_score"
]

TARGET_COLUMN = "label"

ROLLING_WINDOW_SECONDS = 300
RED_THRESHOLD_SECONDS = 60
YELLOW_THRESHOLD_SECONDS = 30

FEATURE_WINDOW_SECONDS = 10
FRAME_SAMPLE_INTERVAL_SECONDS = 0.25
STATUS_SAMPLE_INTERVAL_SECONDS = 1.0


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def clamp(value, low=0.0, high=1.0):
    """Clamp a numeric value to a fixed range."""
    return max(low, min(high, value))


@st.cache_resource
def train_random_forest_model(csv_path):
    """
    Train a Random Forest model from the collected CSV file.

    If the CSV file is missing or invalid, the app will still work using
    the rule-based visual attention logic.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if not all(col in df.columns for col in required_columns):
        return None

    df = df.dropna(subset=required_columns).copy()

    y = df[TARGET_COLUMN].astype(str).str.lower().map({
        "distracted": 0,
        "focused": 1
    })

    valid_rows = y.notna()
    X = df.loc[valid_rows, FEATURE_COLUMNS]
    y = y.loc[valid_rows]

    if len(y.unique()) < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


def get_light_status(distracted_time):
    """
    Convert recent distracted time into a traffic-light status.
    """

    if distracted_time >= RED_THRESHOLD_SECONDS:
        return {
            "label": "Distracted",
            "emoji": "🔴",
            "color": "#ef4444",
            "message": "High distraction risk detected in the recent 5-minute window."
        }

    if distracted_time >= YELLOW_THRESHOLD_SECONDS:
        return {
            "label": "Slightly Distracted",
            "emoji": "🟡",
            "color": "#f59e0b",
            "message": "Some distraction signs detected in the recent 5-minute window."
        }

    return {
        "label": "Focused",
        "emoji": "🟢",
        "color": "#22c55e",
        "message": "The user appears visually focused in the recent 5-minute window."
    }


def render_status_card(status, distracted_time, current_focus_score, reason):
    """
    Render the main traffic-light status card.
    """

    st.markdown(
        f"""
        <div style="
            padding: 24px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 12px;
        ">
            <div style="display:flex; align-items:center; gap:18px;">
                <div style="
                    width:36px;
                    height:36px;
                    border-radius:50%;
                    background:{status['color']};
                    box-shadow: 0 0 24px {status['color']};
                    flex-shrink:0;
                "></div>
                <div>
                    <div style="font-size:14px; color:#cbd5e1;">
                        Current Focus Status
                    </div>
                    <div style="font-size:32px; font-weight:700; color:white;">
                        {status['emoji']} {status['label']}
                    </div>
                    <div style="font-size:15px; color:#cbd5e1;">
                        {status['message']}
                    </div>
                    <div style="font-size:14px; color:#cbd5e1; margin-top:6px;">
                        Distracted time in recent 5 minutes: {distracted_time:.1f} seconds |
                        Current focus score: {current_focus_score:.2f}
                    </div>
                    <div style="font-size:14px; color:#cbd5e1; margin-top:4px;">
                        Current reason: {reason}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_distraction_progress_bar(distracted_time):
    """
    Render a progress bar showing how close the user is to the red alert threshold.

    The progress bar fills as distracted time accumulates within the recent
    5-minute rolling window. When it reaches 100%, the red alert is triggered.
    """

    progress = clamp(distracted_time / RED_THRESHOLD_SECONDS, 0.0, 1.0)
    progress_percent = progress * 100

    remaining_seconds = max(RED_THRESHOLD_SECONDS - distracted_time, 0.0)

    if distracted_time >= RED_THRESHOLD_SECONDS:
        bar_color = "#ef4444"
        status_label = "Red Alert Triggered"
        status_message = "The distraction limit has been reached."
    elif distracted_time >= YELLOW_THRESHOLD_SECONDS:
        bar_color = "#f59e0b"
        status_label = "Warning Zone"
        status_message = f"{remaining_seconds:.1f} seconds remaining before red alert."
    else:
        bar_color = "#22c55e"
        status_label = "Focused Zone"
        status_message = f"{remaining_seconds:.1f} seconds remaining before red alert."

    st.markdown(
        f"""
        <div style="
            margin-top: 18px;
            margin-bottom: 22px;
            padding: 18px;
            border-radius: 16px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:10px;
            ">
                <div style="
                    font-size:16px;
                    font-weight:700;
                    color:#e5e7eb;
                ">
                    Distraction Alert Progress
                </div>
                <div style="
                    font-size:15px;
                    font-weight:700;
                    color:{bar_color};
                ">
                    {status_label} — {progress_percent:.1f}%
                </div>
            </div>

            <div style="
                width:100%;
                height:24px;
                background:rgba(255,255,255,0.10);
                border-radius:999px;
                overflow:hidden;
                border:1px solid rgba(255,255,255,0.14);
            ">
                <div style="
                    width:{progress_percent:.1f}%;
                    height:100%;
                    background:{bar_color};
                    border-radius:999px;
                    box-shadow:0 0 14px {bar_color};
                    transition: width 0.4s ease;
                "></div>
            </div>

            <div style="
                display:flex;
                justify-content:space-between;
                color:#94a3b8;
                font-size:12px;
                margin-top:7px;
            ">
                <span>0 sec</span>
                <span>Yellow: {YELLOW_THRESHOLD_SECONDS} sec</span>
                <span>Red: {RED_THRESHOLD_SECONDS} sec</span>
            </div>

            <div style="
                font-size:14px;
                color:#cbd5e1;
                margin-top:10px;
            ">
                Distracted time in recent 5 minutes: {distracted_time:.1f} seconds. 
                {status_message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_recent_history(history_df):
    """
    Plot recent focus score history and highlight distracted points.
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    if history_df.empty:
        ax.set_title("No camera samples collected yet")
        ax.set_ylim(0, 1)
        return fig

    focused_df = history_df[~history_df["is_distracted"]]
    distracted_df = history_df[history_df["is_distracted"]]

    ax.plot(
        history_df["session_time"],
        history_df["focus_score"],
        color="#60a5fa",
        linewidth=2.5,
        label="Focus Score"
    )

    ax.scatter(
        focused_df["session_time"],
        focused_df["focus_score"],
        color="#22c55e",
        s=35,
        label="Focused"
    )

    ax.scatter(
        distracted_df["session_time"],
        distracted_df["focus_score"],
        color="#ef4444",
        s=60,
        label="Distracted"
    )

    for _, row in distracted_df.iterrows():
        ax.axvspan(
            row["session_time"] - 0.5,
            row["session_time"] + 0.5,
            color="#ef4444",
            alpha=0.12
        )

    ax.axhline(
        y=0.55,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        label="Attention Threshold"
    )

    ax.set_title("Recent Camera-Based Focus Timeline")
    ax.set_xlabel("Session Time (seconds)")
    ax.set_ylabel("Focus Score")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    return fig


# --------------------------------------------------
# Video processor
# --------------------------------------------------
class CameraFocusProcessor(VideoProcessorBase):
    """
    Process webcam frames, extract simple camera-based behavioral features,
    and estimate whether the user appears distracted.
    """

    def __init__(self, model=None):
        self.model = model

        self.lock = threading.Lock()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.frame_samples = deque()
        self.status_history = deque()

        self.previous_face_center = None
        self.last_frame_sample_time = 0.0
        self.last_status_update_time = 0.0
        self.session_start_time = time.time()

        self.current_focus_score = 1.0
        self.current_reason = "Waiting for camera samples."
        self.current_is_distracted = False
        self.current_features = {feature: 0.0 for feature in FEATURE_COLUMNS}

    def recv(self, frame):
        """
        Receive a webcam frame, process it, and return an annotated frame.
        """

        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        frame_features = self.extract_frame_features(img, now)

        if now - self.last_frame_sample_time >= FRAME_SAMPLE_INTERVAL_SECONDS:
            with self.lock:
                self.frame_samples.append(frame_features)
                self.trim_frame_samples(now)

            self.last_frame_sample_time = now

        if now - self.last_status_update_time >= STATUS_SAMPLE_INTERVAL_SECONDS:
            with self.lock:
                aggregated_features = self.aggregate_recent_features()
                is_distracted, reason = self.predict_distraction(aggregated_features)

                self.current_focus_score = aggregated_features["focus_score"]
                self.current_reason = reason
                self.current_is_distracted = is_distracted
                self.current_features = aggregated_features

                self.status_history.append({
                    "time": now,
                    "session_time": now - self.session_start_time,
                    "focus_score": aggregated_features["focus_score"],
                    "is_distracted": is_distracted,
                    "reason": reason
                })

                self.trim_status_history(now)

            self.last_status_update_time = now

        annotated_img = self.draw_overlay(img, frame_features)

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    def extract_frame_features(self, img, timestamp):
        """
        Extract simple visual features from one webcam frame.
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        avg_brightness = float(np.mean(gray))
        avg_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        frame_h, frame_w = img.shape[:2]

        face_present = len(faces) > 0
        face_box = None
        face_area_ratio = 0.0
        center_offset = 1.0
        centeredness = 0.0
        movement = 0.0

        if face_present:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            face_box = (x, y, w, h)

            face_area_ratio = (w * h) / (frame_w * frame_h)

            face_center_x = x + w / 2
            face_center_y = y + h / 2

            dx = abs(face_center_x - frame_w / 2) / frame_w
            dy = abs(face_center_y - frame_h / 2) / frame_h

            center_offset = clamp(np.sqrt(dx * dx + dy * dy) / 0.707)
            centeredness = 1.0 - center_offset

            current_center = np.array([face_center_x / frame_w, face_center_y / frame_h])

            if self.previous_face_center is not None:
                movement = float(np.linalg.norm(current_center - self.previous_face_center))
                movement = clamp(movement * 5.0)

            self.previous_face_center = current_center
        else:
            self.previous_face_center = None

        return {
            "time": timestamp,
            "avg_brightness": avg_brightness,
            "avg_blur": avg_blur,
            "face_present": face_present,
            "face_box": face_box,
            "face_area_ratio": face_area_ratio,
            "center_offset": center_offset,
            "centeredness": centeredness,
            "movement": movement
        }

    def trim_frame_samples(self, now):
        """
        Keep only recent frame samples for feature aggregation.
        """

        while self.frame_samples and now - self.frame_samples[0]["time"] > FEATURE_WINDOW_SECONDS:
            self.frame_samples.popleft()

    def trim_status_history(self, now):
        """
        Keep only recent status samples for rolling-window analysis.
        """

        while self.status_history and now - self.status_history[0]["time"] > ROLLING_WINDOW_SECONDS:
            self.status_history.popleft()

    def aggregate_recent_features(self):
        """
        Aggregate recent frame-level samples into one feature vector.
        """

        if not self.frame_samples:
            return {
                "avg_brightness": 0.0,
                "avg_blur": 0.0,
                "face_ratio": 0.0,
                "missing_face_count": 0,
                "avg_face_area_ratio": 0.0,
                "avg_center_offset": 1.0,
                "avg_centeredness": 0.0,
                "avg_movement": 1.0,
                "movement_stability": 0.0,
                "focus_score": 0.0
            }

        samples = list(self.frame_samples)

        avg_brightness = float(np.mean([s["avg_brightness"] for s in samples]))
        avg_blur = float(np.mean([s["avg_blur"] for s in samples]))

        face_present_values = np.array([1.0 if s["face_present"] else 0.0 for s in samples])
        face_ratio = float(np.mean(face_present_values))
        missing_face_count = int(np.sum(face_present_values == 0.0))

        avg_face_area_ratio = float(np.mean([s["face_area_ratio"] for s in samples]))
        avg_center_offset = float(np.mean([s["center_offset"] for s in samples]))
        avg_centeredness = float(np.mean([s["centeredness"] for s in samples]))
        avg_movement = float(np.mean([s["movement"] for s in samples]))

        movement_stability = 1.0 - clamp(avg_movement)

        blur_score = clamp(avg_blur / 150.0)
        brightness_score = 1.0 - abs(avg_brightness - 128.0) / 128.0
        brightness_score = clamp(brightness_score)

        face_area_score = clamp(avg_face_area_ratio / 0.35)

        focus_score = (
            0.25 * face_ratio +
            0.20 * avg_centeredness +
            0.15 * movement_stability +
            0.15 * face_area_score +
            0.10 * blur_score +
            0.10 * brightness_score +
            0.05 * (1.0 - clamp(missing_face_count / max(len(samples), 1)))
        )

        focus_score = clamp(focus_score)

        return {
            "avg_brightness": avg_brightness,
            "avg_blur": avg_blur,
            "face_ratio": face_ratio,
            "missing_face_count": missing_face_count,
            "avg_face_area_ratio": avg_face_area_ratio,
            "avg_center_offset": avg_center_offset,
            "avg_centeredness": avg_centeredness,
            "avg_movement": avg_movement,
            "movement_stability": movement_stability,
            "focus_score": focus_score
        }

    def predict_distraction(self, features):
        """
        Predict whether the user appears distracted.

        The trained Random Forest model is used when available.
        Severe visual conditions are still treated as distraction signals.
        """

        severe_no_face = features["face_ratio"] < 0.50
        severe_off_center = features["avg_center_offset"] > 0.45
        severe_far = features["avg_face_area_ratio"] < 0.04
        severe_movement = features["avg_movement"] > 0.45
        low_score = features["focus_score"] < 0.55

        model_predicts_distracted = False

        if self.model is not None:
            feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            prediction = int(self.model.predict(feature_df)[0])
            model_predicts_distracted = prediction == 0

        is_distracted = (
            model_predicts_distracted
            or severe_no_face
            or severe_off_center
            or severe_far
            or severe_movement
            or low_score
        )

        if severe_no_face:
            reason = "Face is missing or turned away from the camera."
        elif severe_off_center:
            reason = "Face is far from the center of the camera frame."
        elif severe_far:
            reason = "Face appears too small or too far from the camera."
        elif severe_movement:
            reason = "Movement level is high."
        elif model_predicts_distracted:
            reason = "The trained model predicts a distracted state."
        elif low_score:
            reason = "The visual focus score is below the attention threshold."
        else:
            reason = "Face position and movement appear stable."

        return is_distracted, reason

    def compute_recent_distracted_time(self):
        """
        Estimate total distracted time in the rolling 5-minute window.
        """

        if not self.status_history:
            return 0.0

        rows = list(self.status_history)
        now = time.time()

        distracted_time = 0.0

        for i, row in enumerate(rows):
            if i < len(rows) - 1:
                duration = rows[i + 1]["time"] - row["time"]
            else:
                duration = min(now - row["time"], STATUS_SAMPLE_INTERVAL_SECONDS)

            duration = clamp(duration, 0.0, 5.0)

            if row["is_distracted"]:
                distracted_time += duration

        return distracted_time

    def get_dashboard_state(self):
        """
        Return a thread-safe snapshot of current dashboard state.
        """

        with self.lock:
            distracted_time = self.compute_recent_distracted_time()
            history_df = pd.DataFrame(list(self.status_history))

            if not history_df.empty:
                history_df = history_df.sort_values("time")

            return {
                "distracted_time": distracted_time,
                "current_focus_score": self.current_focus_score,
                "current_reason": self.current_reason,
                "current_is_distracted": self.current_is_distracted,
                "current_features": dict(self.current_features),
                "history_df": history_df
            }

    def draw_overlay(self, img, frame_features):
        """
        Draw face box and current status on the returned webcam frame.
        """

        output = img.copy()

        if frame_features["face_box"] is not None:
            x, y, w, h = frame_features["face_box"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (34, 197, 94), 2)

        with self.lock:
            is_distracted = self.current_is_distracted
            focus_score = self.current_focus_score
            reason = self.current_reason

        if is_distracted:
            color = (68, 68, 239)
            label = "Distracted"
        else:
            color = (94, 197, 34)
            label = "Focused"

        cv2.putText(
            output,
            f"Status: {label}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        cv2.putText(
            output,
            f"Focus Score: {focus_score:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            output,
            reason[:60],
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        return output


# --------------------------------------------------
# Load optional trained model
# --------------------------------------------------
rf_model = train_random_forest_model(CSV_PATH)


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------
st.title("PiFocus: Real-Time Visual Attention Sensor")
st.caption(
    "This demo uses the webcam to estimate visible attention patterns. "
    "It is not a perfect cognitive focus detector."
)

st.info(
    "Click START and allow camera permission in the browser. "
    "The system will monitor face presence, face position, face size, and movement stability."
)

if rf_model is None:
    st.warning(
        "No valid trained Random Forest model was loaded. "
        "The app will use rule-based camera behavior analysis."
    )
else:
    st.success(
        "Random Forest model loaded from the collected CSV file. "
        "The app will combine model prediction with rule-based visual checks."
    )

st.divider()


# --------------------------------------------------
# Webcam streamer
# --------------------------------------------------
ctx = webrtc_streamer(
    key="pifocus-camera-monitor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: CameraFocusProcessor(model=rf_model),
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# --------------------------------------------------
# Live dashboard placeholders
# --------------------------------------------------
status_placeholder = st.empty()
metrics_placeholder = st.empty()
timeline_placeholder = st.empty()
explanation_placeholder = st.empty()


if ctx.video_processor:
    while ctx.state.playing:
        state = ctx.video_processor.get_dashboard_state()

        distracted_time = state["distracted_time"]
        current_focus_score = state["current_focus_score"]
        current_reason = state["current_reason"]
        history_df = state["history_df"]

        light_status = get_light_status(distracted_time)

        with status_placeholder.container():
            st.subheader("Current Focus Status")
            render_status_card(
                status=light_status,
                distracted_time=distracted_time,
                current_focus_score=current_focus_score,
                reason=current_reason
            )

        with metrics_placeholder.container():
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Distracted Time in Recent 5 Minutes",
                    f"{distracted_time:.1f} sec"
                )

            with metric_col2:
                st.metric(
                    "Current Focus Score",
                    f"{current_focus_score:.2f}"
                )

            with metric_col3:
                st.metric(
                    "Alert Threshold",
                    f"{RED_THRESHOLD_SECONDS} sec / 5 min"
                )

            render_distraction_progress_bar(distracted_time)

        with timeline_placeholder.container():
            st.subheader("Recent Focus Timeline")
            fig = plot_recent_history(history_df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if not history_df.empty:
                recent_distracted = history_df[history_df["is_distracted"]]

                if len(recent_distracted) > 0:
                    recent_times = (
                        recent_distracted["session_time"]
                        .round(1)
                        .astype(str)
                        .tolist()
                    )

                    st.warning(
                        "Distracted samples detected at session seconds: "
                        + " | ".join(recent_times[-20:])
                    )
                else:
                    st.success("No distracted samples detected in the recent window.")

        with explanation_placeholder.container():
            st.subheader("User-Friendly Explanation")

            if distracted_time >= RED_THRESHOLD_SECONDS:
                st.error(
                    "The system detected more than 1 minute of distracted behavior "
                    "within the recent 5-minute window. Please return to the task and face the screen."
                )
            elif distracted_time >= YELLOW_THRESHOLD_SECONDS:
                st.warning(
                    "The system detected some distraction signs. "
                    "Try to keep your face centered and reduce unrelated activity."
                )
            else:
                st.success(
                    "The recent camera behavior looks stable. "
                    "The user appears visually focused."
                )

            st.markdown("### Current Reason")
            st.write(current_reason)

            st.markdown("### Suggested Action")
            if distracted_time >= RED_THRESHOLD_SECONDS:
                st.write("- Return to the main document or task.")
                st.write("- Keep your face closer to the center of the camera frame.")
                st.write("- Reduce unrelated tabs, apps, or notifications.")
            elif distracted_time >= YELLOW_THRESHOLD_SECONDS:
                st.write("- Keep your posture stable.")
                st.write("- Avoid turning away from the screen for too long.")
                st.write("- Refocus on the current task.")
            else:
                st.write("- Keep the current working rhythm.")
                st.write("- Continue using the task-related document.")
                st.write("- Take a planned break after a longer work session.")

        time.sleep(1)

else:
    st.info("Waiting for the camera stream to start.")


# --------------------------------------------------
# Technical details
# --------------------------------------------------
with st.expander("Technical Details", expanded=False):
    st.markdown(
        """
        The real-time camera version estimates visual attention using:

        - Face presence
        - Face size in the frame
        - Face center offset
        - Face centeredness
        - Movement level
        - Brightness and blur

        A red alert is triggered when distracted behavior accumulates to at least
        60 seconds within the recent 5-minute rolling window.
        """
    )

    st.markdown("### Thresholds")

    threshold_df = pd.DataFrame({
        "Status": ["Focused", "Slightly Distracted", "Distracted"],
        "Recent Distracted Time": [
            "< 30 seconds",
            "30–60 seconds",
            ">= 60 seconds"
        ],
        "Light": ["Green", "Yellow", "Red"]
    })

    st.dataframe(threshold_df, use_container_width=True)


st.caption(
    "PiFocus estimates visible attention patterns from camera-based behavioral cues. "
    "It should be interpreted as a visual attention sensor, not a perfect focus detector."
)