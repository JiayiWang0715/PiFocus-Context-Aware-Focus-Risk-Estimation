from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="PiFocus Dashboard",
    page_icon="🎯",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = ROOT_DIR / "session_features_v2(1).csv"
APP_USAGE_LOG = RESULTS_DIR / "app_usage_samples.csv"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def clamp(value, low=0.0, high=1.0):
    """Keep a numeric value within a fixed range."""
    return max(low, min(high, value))


def compute_visual_attention_score(features):
    """
    Estimate visual attention from camera-based behavioral features.
    Higher score means the user appears more visually attentive.
    """

    face_ratio = clamp(features["face_ratio"])
    centeredness = clamp(features["avg_centeredness"])
    missing_face_score = 1.0 - clamp(features["missing_face_count"] / 10.0)

    # In this demo, higher blur/sharpness value means the image is more usable.
    blur_score = clamp(features["avg_blur"] / 100.0)

    # Brightness close to 128 is treated as more stable.
    brightness_score = 1.0 - abs(features["avg_brightness"] - 128.0) / 128.0
    brightness_score = clamp(brightness_score)

    # Larger face area usually means the user is closer to the camera.
    face_area_score = clamp(features["avg_face_area_ratio"] / 0.35)

    # Smaller center offset means the face is closer to the center of the frame.
    center_offset_score = 1.0 - clamp(features["avg_center_offset"])

    visual_score = (
        0.25 * face_ratio +
        0.20 * centeredness +
        0.15 * missing_face_score +
        0.10 * blur_score +
        0.10 * brightness_score +
        0.10 * face_area_score +
        0.10 * center_offset_score
    )

    return clamp(visual_score)


def get_task_relevance_score(active_app, main_task_app):
    """
    Estimate task relevance by comparing the current active app with the user's selected main task app.
    """

    if active_app == main_task_app:
        return 0.95

    related_apps = {
        "Google Docs": ["Browser", "Chrome - Research Paper"],
        "Overleaf": ["Browser", "Chrome - Research Paper"],
        "VS Code": ["Browser", "GitHub"],
        "Canvas / CourseWorks": ["Browser", "Google Docs"],
        "Chrome - Research Paper": ["Browser", "Google Docs"],
        "Microsoft Word": ["File Explorer", "Browser"]
    }

    if active_app in related_apps.get(main_task_app, []):
        return 0.75

    low_relevance_apps = [
        "YouTube",
        "Instagram",
        "Netflix",
        "Messages",
        "Discord"
    ]

    if active_app in low_relevance_apps:
        return 0.15

    return 0.45


def compute_focus_risk(visual_score, task_score):
    """
    Combine visual attention and task relevance into a focus risk score.
    Higher risk means the user is more likely to be distracted.
    """

    risk = 1.0 - (0.60 * visual_score + 0.40 * task_score)
    return clamp(risk)


def get_focus_status(risk):
    """
    Convert the focus risk score into a traffic-light user status.
    """

    if risk >= 0.65:
        return {
            "label": "Distracted",
            "subtext": "The user currently appears highly distracted.",
            "color": "#ef4444",
            "emoji": "🔴"
        }

    if risk >= 0.40:
        return {
            "label": "Slightly Distracted",
            "subtext": "The user shows some signs of distraction.",
            "color": "#f59e0b",
            "emoji": "🟡"
        }

    return {
        "label": "Focused",
        "subtext": "The user currently appears visually focused.",
        "color": "#22c55e",
        "emoji": "🟢"
    }


def create_demo_history(current_visual, current_task, periods=24):
    """
    Generate a demo timeline for visualizing recent focus risk.
    In a real system, this should be replaced by logged model outputs.
    """

    times = pd.date_range(
        end=pd.Timestamp.now(),
        periods=periods,
        freq="min"
    )

    visual_history = np.clip(
        np.random.normal(loc=current_visual, scale=0.08, size=periods),
        0,
        1
    )

    task_history = np.clip(
        np.random.normal(loc=current_task, scale=0.10, size=periods),
        0,
        1
    )

    focus_risk = 1.0 - (0.60 * visual_history + 0.40 * task_history)
    focus_risk = np.clip(focus_risk, 0, 1)

    history_df = pd.DataFrame({
        "time": times,
        "visual_attention_score": visual_history,
        "task_relevance_score": task_history,
        "focus_risk_score": focus_risk
    })

    def assign_risk_level(value):
        if value >= 0.65:
            return "high"
        if value >= 0.40:
            return "medium"
        return "low"

    history_df["risk_level"] = history_df["focus_risk_score"].apply(assign_risk_level)

    return history_df


def render_status_light(status, risk):
    """
    Render the current focus status as a traffic-light style card.
    """

    st.markdown(
        f"""
        <div style="
            padding: 22px;
            border-radius: 18px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 10px;
        ">
            <div style="display:flex; align-items:center; gap:18px;">
                <div style="
                    width:32px;
                    height:32px;
                    border-radius:50%;
                    background:{status['color']};
                    box-shadow: 0 0 22px {status['color']};
                    flex-shrink: 0;
                "></div>
                <div>
                    <div style="font-size:14px; color:#cbd5e1;">
                        Current Focus Status
                    </div>
                    <div style="font-size:30px; font-weight:700; color:white;">
                        {status['emoji']} {status['label']}
                    </div>
                    <div style="font-size:15px; color:#cbd5e1;">
                        {status['subtext']} Risk Score: {risk * 100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_alert_display(focus_risk, active_app, main_task_app):
    """
    Render a large user-facing alert panel based on the current focus risk.
    This is designed for demo and user-facing explanation.
    """

    if focus_risk >= 0.65:
        alert_title = "High Distraction Alert"
        alert_icon = "🚨"
        alert_color = "#ef4444"
        alert_bg = "rgba(239, 68, 68, 0.16)"
        alert_border = "rgba(239, 68, 68, 0.65)"
        alert_message = (
            "The system detects a high focus risk. "
            "You may be away from your main task or visually distracted."
        )
        action_message = (
            "Suggested action: return to the main task window, reduce unrelated apps, "
            "and face the screen more steadily."
        )

    elif focus_risk >= 0.40:
        alert_title = "Mild Distraction Warning"
        alert_icon = "⚠️"
        alert_color = "#f59e0b"
        alert_bg = "rgba(245, 158, 11, 0.16)"
        alert_border = "rgba(245, 158, 11, 0.65)"
        alert_message = (
            "The system detects some signs of distraction. "
            "Your current focus state is not critical, but it may need attention."
        )
        action_message = (
            "Suggested action: keep your posture stable and stay on your selected task app."
        )

    else:
        alert_title = "Focus Looks Stable"
        alert_icon = "✅"
        alert_color = "#22c55e"
        alert_bg = "rgba(34, 197, 94, 0.14)"
        alert_border = "rgba(34, 197, 94, 0.55)"
        alert_message = (
            "The system estimates that your current visual attention is stable."
        )
        action_message = (
            "Suggested action: keep working in the current rhythm."
        )

    if active_app != main_task_app:
        app_message = (
            f"Main task app: {main_task_app} | Current active app: {active_app}"
        )
    else:
        app_message = (
            f"You are currently working in your selected main task app: {main_task_app}"
        )

    st.markdown(
        f"""
        <div style="
            padding: 26px;
            border-radius: 20px;
            background: {alert_bg};
            border: 2px solid {alert_border};
            margin-top: 18px;
            margin-bottom: 18px;
            box-shadow: 0 0 24px rgba(0,0,0,0.25);
        ">
            <div style="display:flex; align-items:center; gap:20px;">
                <div style="
                    font-size:46px;
                    line-height:1;
                ">
                    {alert_icon}
                </div>
                <div>
                    <div style="
                        font-size:26px;
                        font-weight:800;
                        color:{alert_color};
                        margin-bottom:8px;
                    ">
                        {alert_title}
                    </div>
                    <div style="
                        font-size:16px;
                        color:#e5e7eb;
                        margin-bottom:8px;
                    ">
                        {alert_message}
                    </div>
                    <div style="
                        font-size:15px;
                        color:#cbd5e1;
                        margin-bottom:8px;
                    ">
                        {action_message}
                    </div>
                    <div style="
                        font-size:14px;
                        color:#94a3b8;
                    ">
                        {app_message} | Focus risk: {focus_risk * 100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_explanation_points(features, active_app, risk):
    """
    Generate user-friendly explanations based on current feature values.
    """

    reasons = []

    if features["missing_face_count"] >= 3:
        reasons.append(
            "The face was missing several times, which may indicate that the user moved away from the screen."
        )

    if features["avg_centeredness"] < 0.65:
        reasons.append(
            "The face is not well centered, which may suggest unstable visual attention."
        )

    if features["avg_center_offset"] > 0.35:
        reasons.append(
            "The face is far from the center of the frame, which may indicate gaze or posture shift."
        )

    if features["avg_face_area_ratio"] < 0.12:
        reasons.append(
            "The face appears small in the camera frame, which may mean the user is far from the screen."
        )

    if features["avg_blur"] < 45:
        reasons.append(
            "The camera image appears blurry, which may reduce the reliability of the estimate."
        )

    if active_app in ["YouTube", "Instagram", "Netflix"]:
        reasons.append(
            f"The current application is {active_app}, which is treated as less task-relevant in this demo."
        )

    if risk < 0.40 and len(reasons) == 0:
        reasons.append(
            "The face appears stable and centered, and the current task context is relevant."
        )

    return reasons[:3]


def get_user_recommendations(risk):
    """
    Generate simple user-facing suggestions based on focus risk.
    """

    if risk >= 0.65:
        return [
            "Return to the main task window.",
            "Reduce unrelated apps or notifications.",
            "Take a short break if you feel tired."
        ]

    if risk >= 0.40:
        return [
            "Try to keep your face centered in the camera frame.",
            "Reduce frequent movement or window switching.",
            "Return to a task-relevant application."
        ]

    return [
        "Keep your current working posture.",
        "Stay on the task-relevant application.",
        "Consider taking a planned break after a longer work session."
    ]


def show_technical_image(image_path, caption):
    """
    Display a technical result image if it exists.
    """

    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path.name}")

def load_real_app_usage_segments(log_path):
    """
    Load real app usage data from app_usage_samples.csv and convert it into timeline segments.
    """

    if not log_path.exists():
        return None

    usage_df = pd.read_csv(log_path)

    if usage_df.empty:
        return None

    required_columns = ["timestamp", "active_app"]
    if not all(col in usage_df.columns for col in required_columns):
        return None

    usage_df["timestamp"] = pd.to_datetime(usage_df["timestamp"], errors="coerce")
    usage_df = usage_df.dropna(subset=["timestamp", "active_app"]).copy()

    if usage_df.empty or len(usage_df) < 2:
        return None

    usage_df = usage_df.sort_values("timestamp").copy()

    usage_df["next_timestamp"] = usage_df["timestamp"].shift(-1)
    usage_df["duration_seconds"] = (
        usage_df["next_timestamp"] - usage_df["timestamp"]
    ).dt.total_seconds()

    usage_df["duration_seconds"] = usage_df["duration_seconds"].fillna(2.0)
    usage_df["duration_seconds"] = usage_df["duration_seconds"].clip(lower=0, upper=10)

    usage_df["elapsed_start"] = (
        usage_df["timestamp"] - usage_df["timestamp"].iloc[0]
    ).dt.total_seconds()

    usage_df["elapsed_end"] = usage_df["elapsed_start"] + usage_df["duration_seconds"]

    segments = []

    current_app = usage_df.iloc[0]["active_app"]
    current_start = usage_df.iloc[0]["elapsed_start"]
    current_end = usage_df.iloc[0]["elapsed_end"]

    for _, row in usage_df.iloc[1:].iterrows():
        app = row["active_app"]

        if app == current_app:
            current_end = row["elapsed_end"]
        else:
            segments.append({
                "app": current_app,
                "start": current_start,
                "end": current_end,
                "duration_seconds": current_end - current_start
            })

            current_app = app
            current_start = row["elapsed_start"]
            current_end = row["elapsed_end"]

    segments.append({
        "app": current_app,
        "start": current_start,
        "end": current_end,
        "duration_seconds": current_end - current_start
    })

    return pd.DataFrame(segments)


def create_demo_app_usage_segments(session_df):
    """
    Create a simulated app usage timeline from collected focus data.
    This is used when no real app usage log is available.
    """

    if session_df is None or session_df.empty or "elapsed_seconds" not in session_df.columns:
        return pd.DataFrame()

    session_df = session_df.sort_values("elapsed_seconds").copy()

    demo_segments = []
    segment_start = float(session_df["elapsed_seconds"].iloc[0])
    current_app = None

    for _, row in session_df.iterrows():
        time_value = float(row["elapsed_seconds"])

        if "label" in session_df.columns:
            label = str(row["label"]).lower()
        else:
            label = "focused" if row.get("focus_score", 1.0) >= 0.55 else "distracted"

        # Demo logic:
        # Focused periods are shown as Google Docs.
        # Distracted periods are mapped to several non-task apps.
        if label == "focused":
            app = "Google Docs"
        else:
            time_bucket = int(time_value // 30) % 6

            if time_bucket == 0:
                app = "YouTube"
            elif time_bucket == 1:
                app = "Browser"
            elif time_bucket == 2:
                app = "Messages"
            elif time_bucket == 3:
                app = "Instagram"
            elif time_bucket == 4:
                app = "VS Code"
            else:
                app = "Other Activity"

        if current_app is None:
            current_app = app
            segment_start = time_value
        elif app != current_app:
            demo_segments.append({
                "app": current_app,
                "start": segment_start,
                "end": time_value,
                "duration_seconds": time_value - segment_start
            })

            current_app = app
            segment_start = time_value

    final_time = float(session_df["elapsed_seconds"].iloc[-1])

    demo_segments.append({
        "app": current_app,
        "start": segment_start,
        "end": final_time,
        "duration_seconds": final_time - segment_start
    })

    return pd.DataFrame(demo_segments)


def compress_app_segments_to_top_n(app_segments_df, top_n=10):
    """
    Keep only the top N apps by total time.
    All remaining apps are grouped into "Other".
    Adjacent segments with the same grouped label are merged.
    """

    if app_segments_df is None or app_segments_df.empty:
        return pd.DataFrame()

    df = app_segments_df.copy()

    app_totals = (
        df.groupby("app", as_index=False)["duration_seconds"]
        .sum()
        .sort_values("duration_seconds", ascending=False)
    )

    top_apps = app_totals.head(top_n)["app"].tolist()

    df["app_grouped"] = df["app"].apply(
        lambda x: x if x in top_apps else "Other"
    )

    df = df.sort_values("start").copy()

    merged_segments = []

    current_app = df.iloc[0]["app_grouped"]
    current_start = df.iloc[0]["start"]
    current_end = df.iloc[0]["end"]

    for _, row in df.iloc[1:].iterrows():
        next_app = row["app_grouped"]
        next_start = row["start"]
        next_end = row["end"]

        if next_app == current_app and abs(next_start - current_end) < 1e-9:
            current_end = next_end
        else:
            merged_segments.append({
                "app": current_app,
                "start": current_start,
                "end": current_end,
                "duration_seconds": current_end - current_start
            })

            current_app = next_app
            current_start = next_start
            current_end = next_end

    merged_segments.append({
        "app": current_app,
        "start": current_start,
        "end": current_end,
        "duration_seconds": current_end - current_start
    })

    return pd.DataFrame(merged_segments)


def summarize_app_usage(app_segments_df):
    """
    Summarize total time spent in each app.
    """

    if app_segments_df is None or app_segments_df.empty:
        return pd.DataFrame()

    summary_df = (
        app_segments_df
        .groupby("app", as_index=False)["duration_seconds"]
        .sum()
        .sort_values("duration_seconds", ascending=False)
    )

    summary_df["minutes"] = summary_df["duration_seconds"] / 60.0

    return summary_df


def plot_app_usage_timeline(app_segments_df):
    """
    Plot a horizontal app usage timeline.
    Each app is shown with a different color.
    Apps outside the top 10 should already be grouped into 'Other'.
    """

    fig, ax = plt.subplots(figsize=(12, 3.3))

    if app_segments_df is None or app_segments_df.empty:
        ax.set_title("No app usage data available")
        ax.set_yticks([])
        return fig

    app_names = app_segments_df["app"].dropna().unique().tolist()

    # Use matplotlib tab10 colors for main apps
    tab10_colors = list(plt.get_cmap("tab10").colors)

    color_map = {}
    color_index = 0

    for app in app_names:
        if app == "Other":
            color_map[app] = "#9ca3af"  # Gray
        else:
            color_map[app] = tab10_colors[color_index % 10]
            color_index += 1

    y_position = 0

    for _, row in app_segments_df.iterrows():
        app = row["app"]
        start = row["start"]
        duration = max(row["end"] - row["start"], 0)

        ax.barh(
            y=y_position,
            width=duration,
            left=start,
            height=0.5,
            color=color_map.get(app, "#9ca3af"),
            edgecolor="white",
            linewidth=0.6
        )

        # Only label sufficiently long segments to avoid clutter
        if duration >= 60:
            ax.text(
                start + duration / 2,
                y_position,
                app,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold"
            )

    ax.set_title("Task / App Usage Timeline")
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)

    legend_handles = [
        mpatches.Patch(color=color_map[app], label=app)
        for app in app_names
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(app_names), 6),
        frameon=False
    )

    plt.tight_layout()

    return fig

    app_names = app_segments_df["app"].dropna().unique().tolist()

    color_map = {
        "Google Docs": "#22c55e",
        "VS Code": "#3b82f6",
        "Overleaf": "#14b8a6",
        "Browser": "#f59e0b",
        "YouTube": "#ef4444",
        "Instagram": "#ec4899",
        "Messages": "#a855f7",
        "Unknown": "#64748b"
    }

    y_position = 0

    for _, row in app_segments_df.iterrows():
        app = row["app"]
        start = row["start"]
        duration = max(row["end"] - row["start"], 0)

        color = color_map.get(app, "#94a3b8")

        ax.barh(
            y=y_position,
            width=duration,
            left=start,
            height=0.45,
            color=color,
            edgecolor="white",
            linewidth=0.5
        )

        if duration >= 20:
            ax.text(
                start + duration / 2,
                y_position,
                app,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold"
            )

    ax.set_title("Task / App Usage Timeline")
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)

    return fig

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.title("PiFocus Controls")
st.sidebar.write("Use these controls to simulate user behavior and task context.")

main_task_app = st.sidebar.selectbox(
    "Main Task App",
    [
        "Google Docs",
        "Overleaf",
        "VS Code",
        "Canvas / CourseWorks",
        "Chrome - Research Paper",
        "Browser",
        "Microsoft Word"
    ],
    index=0
)

active_app = st.sidebar.selectbox(
    "Current Active App",
    [
        "Google Docs",
        "Overleaf",
        "VS Code",
        "Canvas / CourseWorks",
        "Chrome - Research Paper",
        "Browser",
        "Microsoft Word",
        "YouTube",
        "Instagram",
        "Netflix",
        "Messages",
        "Discord",
        "File Explorer",
        "Unknown"
    ],
    index=0
)

avg_brightness = st.sidebar.slider("Average Brightness", 0, 255, 128)
avg_blur = st.sidebar.slider("Average Blur / Sharpness", 0, 150, 80)
face_ratio = st.sidebar.slider("Face Ratio", 0.0, 1.0, 0.90)
missing_face_count = st.sidebar.slider("Missing Face Count", 0, 20, 1)
avg_face_area_ratio = st.sidebar.slider("Average Face Area Ratio", 0.0, 1.0, 0.20)
avg_center_offset = st.sidebar.slider("Average Center Offset", 0.0, 1.0, 0.20)
avg_centeredness = st.sidebar.slider("Average Centeredness", 0.0, 1.0, 0.85)
avg_movement = st.sidebar.slider("Average Movement", 0.0, 1.0, 0.20)
movement_stability = st.sidebar.slider("Movement Stability", 0.0, 1.0, 0.80)
focus_score = st.sidebar.slider("Focus Score", 0.0, 1.0, 0.80)

features = {
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

visual_score = compute_visual_attention_score(features)
task_score = get_task_relevance_score(active_app, main_task_app)
focus_risk = compute_focus_risk(visual_score, task_score)
status = get_focus_status(focus_risk)
history_df = create_demo_history(visual_score, task_score)


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------
st.title("PiFocus: Visual Attention Sensor")
st.caption(
    "A user-friendly dashboard for estimating focus risk from camera-based behavioral features."
)

st.divider()


# --------------------------------------------------
# Current focus status
# --------------------------------------------------
st.subheader("Current Focus Status")

render_status_light(status, focus_risk)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Visual Attention", f"{visual_score * 100:.1f}%")

with metric_col2:
    st.metric("Task Relevance", f"{task_score * 100:.1f}%")

with metric_col3:
    st.metric("Focus Risk", f"{focus_risk * 100:.1f}%")

def render_focus_progress_bar(focus_risk):
    """
    Render a user-facing focus risk progress bar.
    """

    risk_percent = focus_risk * 100

    if focus_risk >= 0.65:
        bar_color = "#ef4444"
        status_text = "High Risk"
    elif focus_risk >= 0.40:
        bar_color = "#f59e0b"
        status_text = "Medium Risk"
    else:
        bar_color = "#22c55e"
        status_text = "Low Risk"

    st.markdown(
        f"""
        <div style="margin-top: 18px; margin-bottom: 20px;">
            <div style="
                display:flex;
                justify-content:space-between;
                margin-bottom:8px;
                color:#e5e7eb;
                font-size:15px;
                font-weight:600;
            ">
                <span>Focus Risk Progress</span>
                <span>{status_text} — {risk_percent:.1f}%</span>
            </div>

            <div style="
                width:100%;
                height:22px;
                background:rgba(255,255,255,0.10);
                border-radius:999px;
                overflow:hidden;
                border:1px solid rgba(255,255,255,0.12);
            ">
                <div style="
                    width:{risk_percent:.1f}%;
                    height:100%;
                    background:{bar_color};
                    border-radius:999px;
                    box-shadow:0 0 12px {bar_color};
                "></div>
            </div>

            <div style="
                display:flex;
                justify-content:space-between;
                color:#94a3b8;
                font-size:12px;
                margin-top:6px;
            ">
                <span>Focused</span>
                <span>Slightly Distracted</span>
                <span>Distracted</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_alert_display(
    focus_risk=focus_risk,
    active_app=active_app,
    main_task_app=main_task_app
)

if active_app == main_task_app:
    st.success(f"You are currently working in your selected main task app: {main_task_app}.")
else:
    st.warning(
        f"Your selected main task app is {main_task_app}, "
        f"but the current active app is {active_app}."
    )

st.divider()


# --------------------------------------------------
# Timeline visualization
# --------------------------------------------------
st.subheader("Recent Focus Risk Timeline")

fig, ax = plt.subplots(figsize=(12, 4))

# Background risk zones
ax.axhspan(0.65, 1.0, color="#ef4444", alpha=0.14, label="High Risk Zone")
ax.axhspan(0.40, 0.65, color="#f59e0b", alpha=0.12, label="Medium Risk Zone")
ax.axhspan(0.00, 0.40, color="#22c55e", alpha=0.10, label="Low Risk Zone")

st.subheader("Recent Focus Timeline")

# --------------------------------------------------
# App usage timeline
# --------------------------------------------------
st.subheader("Task / App Usage Timeline")

real_app_segments_df = load_real_app_usage_segments(APP_USAGE_LOG)

if real_app_segments_df is not None:
    raw_app_segments_df = real_app_segments_df
    st.caption("Using real local app usage log from app_usage_samples.csv.")
else:
    raw_app_segments_df = create_demo_app_usage_segments(plot_df)
    st.caption("Using simulated app usage timeline based on collected focus data.")

# Keep only top 10 apps and group the rest into "Other"
app_segments_df = compress_app_segments_to_top_n(raw_app_segments_df, top_n=10)

app_timeline_fig = plot_app_usage_timeline(app_segments_df)
st.pyplot(app_timeline_fig, use_container_width=True)
plt.close(app_timeline_fig)

app_summary_df = summarize_app_usage(app_segments_df)

if not app_summary_df.empty:
    st.markdown("### Time Spent by App (Top 10 + Other)")

    summary_display_df = app_summary_df.copy()
    summary_display_df["minutes"] = summary_display_df["minutes"].round(2)
    summary_display_df["duration_seconds"] = summary_display_df["duration_seconds"].round(1)

    st.dataframe(
        summary_display_df.rename(columns={
            "app": "App",
            "duration_seconds": "Duration (seconds)",
            "minutes": "Duration (minutes)"
        }),
        use_container_width=True
    )

    google_docs_time = app_summary_df.loc[
        app_summary_df["app"] == "Google Docs",
        "duration_seconds"
    ].sum()

    total_time = app_summary_df["duration_seconds"].sum()
    other_app_time = total_time - google_docs_time

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Time on Google Docs", f"{google_docs_time / 60:.1f} min")

    with metric_col2:
        st.metric("Time on Other Apps", f"{other_app_time / 60:.1f} min")

    with metric_col3:
        task_ratio = google_docs_time / total_time if total_time > 0 else 0.0
        st.metric("Task Time Ratio", f"{task_ratio * 100:.1f}%")

    if other_app_time > 60:
        st.warning(
            "The user spent noticeable time outside the main task. "
            "This may indicate task switching or distraction."
        )
    else:
        st.success(
            "Most of the recent session was spent on the main task."
        )
else:
    st.info("No app usage timeline is available.")

# --------------------------------------------------
# App switching and split-screen tracking
# --------------------------------------------------
st.subheader("App Switching and Split-Screen Activity")

APP_USAGE_LOG = RESULTS_DIR / "app_usage_samples.csv"


def load_app_usage_summary(log_path):
    """
    Load app usage samples and estimate time spent in each app.
    """

    if not log_path.exists():
        return None, None, None

    usage_df = pd.read_csv(log_path)

    if usage_df.empty:
        return None, None, None

    usage_df["timestamp"] = pd.to_datetime(usage_df["timestamp"])
    usage_df = usage_df.sort_values("timestamp")

    # Estimate duration between samples.
    usage_df["next_timestamp"] = usage_df["timestamp"].shift(-1)
    usage_df["duration_seconds"] = (
        usage_df["next_timestamp"] - usage_df["timestamp"]
    ).dt.total_seconds()

    # Use a default duration for the latest sample.
    usage_df["duration_seconds"] = usage_df["duration_seconds"].fillna(2)

    # Avoid extreme values if the logger was paused.
    usage_df["duration_seconds"] = usage_df["duration_seconds"].clip(lower=0, upper=10)

    app_summary = (
        usage_df
        .groupby("active_app", as_index=False)["duration_seconds"]
        .sum()
        .sort_values("duration_seconds", ascending=False)
    )

    split_screen_time = usage_df.loc[
        usage_df["split_screen_detected"] == True,
        "duration_seconds"
    ].sum()

    total_time = usage_df["duration_seconds"].sum()

    return usage_df, app_summary, {
        "total_time": total_time,
        "split_screen_time": split_screen_time
    }


usage_df, app_summary, usage_stats = load_app_usage_summary(APP_USAGE_LOG)

if usage_df is None:
    st.warning(
        "No app usage data found yet. Run `python activity_logger.py` in a separate terminal to collect app switching data."
    )
else:
    total_seconds = usage_stats["total_time"]
    split_seconds = usage_stats["split_screen_time"]

    google_docs_seconds = app_summary.loc[
        app_summary["active_app"] == "Google Docs",
        "duration_seconds"
    ].sum()

    off_task_seconds = total_seconds - google_docs_seconds

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Time on Google Docs", f"{google_docs_seconds / 60:.1f} min")

    with col2:
        st.metric("Time on Other Apps", f"{off_task_seconds / 60:.1f} min")

    with col3:
        st.metric("Possible Split-Screen Time", f"{split_seconds / 60:.1f} min")

    st.markdown("### Time Spent by App")

    app_summary_display = app_summary.copy()
    app_summary_display["minutes"] = app_summary_display["duration_seconds"] / 60

    st.bar_chart(
        app_summary_display.set_index("active_app")["minutes"]
    )

    st.markdown("### Recent App Activity")

    recent_activity = usage_df[
        [
            "timestamp",
            "active_app",
            "split_screen_detected",
            "visible_other_apps"
        ]
    ].tail(10)

    st.dataframe(recent_activity, use_container_width=True)

    if off_task_seconds > 60:
        st.warning(
            "The user spent noticeable time outside Google Docs. This may indicate task switching or distraction."
        )
    else:
        st.success(
            "Most of the recent session was spent in Google Docs."
        )

    if split_seconds > 30:
        st.info(
            "Possible split-screen behavior was detected. Google Docs may have been visible while another app was also open."
        )

# Main risk line
ax.plot(
    history_df["time"],
    history_df["focus_risk_score"],
    color="#60a5fa",
    linewidth=2.8,
    label="Focus Risk"
)

# Highlight individual risk points
for _, row in history_df.iterrows():
    if row["risk_level"] == "high":
        point_color = "#ef4444"
        point_size = 70
    elif row["risk_level"] == "medium":
        point_color = "#f59e0b"
        point_size = 50
    else:
        point_color = "#22c55e"
        point_size = 32

    ax.scatter(
        row["time"],
        row["focus_risk_score"],
        color=point_color,
        s=point_size,
        zorder=3
    )

# Label high-risk moments directly on the timeline
high_risk_rows = history_df[history_df["risk_level"] == "high"]

for _, row in high_risk_rows.iterrows():
    ax.annotate(
        "Distracted",
        (row["time"], row["focus_risk_score"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        color="#ef4444",
        fontweight="bold"
    )

ax.set_title("Focus Risk Over Time")
ax.set_ylabel("Risk Score")
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.25)
ax.legend(loc="upper left")
fig.autofmt_xdate()

st.pyplot(fig, use_container_width=True)

# Show detected distraction moments in plain language
high_times = history_df[history_df["risk_level"] == "high"]["time"].dt.strftime("%H:%M").tolist()
medium_times = history_df[history_df["risk_level"] == "medium"]["time"].dt.strftime("%H:%M").tolist()

if high_times:
    st.error("High distraction moments detected at: " + " | ".join(high_times))

if medium_times:
    st.warning("Slight distraction moments detected at: " + " | ".join(medium_times))

if not high_times and not medium_times:
    st.success("No obvious distraction period was detected in the recent timeline.")

st.divider()


# --------------------------------------------------
# User-friendly explanation
# --------------------------------------------------
st.subheader("What This Means")

explain_col, action_col = st.columns(2)

with explain_col:
    st.markdown("### Why the system thinks this")
    explanation_points = get_explanation_points(features, active_app, focus_risk)

    for point in explanation_points:
        st.write(f"- {point}")

with action_col:
    st.markdown("### Suggested action")
    recommendations = get_user_recommendations(focus_risk)

    for rec in recommendations:
        st.write(f"- {rec}")

st.divider()


# --------------------------------------------------
# Simple user summary
# --------------------------------------------------
st.subheader("Simple Summary")

if focus_risk >= 0.65:
    summary = (
        "The system estimates that the user is currently highly distracted. "
        "This may be caused by unstable face position, missing face detection, "
        "or low task relevance. The user may benefit from returning to the main task."
    )
elif focus_risk >= 0.40:
    summary = (
        "The system estimates that the user is slightly distracted. "
        "The user is not fully off-task, but some visual or task-context signals "
        "suggest reduced attention stability."
    )
else:
    summary = (
        "The system estimates that the user is currently focused. "
        "The visual behavior appears stable and the task context is likely relevant."
    )

st.info(summary)

st.divider()


# --------------------------------------------------
# Technical details for developers or reports
# --------------------------------------------------
with st.expander("Technical Details", expanded=False):
    st.markdown("### Model Evaluation Outputs")

    logistic_confusion_matrix_path = RESULTS_DIR / "logistic_regression_confusion_matrix.png"
    random_forest_confusion_matrix_path = RESULTS_DIR / "random_forest_confusion_matrix.png"
    feature_importance_path = RESULTS_DIR / "random_forest_feature_importance.png"

    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        show_technical_image(
            logistic_confusion_matrix_path,
            "Logistic Regression Confusion Matrix"
        )

    with img_col2:
        show_technical_image(
            random_forest_confusion_matrix_path,
            "Random Forest Confusion Matrix"
        )

    with img_col3:
        show_technical_image(
            feature_importance_path,
            "Random Forest Feature Importance"
        )

    st.markdown(
        """
        These figures are included for technical reference.  
        The main dashboard is designed for user-facing interpretation rather than engineering analysis.
        """
    )


# --------------------------------------------------
# Final note
# --------------------------------------------------
st.caption(
    "PiFocus should be interpreted as a visual attention sensor, not a perfect cognitive focus detector."
)