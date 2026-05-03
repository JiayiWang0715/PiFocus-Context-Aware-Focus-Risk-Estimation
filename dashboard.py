#!/usr/bin/env python3
"""
Local Streamlit dashboard for PiFocus context-aware focus risk estimation.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("data/fused_focus_log.csv")
REQUIRED_COLUMNS = [
    "timestamp",
    "visual_attention_prob",
    "visual_attention_state",
    "task_relevance_score",
    "risk_score",
    "smoothed_risk_score",
    "final_state",
    "active_app",
    "window_title",
    "task_mode",
    "idle_seconds",
]
STATE_LABELS = {
    "focused_on_task": "🟢 Focused on Task",
    "present_but_off_task": "🟡 Present but Off Task",
    "absent_or_disengaged": "🔴 Absent / Disengaged",
    "uncertain": "🟠 Uncertain",
}
TIMELINE_COLUMNS = [
    "visual_attention_prob",
    "task_relevance_score",
    "smoothed_risk_score",
]
STATE_TIME_COLUMNS = {
    "focused_on_task": "Effective Focus Time",
    "present_but_off_task": "Off-task Time",
    "absent_or_disengaged": "Absent Time",
}

# --------------------------------------------------
# Focus alert configuration
# --------------------------------------------------
FOCUS_ALERT_WINDOW_SECONDS = 300   # 5-minute rolling window
FOCUS_ALERT_RED_SECONDS = 10       # Red alert when distraction reaches 60 seconds
FOCUS_ALERT_YELLOW_SECONDS = 5    # Yellow warning when distraction reaches 30 seconds

DISTRACTED_STATES = {
    "present_but_off_task",
    "absent_or_disengaged",
}

st.set_page_config(page_title="PiFocus Dashboard", layout="wide")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            display: none;
        }
        .stApp {
            background: #eef3f8;
            color: #142033;
        }
        [data-testid="stSidebar"] {
            background: #dde7f2;
            border-right: 1px solid #c7d4e4;
        }
        .main .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3, p, span, label {
            color: #142033;
        }
        h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        .privacy-caption {
            color: #55657a;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .metric-card,
        .context-card {
            background: #ffffff;
            border: 1px solid #d6e0ec;
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(39, 57, 81, 0.08);
        }
        .metric-card {
            min-height: 96px;
            padding: 0.85rem 0.9rem;
        }
        .progress-label {
            color: #223049;
            font-size: 0.95rem;
            font-weight: 850;
            margin: 0.8rem 0 0.35rem;
        }
        .metric-label {
            color: #5a6b80;
            font-size: 0.8rem;
            font-weight: 750;
            line-height: 1.15;
            margin-bottom: 0.35rem;
            white-space: nowrap;
        }
        .metric-value {
            color: #152033;
            font-size: clamp(1.08rem, 1.7vw, 1.48rem);
            font-weight: 850;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .state-focused {
            border-top: 5px solid #16a66a;
        }
        .state-off-task,
        .state-uncertain {
            border-top: 5px solid #f0b429;
        }
        .state-absent {
            border-top: 5px solid #df4b55;
        }
        .context-card {
            padding: 0.95rem 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .context-title {
            color: #223049;
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 0.75rem;
        }
        .context-grid {
            display: grid;
            grid-template-columns: 1fr 1.7fr 0.8fr 0.8fr;
            gap: 0.75rem;
        }
        .context-label {
            color: #66758b;
            font-size: 0.75rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .context-value {
            color: #152033;
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .logger-warning {
            background: #fff5d8;
            border: 1px solid #edc85d;
            border-radius: 6px;
            color: #5d4300;
            font-size: 0.9rem;
            margin-top: 0.75rem;
            padding: 0.55rem 0.65rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
        }
        @media (max-width: 900px) {
            .context-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "data/fused_focus_log.csv is missing required columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in [
        "visual_attention_prob",
        "task_relevance_score",
        "risk_score",
        "smoothed_risk_score",
        "idle_seconds",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for optional_column in ["elapsed_seconds", "elapsed_seconds_x"]:
        if optional_column in df.columns:
            df[optional_column] = pd.to_numeric(df[optional_column], errors="coerce")

    elapsed_column = get_elapsed_column(df)
    if elapsed_column is None:
        raise ValueError(
            "data/fused_focus_log.csv is missing elapsed_seconds or elapsed_seconds_x"
        )

    df["display_elapsed_seconds"] = pd.to_numeric(df[elapsed_column], errors="coerce")
    df = df.sort_values("display_elapsed_seconds").reset_index(drop=True)
    df["display_active_app"] = df["active_app"].apply(display_app)
    df["display_final_state"] = df["final_state"].map(STATE_LABELS).fillna("🟠 Uncertain")
    return df


def get_elapsed_column(df: pd.DataFrame) -> str | None:
    if "elapsed_seconds" in df.columns:
        return "elapsed_seconds"
    if "elapsed_seconds_x" in df.columns:
        return "elapsed_seconds_x"
    return None


def display_app(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text or text.upper() == "ERROR":
        return "Unknown / Logger Error"
    return text


def format_percent(value: object) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "N/A"


def format_score(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_duration(seconds: object) -> str:
    try:
        numeric_seconds = float(seconds)
        if pd.isna(numeric_seconds):
            numeric_seconds = 0
        total_seconds = max(0, int(round(numeric_seconds)))
    except (TypeError, ValueError):
        total_seconds = 0
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def state_class(state: object) -> str:
    return {
        "focused_on_task": "state-focused",
        "present_but_off_task": "state-off-task",
        "uncertain": "state-uncertain",
        "absent_or_disengaged": "state-absent",
    }.get(str(state), "state-uncertain")


def metric_card(label: str, value: str, css_class: str = "") -> None:
    safe_label = escape(label)
    safe_value = escape(value)
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value" title="{safe_value}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_card(row: pd.Series) -> None:
    app = display_app(row.get("active_app"))
    window_title = "" if pd.isna(row.get("window_title")) else str(row.get("window_title"))
    task_mode = "" if pd.isna(row.get("task_mode")) else str(row.get("task_mode"))
    idle_seconds = format_score(row.get("idle_seconds"))
    has_logger_error = app == "Unknown / Logger Error"

    warning = ""
    if has_logger_error:
        warning = (
            '<div class="logger-warning">'
            "Activity logger could not read active window for this row."
            "</div>"
        )

    st.markdown(
        f"""
        <div class="context-card">
            <div class="context-title">Current Activity Context</div>
            <div class="context-grid">
                <div>
                    <div class="context-label">active_app</div>
                    <div class="context-value">{escape(app)}</div>
                </div>
                <div>
                    <div class="context-label">window_title</div>
                    <div class="context-value">{escape(window_title or "Untitled")}</div>
                </div>
                <div>
                    <div class="context-label">task_mode</div>
                    <div class="context-value">{escape(task_mode or "Unknown")}</div>
                </div>
                <div>
                    <div class="context-label">idle_seconds</div>
                    <div class="context-value">{escape(idle_seconds)}</div>
                </div>
            </div>
            {warning}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_app_usage_segments(df: pd.DataFrame) -> pd.DataFrame:
    app_df = df[["display_elapsed_seconds", "display_active_app"]].copy()
    app_df["segment_id"] = app_df["display_active_app"].ne(
        app_df["display_active_app"].shift()
    ).cumsum()

    rows = []
    for _, segment in app_df.groupby("segment_id", sort=True):
        start = segment["display_elapsed_seconds"].iloc[0]
        end = segment["display_elapsed_seconds"].iloc[-1]
        if len(segment) > 1 and pd.notna(start) and pd.notna(end):
            duration = max(float(end) - float(start) + 1.0, 1.0)
        else:
            duration = 1.0

        rows.append(
            {
                "start": format_duration(start),
                "end": format_duration(end),
                "active_app": segment["display_active_app"].iloc[0],
                "duration_seconds": round(duration, 1),
                "minutes": round(duration / 60, 2),
            }
        )

    return pd.DataFrame(rows)


def recent_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "display_elapsed",
        "display_active_app",
        "window_title",
        "task_mode",
        "visual_attention_prob",
        "task_relevance_score",
        "smoothed_risk_score",
        "display_final_state",
    ]
    return [column for column in columns if column and column in df.columns]


def state_duration_seconds(df: pd.DataFrame, state: str) -> float:
    if df.empty:
        return 0.0

    sorted_df = df.sort_values("display_elapsed_seconds").reset_index(drop=True)
    elapsed = sorted_df["display_elapsed_seconds"]
    deltas = elapsed.shift(-1) - elapsed
    fallback_delta = deltas[deltas > 0].median()
    if pd.isna(fallback_delta) or fallback_delta <= 0:
        fallback_delta = 1.0
    deltas = deltas.fillna(fallback_delta).clip(lower=0)
    return float(deltas[sorted_df["final_state"] == state].sum())

def recent_distracted_seconds(
    df: pd.DataFrame,
    selected_elapsed: float,
    window_seconds: float = FOCUS_ALERT_WINDOW_SECONDS,
) -> float:
    """
    Estimate accumulated distracted time within a recent rolling window.

    Distracted time is computed from final_state:
    - present_but_off_task
    - absent_or_disengaged

    When this accumulated time reaches FOCUS_ALERT_RED_SECONDS,
    the alert progress bar becomes full and triggers a red alert.
    """

    if df.empty:
        return 0.0

    sorted_df = df.sort_values("display_elapsed_seconds").reset_index(drop=True)

    current_time = float(selected_elapsed or 0)
    start_time = max(0.0, current_time - window_seconds)

    window_df = sorted_df[
        (sorted_df["display_elapsed_seconds"] >= start_time)
        & (sorted_df["display_elapsed_seconds"] <= current_time)
    ].copy()

    if window_df.empty:
        return 0.0

    elapsed = window_df["display_elapsed_seconds"]
    deltas = elapsed.shift(-1) - elapsed

    fallback_delta = sorted_df["display_elapsed_seconds"].diff()
    fallback_delta = fallback_delta[fallback_delta > 0].median()

    if pd.isna(fallback_delta) or fallback_delta <= 0:
        fallback_delta = 1.0

    deltas = deltas.fillna(fallback_delta).clip(lower=0, upper=10)

    distracted_mask = window_df["final_state"].isin(DISTRACTED_STATES)

    return float(deltas[distracted_mask].sum())


def render_focus_alert_progress(distracted_seconds: float) -> None:
    """
    Render a user-facing focus alert progress bar.

    The bar fills as distracted time accumulates in the recent 5-minute window.
    When it reaches 100%, the UI shows a red alert.
    """

    progress = min(max(distracted_seconds / FOCUS_ALERT_RED_SECONDS, 0.0), 1.0)
    progress_percent = progress * 100
    remaining_seconds = max(FOCUS_ALERT_RED_SECONDS - distracted_seconds, 0.0)

    if distracted_seconds >= FOCUS_ALERT_RED_SECONDS:
        status_text = "Red Alert Triggered"
        status_icon = "🔴"
        bar_color = "#df4b55"
        background_color = "#fde8ea"
        border_color = "#df4b55"
        message = (
            "The user has spent too much time off task or disengaged in the recent "
            "5-minute window. Please return to the main task."
        )
    elif distracted_seconds >= FOCUS_ALERT_YELLOW_SECONDS:
        status_text = "Warning Zone"
        status_icon = "🟡"
        bar_color = "#f0b429"
        background_color = "#fff5d8"
        border_color = "#f0b429"
        message = (
            f"The user shows signs of distraction. "
            f"{remaining_seconds:.0f} seconds remain before the red alert."
        )
    else:
        status_text = "Focus Looks Stable"
        status_icon = "🟢"
        bar_color = "#16a66a"
        background_color = "#e9f8f0"
        border_color = "#16a66a"
        message = (
            f"The user is currently within the focused range. "
            f"{remaining_seconds:.0f} seconds remain before the red alert."
        )

    st.markdown(
        f"""
        <div style="
            background: {background_color};
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin: 1rem 0 1rem 0;
            box-shadow: 0 8px 22px rgba(39, 57, 81, 0.08);
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.65rem;
            ">
                <div style="
                    color: #142033;
                    font-size: 1.05rem;
                    font-weight: 850;
                ">
                    {status_icon} Focus Alert Progress
                </div>
                <div style="
                    color: {bar_color};
                    font-size: 1rem;
                    font-weight: 850;
                ">
                    {status_text} — {progress_percent:.0f}%
                </div>
            </div>

            <div style="
                width: 100%;
                height: 24px;
                background: rgba(20, 32, 51, 0.12);
                border-radius: 999px;
                overflow: hidden;
            ">
                <div style="
                    width: {progress_percent:.1f}%;
                    height: 100%;
                    background: {bar_color};
                    border-radius: 999px;
                    transition: width 0.4s ease;
                "></div>
            </div>

            <div style="
                display: flex;
                justify-content: space-between;
                margin-top: 0.45rem;
                color: #55657a;
                font-size: 0.82rem;
                font-weight: 700;
            ">
                <span>0 sec</span>
                <span>Yellow: {FOCUS_ALERT_YELLOW_SECONDS} sec</span>
                <span>Red: {FOCUS_ALERT_RED_SECONDS} sec</span>
            </div>

            <div style="
                margin-top: 0.75rem;
                color: #142033;
                font-size: 0.92rem;
                font-weight: 700;
            ">
                Distracted time in recent 5 minutes:
                {distracted_seconds:.1f} seconds
            </div>

            <div style="
                margin-top: 0.35rem;
                color: #334155;
                font-size: 0.9rem;
            ">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def main() -> None:
    apply_styles()

    st.title("PiFocus: Context-Aware Focus Risk Estimation")
    st.markdown(
        '<div class="privacy-caption">'
        "This dashboard visualizes local fused outputs. It does not store raw video, "
        "screenshots, keystrokes, or audio."
        "</div>",
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error("Run python src/fusion_engine.py first.")
        return

    st.sidebar.header("Controls")
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        rerun_app()
    target_minutes = st.sidebar.number_input(
        "Target work duration in minutes",
        min_value=1,
        max_value=240,
        value=45,
        step=5,
    )

    try:
        df = load_data(DATA_PATH)
    except ValueError as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.error("data/fused_focus_log.csv exists, but it has no rows to display.")
        return

    selected_index = st.sidebar.slider(
        "Select row index",
        min_value=0,
        max_value=len(df) - 1,
        value=len(df) - 1,
        step=1,
    )
    selected_row = df.iloc[selected_index]
    selected_elapsed = selected_row.get("display_elapsed_seconds", 0)
    target_duration_seconds = float(target_minutes) * 60
    progress_value = min(
        max(float(selected_elapsed or 0) / target_duration_seconds, 0.0),
        1.0,
    )

    st.sidebar.caption("Selected row")
    st.sidebar.write(f"Selected moment: {format_duration(selected_elapsed)}")
    st.sidebar.write(f"Current app: {display_app(selected_row.get('active_app'))}")
    st.sidebar.write(f"State: {selected_row.get('display_final_state', '🟠 Uncertain')}")

    progress_columns = st.columns(4)
    with progress_columns[0]:
        metric_card(
            "Session Progress",
            f"{format_duration(selected_elapsed)} / {format_duration(target_duration_seconds)}",
        )
    with progress_columns[1]:
        metric_card(
            STATE_TIME_COLUMNS["focused_on_task"],
            format_duration(state_duration_seconds(df, "focused_on_task")),
        )
    with progress_columns[2]:
        metric_card(
            STATE_TIME_COLUMNS["present_but_off_task"],
            format_duration(state_duration_seconds(df, "present_but_off_task")),
        )
    with progress_columns[3]:
        metric_card(
            STATE_TIME_COLUMNS["absent_or_disengaged"],
            format_duration(state_duration_seconds(df, "absent_or_disengaged")),
        )
    st.markdown(
        '<div class="progress-label">Progress toward target work duration</div>',
        unsafe_allow_html=True,
    )
    st.progress(progress_value)

    card_columns = st.columns(4)
    with card_columns[0]:
        metric_card(
            "🚦 Final State",
            str(selected_row.get("display_final_state", "🟠 Uncertain")),
            state_class(selected_row.get("final_state")),
        )
    with card_columns[1]:
        metric_card("👁 Visual Attention", format_percent(selected_row.get("visual_attention_prob")))
    with card_columns[2]:
        metric_card("🧭 Task Relevance", format_score(selected_row.get("task_relevance_score")))
    with card_columns[3]:
        metric_card("⚠️ Smoothed Risk", format_score(selected_row.get("smoothed_risk_score")))
    
        focus_alert_seconds = recent_distracted_seconds(
        df=df,
        selected_elapsed=float(selected_elapsed or 0),
    )

    render_focus_alert_progress(focus_alert_seconds)

    context_card(selected_row)

    hart_df = df.copy()
    chart_df = chart_df.set_index("display_elapsed_seconds")

    usage_summary = (
        segments.groupby("active_app", as_index=False)["duration_seconds"]
        .sum()
        .sort_values("duration_seconds", ascending=False)
    )
    usage_summary["minutes"] = (usage_summary["duration_seconds"] / 60).round(2)
    st.dataframe(
        usage_summary[["active_app", "duration_seconds", "minutes"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Recent Activity")
    recent = df.tail(20).copy().sort_index(ascending=False)
    recent["display_elapsed"] = recent["display_elapsed_seconds"].apply(format_duration)
    display_columns = recent_columns(recent)
    rename_columns = {
        "display_active_app": "active_app",
        "display_final_state": "final_state",
    }
    st.dataframe(
        recent[display_columns].rename(columns=rename_columns),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
