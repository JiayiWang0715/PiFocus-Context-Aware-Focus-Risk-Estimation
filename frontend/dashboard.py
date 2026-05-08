#!/usr/bin/env python3
"""
Local Streamlit dashboard for PiFocus context-aware focus risk estimation.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


DATA_PATH = Path("data/fused_focus_log.csv")
REQUIRED_COLUMNS = [
    "timestamp",
    "visual_attention_prob",
    "visual_attention_state",
    "task_relevance_score",
    "risk_score",
    "smoothed_risk_score",
    "instantaneous_risk_score",
    "accumulated_risk_score",
    "risk_level",
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
    "accumulated_risk_score",
]
STATE_TIME_COLUMNS = {
    "focused_on_task": "Effective Focus Time",
    "present_but_off_task": "Off-task Time",
    "absent_or_disengaged": "Absent Time",
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
        .section-label {
            color: #43546c;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0.35rem 0 0.55rem;
            text-transform: uppercase;
        }
        .metric-card,
        .context-card {
            background: #ffffff;
            border: 1px solid #d6e0ec;
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(39, 57, 81, 0.08);
        }
        .metric-card {
            min-height: 104px;
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
        }
        .metric-value {
            color: #152033;
            font-size: clamp(1.08rem, 1.7vw, 1.48rem);
            font-weight: 850;
            line-height: 1.15;
            overflow-wrap: anywhere;
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
        .risk-low {
            border-top: 5px solid #16a66a;
        }
        .risk-caution {
            border-top: 5px solid #f0b429;
        }
        .risk-high {
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
        .risk-banner {
            border-radius: 8px;
            font-size: 1.02rem;
            font-weight: 850;
            line-height: 1.3;
            margin: 0.35rem 0 1rem;
            padding: 0.85rem 1rem;
        }
        .risk-banner-high {
            background: #fff1f1;
            border: 1px solid #df4b55;
            border-left: 6px solid #df4b55;
            box-shadow: 0 8px 22px rgba(129, 28, 35, 0.12);
            color: #7a1f27;
        }
        .risk-banner-caution {
            background: #fff8e6;
            border: 1px solid #f0b429;
            border-left: 6px solid #f0b429;
            box-shadow: 0 8px 22px rgba(138, 92, 13, 0.1);
            color: #6f4a00;
        }
        .risk-banner-low {
            background: #eef8ff;
            border: 1px solid #6bb7d9;
            border-left: 6px solid #2f9f74;
            box-shadow: 0 8px 22px rgba(30, 105, 129, 0.08);
            color: #17485e;
        }
        .playback-status {
            background: #ffffff;
            border: 1px solid #d6e0ec;
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(39, 57, 81, 0.08);
            color: #223049;
            font-size: 1rem;
            font-weight: 850;
            margin: 0.25rem 0 1rem;
            padding: 0.75rem 0.9rem;
        }
        .state-band {
            align-items: stretch;
            background: #ffffff;
            border: 1px solid #d6e0ec;
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(39, 57, 81, 0.08);
            display: flex;
            height: 34px;
            margin-bottom: 0.55rem;
            overflow: hidden;
            width: 100%;
        }
        .state-band-segment {
            min-width: 5px;
        }
        .band-focused {
            background: #16a66a;
        }
        .band-off-task {
            background: #f0b429;
        }
        .band-absent {
            background: #df4b55;
        }
        .band-uncertain {
            background: #e88633;
        }
        .state-legend {
            color: #55657a;
            display: flex;
            flex-wrap: wrap;
            font-size: 0.82rem;
            font-weight: 700;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .legend-dot {
            border-radius: 999px;
            display: inline-block;
            height: 0.65rem;
            margin-right: 0.25rem;
            width: 0.65rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 800;
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
        "instantaneous_risk_score",
        "accumulated_risk_score",
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
    df["display_window_title"] = df["window_title"].apply(
        lambda value: display_text(value, "Untitled")
    )
    df["display_task_mode"] = df["task_mode"].apply(display_text)
    df["display_idle_seconds"] = df["idle_seconds"].apply(format_idle_seconds)
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


def display_text(value: object, fallback: str = "Unknown") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nat":
        return fallback
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


def format_idle_seconds(value: object) -> str:
    try:
        numeric_seconds = float(value)
        if pd.isna(numeric_seconds):
            return "N/A"
        return f"{max(0, int(round(numeric_seconds)))}s"
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


def section_label(text: str) -> None:
    st.markdown(
        f'<div class="section-label">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def state_class(state: object) -> str:
    return {
        "focused_on_task": "state-focused",
        "present_but_off_task": "state-off-task",
        "uncertain": "state-uncertain",
        "absent_or_disengaged": "state-absent",
    }.get(str(state), "state-uncertain")


def risk_class(level: object) -> str:
    normalized = display_text(level).lower()
    if normalized == "low risk":
        return "risk-low"
    if normalized == "caution":
        return "risk-caution"
    if normalized == "high risk":
        return "risk-high"
    return "risk-caution"


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


def risk_level_banner(level: object) -> None:
    normalized = display_text(level).lower()
    if normalized == "high risk":
        message = "⚠ High focus risk detected. Consider returning to the task or taking a planned break."
        css_class = "risk-banner-high"
    elif normalized == "caution":
        message = "⚠ Focus risk is increasing. The system detected sustained off-task or uncertain behavior."
        css_class = "risk-banner-caution"
    elif normalized == "low risk":
        message = "Focus context looks stable."
        css_class = "risk-banner-low"
    else:
        return

    st.markdown(
        f'<div class="risk-banner {css_class}">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def state_band_class(state: object) -> str:
    return {
        "focused_on_task": "band-focused",
        "present_but_off_task": "band-off-task",
        "absent_or_disengaged": "band-absent",
        "uncertain": "band-uncertain",
    }.get(str(state), "band-uncertain")


def state_timeline(df: pd.DataFrame) -> None:
    if df.empty:
        return

    sorted_df = df.sort_values("display_elapsed_seconds").reset_index(drop=True)
    deltas = (
        sorted_df["display_elapsed_seconds"].shift(-1)
        - sorted_df["display_elapsed_seconds"]
    ).fillna(1).clip(lower=0, upper=5)
    total_duration = max(float(deltas.sum()), 1.0)

    segments = []
    for index, row in sorted_df.iterrows():
        width = max(float(deltas.iloc[index]) / total_duration * 100, 0.8)
        state = str(row.get("final_state", "uncertain"))
        label = STATE_LABELS.get(state, STATE_LABELS["uncertain"])
        segments.append(
            '<div class="state-band-segment '
            f'{state_band_class(state)}" '
            f'style="width: {width:.3f}%;" '
            f'title="{escape(label)}"></div>'
        )

    st.markdown(
        f"""
        <div class="state-band">{''.join(segments)}</div>
        <div class="state-legend">
            <span><span class="legend-dot band-focused"></span>Focused</span>
            <span><span class="legend-dot band-off-task"></span>Off Task</span>
            <span><span class="legend-dot band-absent"></span>Absent</span>
            <span><span class="legend-dot band-uncertain"></span>Uncertain</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_card(row: pd.Series) -> None:
    app = display_app(row.get("active_app"))
    window_title = display_text(row.get("window_title"), "Untitled")
    task_mode = display_text(row.get("task_mode"))
    idle_seconds = format_idle_seconds(row.get("idle_seconds"))
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
                    <div class="context-label">Current App</div>
                    <div class="context-value">{escape(app)}</div>
                </div>
                <div>
                    <div class="context-label">Window Title</div>
                    <div class="context-value">{escape(window_title)}</div>
                </div>
                <div>
                    <div class="context-label">Task Mode</div>
                    <div class="context-value">{escape(task_mode)}</div>
                </div>
                <div>
                    <div class="context-label">Idle Seconds</div>
                    <div class="context-value">{escape(idle_seconds)}</div>
                </div>
            </div>
            {warning}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_app_usage_segments(df: pd.DataFrame) -> pd.DataFrame:
    segment_columns = ["start", "end", "active_app", "duration_seconds", "minutes"]
    if df.empty:
        return pd.DataFrame(columns=segment_columns)

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

    return pd.DataFrame(rows, columns=segment_columns)


def recent_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "display_elapsed",
        "display_active_app",
        "display_window_title",
        "display_task_mode",
        "visual_attention_prob",
        "task_relevance_score",
        "accumulated_risk_score",
        "risk_level",
        "display_final_state",
    ]
    return [column for column in columns if column and column in df.columns]


def state_duration_seconds(df: pd.DataFrame, state: str) -> float:
    if df.empty:
        return 0.0

    sorted_df = df.sort_values("display_elapsed_seconds").reset_index(drop=True)
    elapsed = sorted_df["display_elapsed_seconds"]
    deltas = (elapsed.shift(-1) - elapsed).fillna(0).clip(lower=0, upper=5)
    return float(deltas[sorted_df["final_state"] == state].sum())


def session_duration_seconds(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    elapsed = pd.to_numeric(df["display_elapsed_seconds"], errors="coerce").dropna()
    if elapsed.empty:
        return 0.0
    return float(max(0, elapsed.max()))


def refresh_dashboard() -> None:
    if st_autorefresh is not None:
        st_autorefresh(interval=1000, key="pifocus_live_monitor_refresh")
        return


def current_session_time(df: pd.DataFrame) -> float:
    return session_duration_seconds(df)


def session_time_status(session_time: float) -> None:
    st.markdown(
        f'<div class="playback-status">Session Time: {format_duration(session_time)}</div>',
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

    refresh_dashboard()
    fallback_refresh = st_autorefresh is None

    try:
        df = load_data(DATA_PATH)
    except ValueError as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.error("data/fused_focus_log.csv exists, but it has no rows to display.")
        return

    display_df = df
    selected_index = len(display_df) - 1
    selected_row = display_df.iloc[selected_index]
    selected_elapsed = selected_row.get("display_elapsed_seconds", 0)
    summary_df = display_df.copy()
    chart_df_source = display_df

    session_time_status(current_session_time(display_df))

    st.sidebar.caption("Selected row")
    st.sidebar.write(f"Session time: {format_duration(selected_elapsed)}")
    st.sidebar.write(f"Current app: {display_app(selected_row.get('active_app'))}")
    st.sidebar.write(f"State: {selected_row.get('display_final_state', '🟠 Uncertain')}")

    section_label("Session Summary")
    progress_columns = st.columns(4)
    with progress_columns[0]:
        metric_card(
            "Session Time",
            f"Session time: {format_duration(selected_elapsed)}",
        )
    with progress_columns[1]:
        metric_card(
            STATE_TIME_COLUMNS["focused_on_task"],
            format_duration(state_duration_seconds(summary_df, "focused_on_task")),
        )
    with progress_columns[2]:
        metric_card(
            STATE_TIME_COLUMNS["present_but_off_task"],
            format_duration(state_duration_seconds(summary_df, "present_but_off_task")),
        )
    with progress_columns[3]:
        metric_card(
            STATE_TIME_COLUMNS["absent_or_disengaged"],
            format_duration(state_duration_seconds(summary_df, "absent_or_disengaged")),
        )

    context_card(selected_row)
    risk_level_banner(selected_row.get("risk_level"))

    section_label("Current State")
    card_columns = st.columns(5)
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
        metric_card(
            "⚠️ Accumulated Risk",
            format_score(selected_row.get("accumulated_risk_score")),
        )
    with card_columns[4]:
        risk_level_value = display_text(selected_row.get("risk_level"))
        metric_card("Risk Level", risk_level_value, risk_class(risk_level_value))

    chart_df = chart_df_source.copy()
    chart_df = chart_df.set_index("display_elapsed_seconds")

    st.subheader("Accumulated Risk Timeline")
    st.line_chart(chart_df[TIMELINE_COLUMNS], height=300)

    st.subheader("State Timeline")
    state_timeline(chart_df_source)

    st.subheader("App Usage Timeline")
    app_usage_columns = ["start", "end", "active_app", "duration_seconds", "minutes"]
    segments = build_app_usage_segments(chart_df_source)
    segments = segments.reindex(columns=app_usage_columns)
    segments["duration_seconds"] = pd.to_numeric(
        segments["duration_seconds"], errors="coerce"
    ).fillna(0.0)
    segments["minutes"] = pd.to_numeric(segments["minutes"], errors="coerce").fillna(0.0)
    st.dataframe(
        segments[app_usage_columns],
        use_container_width=True,
        hide_index=True,
    )

    if segments.empty:
        usage_summary = pd.DataFrame(
            columns=["active_app", "duration_seconds", "minutes"]
        )
    else:
        usage_summary = (
            segments.groupby("active_app", as_index=False)["duration_seconds"]
            .sum()
            .sort_values("duration_seconds", ascending=False)
        )
        usage_summary["duration_seconds"] = pd.to_numeric(
            usage_summary["duration_seconds"], errors="coerce"
        ).fillna(0.0)
        usage_summary["minutes"] = (usage_summary["duration_seconds"] / 60).round(2)
    st.dataframe(
        usage_summary[["active_app", "duration_seconds", "minutes"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Recent Activity")
    recent = chart_df_source.tail(20).copy().sort_index(ascending=False)
    recent["display_elapsed"] = recent["display_elapsed_seconds"].apply(format_duration)
    display_columns = recent_columns(recent)
    rename_columns = {
        "display_active_app": "active_app",
        "display_window_title": "window_title",
        "display_task_mode": "task_mode",
        "display_final_state": "final_state",
        "accumulated_risk_score": "accumulated_risk",
    }
    st.dataframe(
        recent[display_columns].rename(columns=rename_columns),
        use_container_width=True,
        hide_index=True,
    )

    if fallback_refresh:
        import time

        time.sleep(1)
        rerun_app()


if __name__ == "__main__":
    main()
