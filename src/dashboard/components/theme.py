from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

NAVY = "#12243A"
BLUE = "#4B78A8"
TEAL = "#168C82"
AMBER = "#C88A25"
RED = "#B85454"
MUTED = "#718096"
GRID = "#E7ECF2"


def inject_styles() -> None:
    """Apply the shared visual system to the Streamlit application."""
    st.markdown(
        """
        <style>
        :root {
            --cr-navy: #12243A;
            --cr-ink: #202B38;
            --cr-muted: #718096;
            --cr-blue: #4B78A8;
            --cr-teal: #168C82;
            --cr-amber: #C88A25;
            --cr-red: #B85454;
            --cr-border: #E7ECF2;
            --cr-surface: #FFFFFF;
            --cr-background: #F4F6F9;
        }

        html, body, [class*="css"] {
            font-family: Inter, 'Segoe UI', sans-serif;
            color: var(--cr-ink);
        }

        .stApp {
            background: var(--cr-background);
        }

        [data-testid="stHeader"] {
            background: rgba(244, 246, 249, 0.96);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: var(--cr-background);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--cr-navy);
            letter-spacing: -0.025em;
        }

        h1 {
            font-size: clamp(1.8rem, 3vw, 2.35rem) !important;
            font-weight: 700 !important;
            margin-bottom: 0.25rem !important;
        }

        h2 {
            font-size: 1.35rem !important;
            font-weight: 650 !important;
        }

        h3 {
            font-size: 1.05rem !important;
            font-weight: 650 !important;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--cr-muted);
            font-size: 0.94rem;
        }

        [data-testid="stSidebar"] {
            background: var(--cr-navy);
            border-right: 1px solid #20354B;
        }

        [data-testid="stSidebar"] * {
            color: #EAF0F6;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            color: #FFFFFF;
            font-size: 1.08rem !important;
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: #EAF0F6;
        }

        [data-testid="stSidebarNav"] {
            padding-top: 0.6rem;
        }

        [data-testid="stSidebarNav"] a {
            border-radius: 6px;
            margin: 0.15rem 0.5rem;
            transition: background-color 120ms ease;
        }

        [data-testid="stSidebarNav"] a:hover {
            background: rgba(255, 255, 255, 0.10);
        }

        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(75, 120, 168, 0.38);
            border-left: 3px solid #7FB5D5;
        }

        [data-testid="stMetric"] {
            background: var(--cr-surface);
            border: 1px solid var(--cr-border);
            border-top: 3px solid var(--cr-blue);
            border-radius: 8px;
            min-height: 116px;
            padding: 0.95rem 0.78rem;
            box-shadow: 0 2px 8px rgba(18, 36, 58, 0.035);
        }

        [data-testid="stMetricLabel"] {
            color: var(--cr-muted);
            font-size: 0.76rem;
            font-weight: 600;
        }

        [data-testid="stMetricLabel"] p {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
        }

        [data-testid="stMetricValue"] {
            color: var(--cr-navy);
            font-size: 1.65rem;
            font-weight: 700;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--cr-surface);
            border-color: var(--cr-border);
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(18, 36, 58, 0.035);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            overflow: hidden;
            background: var(--cr-surface);
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label {
            color: var(--cr-navy);
            font-weight: 600;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border-color: var(--cr-border);
            border-radius: 6px;
            background: var(--cr-surface);
        }

        [data-testid="stAlert"] {
            border-radius: 7px;
            border-width: 1px;
        }

        [data-testid="stProgress"] > div > div {
            background: var(--cr-teal);
        }

        div[data-testid="stPlotlyChart"] {
            background: var(--cr-surface);
            border: 1px solid var(--cr-border);
            border-radius: 8px;
            padding: 0.55rem 0.7rem 0.2rem;
            box-shadow: 0 2px 8px rgba(18, 36, 58, 0.035);
        }

        @media (max-width: 768px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1.25rem 1rem 2rem;
            }

            [data-testid="stMetric"] {
                min-height: 96px;
                padding: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, *, height: int = 360) -> go.Figure:
    """Apply consistent typography, grid, and palette to an existing Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#202B38", "size": 12},
        title={"font": {"size": 16, "color": NAVY}, "x": 0.02, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 18, "r": 18, "t": 58, "b": 24},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        colorway=[BLUE, TEAL, AMBER, RED],
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont={"color": MUTED})
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont={"color": MUTED})
    return fig
