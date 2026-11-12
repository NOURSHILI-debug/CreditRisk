"""Shared visual tokens and restrained Streamlit component styling."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

NAVY = "#12243A"
INK = "#202B38"
MUTED = "#566474"
BLUE = "#315B78"
TEAL = "#176B5B"
AMBER = "#795400"
RED = "#9D3434"
NEUTRAL = "#566474"
GRID = "#D9E0E7"
SURFACE = "#FFFFFF"
BACKGROUND = "#F4F6F8"


def inject_styles() -> None:
    """Apply application tokens and small, focused Streamlit CSS overrides.

    Streamlit does not expose all component colors and widths as theme tokens.
    The few DOM selectors below are intentionally scoped to dashboard regions;
    verify them when upgrading Streamlit because its frontend markup can change.
    """
    st.markdown(
        f"""
        <style>
        :root {{
            --cr-navy: {NAVY};
            --cr-ink: {INK};
            --cr-muted: {MUTED};
            --cr-blue: {BLUE};
            --cr-teal: {TEAL};
            --cr-amber: {AMBER};
            --cr-red: {RED};
            --cr-neutral: {NEUTRAL};
            --cr-border: {GRID};
            --cr-surface: {SURFACE};
            --cr-background: {BACKGROUND};
            --cr-space-1: 0.4rem;
            --cr-space-2: 0.75rem;
            --cr-space-3: 1rem;
            --cr-space-4: 1.5rem;
            --cr-space-5: 2rem;
        }}

        html, body, [class*="css"] {{
            font-family: Inter, 'Segoe UI', sans-serif;
            color: var(--cr-ink);
        }}
        .stApp {{ background: var(--cr-background); }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1440px;
            padding-top: var(--cr-space-4);
            padding-bottom: var(--cr-space-5);
        }}
        h1, h2, h3 {{
            color: var(--cr-navy);
            letter-spacing: -0.015em;
        }}
        h1 {{
            font-size: clamp(1.65rem, 2.5vw, 2rem) !important;
            font-weight: 680 !important;
            margin-bottom: var(--cr-space-1) !important;
        }}
        h2 {{ font-size: 1.3rem !important; font-weight: 650 !important; }}
        h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; }}
        [data-testid="stCaptionContainer"] {{
            color: var(--cr-muted);
            font-size: 0.92rem;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--cr-navy);
            border-right: 1px solid #20354B;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: #F2F5F8;
        }}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: #CBD6DF;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
            color: #FFFFFF;
            font-size: 1rem !important;
            letter-spacing: 0;
        }}
        [data-testid="stSidebarNav"] {{ padding-top: var(--cr-space-2); }}
        [data-testid="stSidebarNav"] a {{
            border-radius: 4px;
            margin: 0.1rem 0.4rem;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(255, 255, 255, 0.10);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(255, 255, 255, 0.12);
            border-left: 3px solid #A8C4D4;
        }}

        [data-testid="stMetric"] {{
            background: var(--cr-surface);
            border: 1px solid var(--cr-border);
            border-top: 2px solid var(--cr-blue);
            border-radius: 5px;
            min-height: 108px;
            padding: 0.85rem 0.8rem;
            box-shadow: none;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--cr-muted);
            font-size: 0.82rem;
            font-weight: 600;
        }}
        [data-testid="stMetricLabel"] p {{ white-space: normal; }}
        [data-testid="stMetricValue"] {{
            color: var(--cr-navy);
            font-size: 1.55rem;
            font-weight: 680;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--cr-surface);
            border-color: var(--cr-border);
            border-radius: 5px;
            box-shadow: none;
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--cr-border);
            border-radius: 4px;
            background: var(--cr-surface);
        }}
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stCheckbox"] label {{
            color: var(--cr-ink);
            font-weight: 600;
        }}
        div[data-testid="stPlotlyChart"] {{
            background: var(--cr-surface);
            border: 1px solid var(--cr-border);
            border-radius: 5px;
            padding: 0.25rem 0.45rem;
        }}
        [data-testid="stChatMessage"] {{
            max-width: 960px;
            margin-inline: auto;
        }}
        [data-testid="stChatInput"] {{ max-width: 960px; margin-inline: auto; }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
            line-height: 1.6;
            overflow-wrap: anywhere;
        }}

        @media (max-width: 768px) {{
            [data-testid="stMainBlockContainer"] {{
                padding: 1rem 0.85rem 1.5rem;
            }}
            [data-testid="stMetric"] {{
                min-height: 92px;
                padding: 0.7rem;
            }}
            [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, *, height: int = 350) -> go.Figure:
    """Apply consistent typography, grid, margins, and finance-safe colors."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Inter, Segoe UI, sans-serif", "color": INK, "size": 12},
        title={"font": {"size": 15, "color": NAVY}, "x": 0.02, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 24, "r": 18, "t": 58, "b": 38},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        colorway=[BLUE, TEAL, AMBER, RED],
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont={"color": MUTED})
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont={"color": MUTED})
    return fig
