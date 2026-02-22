"""
Shared utilities for the CECL dashboard.
No Streamlit widgets or state here (avoids circular import issues).

Author: Saurabh Chavan
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Plotly chart styling helper
# ---------------------------------------------------------------------------
# Instead of unpacking a template dict (which conflicts with per-chart
# keyword arguments), we apply styling AFTER the chart is configured.
# ---------------------------------------------------------------------------

def style_chart(fig, height=440):
    """Apply consistent professional styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fb",
        font=dict(color="#1e293b", family="Inter, -apple-system, sans-serif", size=13),
        margin=dict(l=60, r=30, t=50, b=50),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(color="#334155")),
    )
    fig.update_xaxes(
        gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#cbd5e1",
        title_font=dict(color="#475569"), tickfont=dict(color="#64748b"),
    )
    fig.update_yaxes(
        gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#cbd5e1",
        title_font=dict(color="#475569"), tickfont=dict(color="#64748b"),
    )
    return fig


# Chart color palette
COLORS = {
    "primary": "#2563eb",
    "danger": "#dc2626",
    "success": "#16a34a",
    "warning": "#d97706",
    "purple": "#7c3aed",
    "cyan": "#0891b2",
    "gray": "#64748b",
}


def info_box(text):
    """Render a styled information box with a blue left border."""
    st.markdown(
        f'<div style="background-color:#eff6ff; border-left:4px solid #2563eb; '
        f'border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0; '
        f'font-size:0.9rem; color:#1e3a5f; line-height:1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


def warning_box(text):
    """Render a styled warning box with an amber left border."""
    st.markdown(
        f'<div style="background-color:#fffbeb; border-left:4px solid #d97706; '
        f'border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0; '
        f'font-size:0.9rem; color:#78350f; line-height:1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


def section_header(text):
    """Render a styled section header."""
    st.markdown(
        f'<div style="color:#1e40af; font-size:1.15rem; font-weight:600; '
        f'margin-top:24px; margin-bottom:8px; padding-bottom:6px; '
        f'border-bottom:2px solid #dbeafe;">{text}</div>',
        unsafe_allow_html=True,
    )