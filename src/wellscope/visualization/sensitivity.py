"""
Cutoff Sensitivity Plotting Module using Plotly.
Visualizes how Net Pay and Net-to-Gross respond to changes in Vsh, PHIE, and Sw cutoffs.
"""

from typing import Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sensitivity_plot(sensitivity_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Creates a 3-panel subplot displaying Net Pay thickness vs Vsh, PHIE, and Sw cutoffs."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Net Pay vs Vsh Cutoff",
            "Net Pay vs PHIE Cutoff",
            "Net Pay vs Sw Cutoff",
        ],
        shared_yaxes=True,
    )

    # 1. Vsh Sensitivity
    v_df = sensitivity_data.get("vsh_sensitivity", pd.DataFrame())
    if not v_df.empty:
        fig.add_trace(
            go.Scatter(
                x=v_df["cutoff_value"],
                y=v_df["net_pay"],
                mode="lines+markers",
                name="Net Pay (m)",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

    # 2. PHIE Sensitivity
    p_df = sensitivity_data.get("phie_sensitivity", pd.DataFrame())
    if not p_df.empty:
        fig.add_trace(
            go.Scatter(
                x=p_df["cutoff_value"],
                y=p_df["net_pay"],
                mode="lines+markers",
                name="Net Pay (m)",
                line=dict(color="#2ca02c", width=2.5),
                marker=dict(size=6),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3. Sw Sensitivity
    s_df = sensitivity_data.get("sw_sensitivity", pd.DataFrame())
    if not s_df.empty:
        fig.add_trace(
            go.Scatter(
                x=s_df["cutoff_value"],
                y=s_df["net_pay"],
                mode="lines+markers",
                name="Net Pay (m)",
                line=dict(color="#d62728", width=2.5),
                marker=dict(size=6),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    fig.update_xaxes(title_text="Vsh Cutoff", row=1, col=1)
    fig.update_xaxes(title_text="PHIE Cutoff (v/v)", row=1, col=2)
    fig.update_xaxes(title_text="Sw Cutoff (v/v)", row=1, col=3)
    fig.update_yaxes(title_text="Net Qualifying Pay (m)", row=1, col=1)

    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return fig
