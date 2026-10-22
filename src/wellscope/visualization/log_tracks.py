"""
Interactive Multi-Track Wireline Log Visualization Module using Plotly.
Generates petroleum engineering log tracks with inverted depth, log-scaled resistivity,
neutron-density crossover shading, QC flags, and candidate pay highlights.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_multi_track_log_plot(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    top_depth: Optional[float] = None,
    base_depth: Optional[float] = None,
    show_qc_shading: bool = True,
    show_pay_shading: bool = True,
    bit_size: float = 8.5,
    gr_clean: Optional[float] = None,
    gr_shale: Optional[float] = None,
    sw_cutoff: Optional[float] = None,
    phie_cutoff: Optional[float] = None,
    height: int = 900,
) -> go.Figure:
    """
    Builds a professional 5 to 7 track wireline petrophysical display:
    Track 1: Caliper & Gamma Ray (GR / Vsh)
    Track 2: Resistivity (Deep RT & Shallow RXO on log scale)
    Track 3: Porosity Logs (RHOB & NPHI overlay with gas/lithology crossover)
    Track 4: Sonic Slowness (DT, if present)
    Track 5: Computed Porosity (PHID, PHIS, PHIE, if present)
    Track 6: Water Saturation (Sw, if present)
    Track 7: Pay Evaluation Flag / Classification
    """
    # Filter to depth range
    plot_df = df.copy()
    if depth_col in plot_df.columns:
        if top_depth is not None:
            plot_df = plot_df[plot_df[depth_col] >= top_depth]
        if base_depth is not None:
            plot_df = plot_df[plot_df[depth_col] <= base_depth]
    else:
        raise ValueError(f"Depth column '{depth_col}' not found in dataframe.")

    if plot_df.empty:
        # Return empty figure with warning
        fig = go.Figure()
        fig.add_annotation(text="No data in selected depth range.", showarrow=False)
        return fig

    depth = plot_df[depth_col]

    # Determine which tracks to show
    has_dt = "DT" in plot_df.columns and plot_df["DT"].notna().any()
    has_poro_calc = any(c in plot_df.columns and plot_df[c].notna().any() for c in ["PHID", "PHIE", "PHIS"])
    has_sw = "SW" in plot_df.columns and plot_df["SW"].notna().any()
    has_pay = "CANDIDATE_PAY" in plot_df.columns and plot_df["CANDIDATE_PAY"].notna().any()

    # Track list
    track_titles = ["Track 1: GR / CALI", "Track 2: Resistivity", "Track 3: RHOB / NPHI"]
    if has_dt:
        track_titles.append("Track 4: Sonic DT")
    if has_poro_calc:
        track_titles.append("Track 5: Porosity (PHID/PHIE)")
    if has_sw:
        track_titles.append("Track 6: Archie Sw")
    if has_pay:
        track_titles.append("Track 7: Pay Evaluation")

    num_tracks = len(track_titles)
    fig = make_subplots(
        rows=1,
        cols=num_tracks,
        shared_yaxes=True,
        subplot_titles=track_titles,
        horizontal_spacing=0.02,
    )

    current_col = 1

    # ----------------------------------------------------
    # TRACK 1: Gamma Ray & Caliper
    # ----------------------------------------------------
    # GR (0 to 150 API)
    if "GR" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["GR"],
                y=depth,
                mode="lines",
                name="GR (API)",
                line=dict(color="#2ca02c", width=1.5, dash="solid"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )
    # Caliper (dashed dark grey)
    if "CALI" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["CALI"],
                y=depth,
                mode="lines",
                name="CALI (in)",
                line=dict(color="#7f7f7f", width=1.2, dash="dash"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )
        # Bit size reference line
        fig.add_shape(
            type="line",
            x0=bit_size,
            x1=bit_size,
            y0=depth.min(),
            y1=depth.max(),
            line=dict(color="grey", width=1, dash="dot"),
            row=1,
            col=current_col,
        )

    # GR Clean and Shale baselines if supplied
    if gr_clean is not None:
        fig.add_shape(
            type="line", x0=gr_clean, x1=gr_clean, y0=depth.min(), y1=depth.max(),
            line=dict(color="green", width=1, dash="dot"), row=1, col=current_col,
        )
    if gr_shale is not None:
        fig.add_shape(
            type="line", x0=gr_shale, x1=gr_shale, y0=depth.min(), y1=depth.max(),
            line=dict(color="brown", width=1, dash="dot"), row=1, col=current_col,
        )

    fig.update_xaxes(
        title_text="GR (API) / CALI (in)",
        range=[0, 150],
        row=1,
        col=current_col,
        gridcolor="#eaeaea",
    )
    current_col += 1

    # ----------------------------------------------------
    # TRACK 2: Resistivity (Logarithmic)
    # ----------------------------------------------------
    if "RT" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["RT"],
                y=depth,
                mode="lines",
                name="RT (Deep)",
                line=dict(color="#d62728", width=1.8, dash="solid"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )
    if "RXO" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["RXO"],
                y=depth,
                mode="lines",
                name="RXO (Shallow)",
                line=dict(color="#ff7f0e", width=1.2, dash="dashdot"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )

    fig.update_xaxes(
        title_text="Resistivity (ohm.m)",
        type="log",
        range=[np.log10(0.2), np.log10(2000)],
        row=1,
        col=current_col,
        gridcolor="#eaeaea",
    )
    current_col += 1

    # ----------------------------------------------------
    # TRACK 3: Porosity Logs (RHOB & NPHI)
    # RHOB: 1.95 to 2.95 g/cm3 (left to right)
    # NPHI: 0.45 to -0.15 (left to right, so high porosity on left)
    # Standard scale: 1.95 - 2.95 g/cm3 corresponds to 0.45 to -0.15 limestone porosity
    # ----------------------------------------------------
    if "RHOB" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["RHOB"],
                y=depth,
                mode="lines",
                name="RHOB (g/cm3)",
                line=dict(color="#1f77b4", width=1.8, dash="solid"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )

    if "NPHI" in plot_df.columns:
        # Scale NPHI to RHOB visual scale for direct crossover display
        # NPHI scale: 0.45 -> 1.95 g/cm3, -0.15 -> 2.95 g/cm3
        # rhob_equiv = 1.95 + (0.45 - nphi) * (2.95 - 1.95) / (0.45 - (-0.15))
        # = 1.95 + (0.45 - nphi) * (1.0 / 0.60)
        nphi_scaled = 1.95 + (0.45 - plot_df["NPHI"]) * (1.0 / 0.60)
        fig.add_trace(
            go.Scatter(
                x=nphi_scaled,
                y=depth,
                mode="lines",
                name="NPHI (scaled to RHOB)",
                line=dict(color="#9467bd", width=1.4, dash="dot"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )

        # Highlight gas/lithology crossover (where NPHI plotted on RHOB scale falls to the left of RHOB)
        # Note: In gas zones, RHOB drops (moves left) and NPHI drops (moves right in porosity, but on inverted scale moves right)
        # So rhob < nphi_scaled indicates gas crossover
        crossover = (plot_df["RHOB"] < nphi_scaled) & plot_df["RHOB"].notna() & plot_df["NPHI"].notna()
        if crossover.any():
            # Add crossover shade
            fig.add_trace(
                go.Scatter(
                    x=np.where(crossover, plot_df["RHOB"], np.nan),
                    y=depth,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    connectgaps=False,
                ),
                row=1,
                col=current_col,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.where(crossover, nphi_scaled, np.nan),
                    y=depth,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 215, 0, 0.35)",  # Gold gas crossover
                    line=dict(width=0),
                    name="Gas / Crossover",
                    showlegend=True,
                    connectgaps=False,
                ),
                row=1,
                col=current_col,
            )

    fig.update_xaxes(
        title_text="RHOB (1.95-2.95) / NPHI (0.45--0.15)",
        range=[1.95, 2.95],
        row=1,
        col=current_col,
        gridcolor="#eaeaea",
    )
    current_col += 1

    # ----------------------------------------------------
    # TRACK 4: Sonic Slowness DT (if available)
    # ----------------------------------------------------
    if has_dt:
        fig.add_trace(
            go.Scatter(
                x=plot_df["DT"],
                y=depth,
                mode="lines",
                name="DT (µs/ft)",
                line=dict(color="#17becf", width=1.5, dash="solid"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(
            title_text="DT (140-40 µs/ft)",
            range=[140, 40],  # Inverted for porosity increase to left
            row=1,
            col=current_col,
            gridcolor="#eaeaea",
        )
        current_col += 1

    # ----------------------------------------------------
    # TRACK 5: Computed Porosities (PHID, PHIE, PHIS)
    # ----------------------------------------------------
    if has_poro_calc:
        if "PHID" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["PHID"],
                    y=depth,
                    mode="lines",
                    name="PHID (Total Density Por)",
                    line=dict(color="#1f77b4", width=1.5, dash="dash"),
                    connectgaps=False,
                ),
                row=1,
                col=current_col,
            )
        if "PHIE" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["PHIE"],
                    y=depth,
                    mode="lines",
                    name="PHIE (Effective Por)",
                    line=dict(color="#2ca02c", width=2.0, dash="solid"),
                    connectgaps=False,
                ),
                row=1,
                col=current_col,
            )
        if "PHIS" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["PHIS"],
                    y=depth,
                    mode="lines",
                    name="PHIS (Sonic Por)",
                    line=dict(color="#17becf", width=1.2, dash="dot"),
                    connectgaps=False,
                ),
                row=1,
                col=current_col,
            )

        if phie_cutoff is not None:
            fig.add_shape(
                type="line", x0=phie_cutoff, x1=phie_cutoff, y0=depth.min(), y1=depth.max(),
                line=dict(color="red", width=1.2, dash="dash"), row=1, col=current_col,
            )

        fig.update_xaxes(
            title_text="Porosity (0.3 - 0.0 v/v)",
            range=[0.35, 0.0],
            row=1,
            col=current_col,
            gridcolor="#eaeaea",
        )
        current_col += 1

    # ----------------------------------------------------
    # TRACK 6: Archie Water Saturation (Sw)
    # ----------------------------------------------------
    if has_sw:
        fig.add_trace(
            go.Scatter(
                x=plot_df["SW"],
                y=depth,
                mode="lines",
                name="Archie Sw (v/v)",
                line=dict(color="#1f77b4", width=2.0, dash="solid"),
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )
        # Hydrocarbon shading (1 - Sw)
        fig.add_trace(
            go.Scatter(
                x=np.ones(len(depth)),
                y=depth,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=current_col,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["SW"],
                y=depth,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(44, 160, 44, 0.25)",  # Green hydrocarbon fill
                line=dict(width=0),
                name="Hydrocarbon (1-Sw)",
                showlegend=True,
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )

        if sw_cutoff is not None:
            fig.add_shape(
                type="line", x0=sw_cutoff, x1=sw_cutoff, y0=depth.min(), y1=depth.max(),
                line=dict(color="red", width=1.2, dash="dash"), row=1, col=current_col,
            )

        fig.update_xaxes(
            title_text="Sw (1.0 - 0.0 v/v)",
            range=[1.0, 0.0],  # Inverted so hydrocarbon is to the right
            row=1,
            col=current_col,
            gridcolor="#eaeaea",
        )
        current_col += 1

    # ----------------------------------------------------
    # TRACK 7: Pay Evaluation Flag
    # ----------------------------------------------------
    if has_pay:
        pay_binary = plot_df["CANDIDATE_PAY"].astype(float)
        fig.add_trace(
            go.Scatter(
                x=pay_binary,
                y=depth,
                mode="lines",
                name="Candidate Pay",
                line=dict(color="#2ca02c", width=2.0),
                fill="tozerox",
                fillcolor="rgba(40, 167, 69, 0.6)",
                connectgaps=False,
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(
            title_text="Pay Flag (0/1)",
            range=[0, 1.1],
            tickvals=[0, 1],
            row=1,
            col=current_col,
            gridcolor="#eaeaea",
        )
        current_col += 1

    # Invert shared Y axis (depth increasing downward)
    fig.update_yaxes(
        autorange="reversed",
        title_text=f"Depth ({depth_col})",
        gridcolor="#eaeaea",
    )

    # General layout styling
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        hovermode="y unified",
    )

    return fig
