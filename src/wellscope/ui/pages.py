"""
Streamlit UI layout, page tabs, state management, and interaction handlers for WellScope.
"""

import os
import io
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from wellscope.io.las_reader import read_las
from wellscope.io.csv_reader import read_csv_log
from wellscope.io.curve_mapping import (
    CANONICAL_CURVES,
    suggest_curve_mapping,
    validate_mapping,
    apply_mapping,
)
from wellscope.core.preprocessing import preprocess_well_data
from wellscope.core.units import convert_temperature
from wellscope.petrophysics.gamma_ray import (
    VSH_MODELS,
    calculate_gamma_ray_index,
    calculate_shale_volume,
)
from wellscope.petrophysics.porosity import (
    STANDARD_MATRICES,
    STANDARD_FLUIDS,
    calculate_density_porosity,
    calculate_sonic_porosity,
    calculate_neutron_density_porosity,
)
from wellscope.petrophysics.saturation import (
    calculate_archie_sw,
    arps_temperature_correction,
)
from wellscope.petrophysics.pay_evaluation import (
    PHIE_MODELS,
    calculate_effective_porosity,
    evaluate_candidate_pay,
    compute_cutoff_sensitivity,
)
from wellscope.visualization.log_tracks import create_multi_track_log_plot
from wellscope.visualization.sensitivity import create_sensitivity_plot
from wellscope.reporting.exports import (
    export_derived_curves_csv,
    export_intervals_csv,
    export_parameter_manifest,
)
from wellscope.reporting.report_builder import (
    generate_html_report,
    generate_pdf_report,
)

NAVIGATION_TABS = [
    "🏠 Home / Well Summary",
    "📥 Import & Curve Mapping",
    "🔍 QC & Preprocessing",
    "📊 Log Viewer",
    "⚙️ Petrophysics",
    "🎯 Pay Evaluation",
    "📄 Report & Export",
]

SYNTHETIC_LAS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sample_synthetic_well.las")
)

def init_session_state():
    """Initializes persistent application session state variables."""
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = NAVIGATION_TABS[0]
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "header" not in st.session_state:
        st.session_state.header = {}
    if "curve_mapping" not in st.session_state:
        st.session_state.curve_mapping = {}
    if "preprocessed_data" not in st.session_state:
        st.session_state.preprocessed_data = None
    if "petro_results" not in st.session_state:
        st.session_state.petro_results = {}
    if "pay_results" not in st.session_state:
        st.session_state.pay_results = {}
    if "eval_df" not in st.session_state:
        st.session_state.eval_df = None
    if "params_manifest" not in st.session_state:
        st.session_state.params_manifest = {}


def render_sidebar():
    """Renders the sidebar navigation, well status badge, and project disclaimer."""
    with st.sidebar:
        st.title("🛢️ WellScope")
        st.caption("Wireline Petrophysics & Formation Evaluation")
        st.markdown("---")

        # Well status indicator
        if st.session_state.raw_data is not None:
            well_name = st.session_state.header.get("well_name", "Loaded Well")
            st.success(f"**Active Well:** {well_name}")
            start_d = st.session_state.header.get("start_depth")
            stop_d = st.session_state.header.get("stop_depth")
            unit = st.session_state.header.get("depth_unit", "M")
            if start_d is not None and stop_d is not None:
                st.caption(f"Depth: {start_d:.1f} - {stop_d:.1f} {unit}")
        else:
            st.warning("No well loaded. Select sample or upload data.")

        st.markdown("---")
        st.subheader("Navigation")
        selected_tab = st.radio(
            "Go to tab:",
            NAVIGATION_TABS,
            index=NAVIGATION_TABS.index(st.session_state.current_tab)
            if st.session_state.current_tab in NAVIGATION_TABS
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.current_tab = selected_tab

        st.markdown("---")
        st.caption(
            "**Notice:** Educational and portfolio tool for petrophysical interpretation. "
            "Not certified for commercial or operational decision making."
        )


def render_main_content():
    """Dispatches the view based on current_tab."""
    tab = st.session_state.current_tab

    if "Home" in tab:
        render_home_tab()
    elif "Import" in tab:
        render_import_tab()
    elif "QC" in tab:
        render_qc_tab()
    elif "Log Viewer" in tab:
        render_log_viewer_tab()
    elif "Petrophysics" in tab:
        render_petrophysics_tab()
    elif "Pay Evaluation" in tab:
        render_pay_evaluation_tab()
    elif "Report" in tab:
        render_report_tab()


# -----------------------------------------------------------------------------
# TAB 1: Home / Well Summary
# -----------------------------------------------------------------------------
def render_home_tab():
    st.header("🏠 WellScope Formation Evaluation")
    st.markdown(
        r"""
        **WellScope** is a local-first engineering application designed for wireline log interpretation,
        quality control, petrophysical calculation, and pay-zone assessment.
        
        ### Key Petrophysical Capabilities:
        - **Log Data Quality Control (QC):** Non-monotonic depth checks, null gaps, caliper washout detection, and outlier identification.
        - **Shale Volume ($V_{sh}$):** Linear Gamma Ray index, Larionov (Tertiary & Older), Clavier, and Stieber empirical models.
        - **Porosity Interpretation:** Density porosity ($\phi_D$), Wyllie time-average sonic porosity ($\phi_S$), and neutron-density crossover diagnostics.
        - **Fluid Saturation:** Archie's water saturation ($S_w$) with optional Arp's temperature-corrected water resistivity ($R_w$).
        - **Candidate Pay Evaluation:** User-controlled cutoff thresholds ($V_{sh}$, $\phi_e$, $S_w$), gross & net thickness, net-to-gross (NTG) calculations, and sensitivity analysis.
        - **Transparent Reporting:** Audit-ready parameters, mathematical equations, caveats, and exports in HTML, PDF, and CSV formats.
        """
    )

    if st.session_state.raw_data is None:
        st.info("👈 Navigate to **Import & Curve Mapping** in the sidebar to load a LAS or CSV file or explore sample data.")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Load Bundled Synthetic Well Data", type="primary"):
                load_sample_synthetic_data()
                st.rerun()
    else:
        st.success("✅ Well data is loaded and ready for interpretation.")
        render_well_summary_card()


def load_sample_synthetic_data():
    """Helper to load the bundled deterministic synthetic dataset."""
    if os.path.exists(SYNTHETIC_LAS_PATH):
        res = read_las(SYNTHETIC_LAS_PATH)
        st.session_state.raw_data = res
        st.session_state.header = res["header"]
        available_mnemonics = list(res["df"].columns)
        st.session_state.curve_mapping = suggest_curve_mapping(available_mnemonics)
        # Run default preprocessing
        prep = preprocess_well_data(
            res["df"],
            st.session_state.curve_mapping,
            depth_unit=res["header"].get("depth_unit", "M"),
        )
        st.session_state.preprocessed_data = prep
        st.session_state.eval_df = prep["clean_df"]


def render_well_summary_card():
    """Displays a well summary card on the Home tab."""
    header = st.session_state.header
    raw_res = st.session_state.raw_data

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Well Name", header.get("well_name", "N/A"))
    with col2:
        st.metric("Depth Interval", f"{header.get('start_depth', 0):.1f} - {header.get('stop_depth', 0):.1f} {header.get('depth_unit', 'M')}")
    with col3:
        st.metric("Total Samples", len(raw_res["df"]))
    with col4:
        st.metric("Curves In Log", len(raw_res["curves_info"]))

    st.subheader("Available Log Curves Inventory")
    inv_df = pd.DataFrame(raw_res["curves_info"])
    if not inv_df.empty:
        st.dataframe(
            inv_df[["mnemonic", "unit", "descr", "valid_count", "null_count", "min", "max"]],
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# TAB 2: Import & Curve Mapping
# -----------------------------------------------------------------------------
def render_import_tab():
    st.header("📥 Import & Curve Mapping")

    st.markdown(
        "Upload wireline log data from **LAS 2.0** or **CSV** formats, or merge core measurements/supplementary curves "
        "onto an existing well log. You can also load the bundled 5-facies synthetic benchmark well."
    )

    import_tabs = st.tabs([
        "📁 Upload LAS File (.las)",
        "📊 Upload CSV Log File (.csv)",
        "🔗 Merge Secondary CSV (Core / Supplementary)",
        "🧪 Bundled Synthetic Well",
    ])

    # 1. LAS Upload Tab
    with import_tabs[0]:
        st.subheader("Upload Wireline Log (LAS 2.0)")
        las_file = st.file_uploader("Select LAS file (.las):", type=["las"], key="las_uploader")
        if las_file is not None:
            try:
                res = read_las(las_file)
                st.session_state.raw_data = res
                st.session_state.header = res["header"]
                st.session_state.curve_mapping = suggest_curve_mapping(list(res["df"].columns))
                st.success(f"Successfully imported LAS file: **{las_file.name}** ({len(res['df'])} samples)")
            except Exception as e:
                st.error(f"Error parsing LAS file: {str(e)}")

    # 2. CSV Upload Tab
    with import_tabs[1]:
        st.subheader("Upload Wireline Log (CSV)")
        c1, c2 = st.columns([2, 1])
        with c1:
            csv_file = st.file_uploader("Select CSV file (.csv):", type=["csv"], key="csv_uploader")
        with c2:
            custom_depth_col = st.text_input("Depth column name (optional):", value="", placeholder="e.g. DEPTH, MD, DEPT")
            depth_unit_select = st.selectbox("Depth Unit:", ["M", "FT"], index=0, key="csv_depth_unit")

        if csv_file is not None:
            try:
                override = custom_depth_col.strip() if custom_depth_col.strip() else None
                res = read_csv_log(csv_file, depth_col_override=override)
                res["header"]["depth_unit"] = depth_unit_select
                st.session_state.raw_data = res
                st.session_state.header = res["header"]
                st.session_state.curve_mapping = suggest_curve_mapping(list(res["df"].columns))
                st.success(f"Successfully imported CSV log: **{csv_file.name}** ({len(res['df'])} samples)")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # 3. Merge CSV into Active Well Tab
    with import_tabs[2]:
        st.subheader("Merge Supplementary CSV Data into Active Well")
        st.caption(
            "Use this feature to merge core plug analyses, lab test results, formation tops, or additional tool runs "
            "into the currently loaded well based on depth alignment."
        )
        if st.session_state.raw_data is None:
            st.warning("Please load a base LAS or CSV well first before merging supplementary data.")
        else:
            base_df = st.session_state.raw_data["df"]
            base_depth_col = st.session_state.curve_mapping.get("DEPTH") or "DEPTH"
            st.info(f"Active base well: **{st.session_state.header.get('well_name', 'Loaded Well')}** ({len(base_df)} samples, depth column: `{base_depth_col}`)")

            sec_csv = st.file_uploader("Upload secondary CSV file to merge:", type=["csv"], key="sec_csv_uploader")
            if sec_csv is not None:
                try:
                    sec_res = read_csv_log(sec_csv)
                    sec_df = sec_res["df"]
                    sec_cols = list(sec_df.columns)
                    c_d1, c_d2 = st.columns(2)
                    with c_d1:
                        sec_depth_col = st.selectbox("Secondary file depth column:", sec_cols, index=0)
                    with c_d2:
                        tolerance_val = st.number_input("Depth match tolerance (m or ft):", value=0.15, step=0.05)

                    new_curves = [c for c in sec_cols if c != sec_depth_col and c not in base_df.columns]
                    st.write(f"New curves to be added: **{', '.join(new_curves) if new_curves else 'None (all already present)'}**")

                    if st.button("Merge into Active Well", type="primary"):
                        # Nearest depth alignment merge
                        prim_sorted = base_df.sort_values(base_depth_col).copy()
                        sec_sorted = sec_df.sort_values(sec_depth_col).dropna(subset=[sec_depth_col]).copy()
                        merged = pd.merge_asof(
                            prim_sorted,
                            sec_sorted[[sec_depth_col] + new_curves],
                            left_on=base_depth_col,
                            right_on=sec_depth_col,
                            direction="nearest",
                            tolerance=tolerance_val,
                        )
                        if sec_depth_col != base_depth_col and sec_depth_col in merged.columns:
                            merged = merged.drop(columns=[sec_depth_col])

                        st.session_state.raw_data["df"] = merged
                        # Refresh curve inventory
                        for nc in new_curves:
                            ser = merged[nc].dropna()
                            st.session_state.raw_data["curves_info"].append({
                                "mnemonic": nc,
                                "unit": "",
                                "descr": "Merged CSV curve",
                                "count": len(merged),
                                "valid_count": len(ser),
                                "null_count": int(merged[nc].isna().sum()),
                                "min": float(ser.min()) if len(ser) > 0 else np.nan,
                                "max": float(ser.max()) if len(ser) > 0 else np.nan,
                                "mean": float(ser.mean()) if len(ser) > 0 else np.nan,
                            })
                        st.session_state.curve_mapping = suggest_curve_mapping(list(merged.columns))
                        st.success(f"Successfully merged {len(new_curves)} curves into active dataset!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to merge secondary CSV: {str(e)}")

    # 4. Bundled Synthetic Well Tab
    with import_tabs[3]:
        st.subheader("Bundled Synthetic Benchmark Well")
        st.markdown(
            "Load the deterministic 150-meter synthetic well (`sample_synthetic_well.las`) "
            "containing 5 distinct petrophysical facies (gas sand, water sand, shale, washout, and tight carbonate)."
        )
        if st.button("🚀 Load Bundled Synthetic Well", key="load_synth_btn"):
            load_sample_synthetic_data()
            st.success("Loaded synthetic well!")
            st.rerun()

    if st.session_state.raw_data is None:
        st.info("👈 Please select an upload option above or load the bundled synthetic well to proceed.")
        return

    st.markdown("---")
    st.subheader("Manual Curve Mnemonic Mapping")
    st.markdown(
        "Vendor-specific tool mnemonics must be explicitly confirmed. "
        "Never silently treat an arbitrary resistivity curve as true formation resistivity ($R_t$)."
    )

    df_cols = ["(None)"] + list(st.session_state.raw_data["df"].columns)
    mapping = st.session_state.curve_mapping

    map_cols = st.columns(3)
    updated_mapping = {}

    curve_keys = list(CANONICAL_CURVES.keys())
    for idx, canon in enumerate(curve_keys):
        col_widget = map_cols[idx % 3]
        info = CANONICAL_CURVES[canon]
        current_val = mapping.get(canon)
        default_idx = df_cols.index(current_val) if current_val in df_cols else 0

        req_label = "*(Required)*" if info["required"] else "*(Optional)*"
        label = f"{canon} - {info['name']} {req_label}"
        help_text = f"{info['description']}. Default units: {', '.join(info['default_units'])}"

        selected = col_widget.selectbox(
            label,
            df_cols,
            index=default_idx,
            help=help_text,
            key=f"map_select_{canon}",
        )
        updated_mapping[canon] = selected if selected != "(None)" else None

    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        if st.button("Confirm Mapping & Preprocess", type="primary"):
            st.session_state.curve_mapping = updated_mapping
            prep = preprocess_well_data(
                st.session_state.raw_data["df"],
                updated_mapping,
                depth_unit=st.session_state.header.get("depth_unit", "M"),
            )
            st.session_state.preprocessed_data = prep
            st.session_state.eval_df = prep["clean_df"]
            st.success("Curve mapping applied successfully!")
            st.rerun()

    # Mapping validation feedback
    is_valid_map, map_warnings = validate_mapping(
        st.session_state.curve_mapping,
        list(st.session_state.raw_data["df"].columns),
    )
    if not is_valid_map:
        for w in map_warnings:
            st.error(w)
    elif map_warnings:
        for w in map_warnings:
            st.warning(w)

    st.subheader("Raw Data Preview")
    st.dataframe(st.session_state.raw_data["df"].head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: QC & Preprocessing
# -----------------------------------------------------------------------------
def render_qc_tab():
    st.header("🔍 Data Quality Control & Preprocessing")

    if st.session_state.preprocessed_data is None:
        st.warning("No preprocessed well data available. Complete Import & Curve Mapping first.")
        return

    prep = st.session_state.preprocessed_data
    depth_val = prep["depth_validation"]
    curve_val = prep["curve_validation"]
    qc_summary = prep["qc_summary"]

    # 1. Depth Integrity Card
    st.subheader("1. Depth Indexing Integrity")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Total Samples", depth_val["total_rows"])
    with d_col2:
        st.metric("Duplicate Depths", depth_val["duplicate_count"])
    with d_col3:
        st.metric("Depth Inversions", depth_val["inversion_count"])
    with d_col4:
        st.metric("Irregular Steps", depth_val["irregular_step_count"])

    if depth_val["errors"]:
        for err in depth_val["errors"]:
            st.error(err)
    elif depth_val["warnings"]:
        for w in depth_val["warnings"]:
            st.warning(w)
    else:
        st.success("✅ Depth grid is strictly monotonic and uniform.")

    # 2. Borehole QC & Washout Card
    st.subheader("2. Borehole Environment & Tool Contact QC")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.metric("Bit Size", f"{qc_summary.get('bit_size_in', 8.5):.1f} in")
    with b_col2:
        st.metric("Washout Threshold", f"{qc_summary.get('washout_threshold_in', 10.0):.1f} in")
    with b_col3:
        st.metric("Washout Samples", qc_summary.get("washout_sample_count", 0))
    with b_col4:
        st.metric("Bad Hole Exclusions", qc_summary.get("bad_hole_sample_count", 0))

    if qc_summary.get("washout_intervals"):
        st.markdown("**Flagged Washout Intervals:**")
        st.dataframe(pd.DataFrame(qc_summary["washout_intervals"]), use_container_width=True, hide_index=True)
    else:
        st.info("No severe borehole washout intervals detected.")

    # 3. Physical Ranges Outliers Table
    st.subheader("3. Physical Bounds & Sensor Outlier Check")
    outlier_rows = []
    for curve, stats in curve_val.items():
        outlier_rows.append({
            "Curve": curve,
            "Observed Min": f"{stats['min_observed']:.2f}",
            "Observed Max": f"{stats['max_observed']:.2f}",
            "Valid Bounds": f"[{stats['expected_min']}, {stats['expected_max']}] {stats['unit']}",
            "Outlier Count": stats["total_outliers"],
            "Diagnostic Note": stats["warning"] or "Within normal physical limits",
        })
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)

    # 4. Gaps Card
    st.subheader("4. Missing Data Gaps")
    gaps = prep.get("gaps", {})
    if gaps:
        for c, gap_list in gaps.items():
            st.warning(f"**{c}:** {len(gap_list)} continuous null gap(s) detected. Gaps will not be silently bridged.")
    else:
        st.info("No large continuous null gaps detected in key logging curves.")


# -----------------------------------------------------------------------------
# TAB 4: Log Viewer
# -----------------------------------------------------------------------------
def render_log_viewer_tab():
    st.header("📊 Interactive Multi-Track Log Viewer")

    if st.session_state.eval_df is None:
        st.warning("No well data available. Load a well in the Import tab first.")
        return

    df = st.session_state.eval_df
    depth_col = "DEPTH"
    min_d = float(df[depth_col].min())
    max_d = float(df[depth_col].max())

    # Interactive controls in an expander
    with st.expander("🛠️ Log Display & Depth Zoom Controls", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            z_top = st.number_input("Top Depth", value=min_d, min_value=min_d, max_value=max_d, step=1.0)
        with c2:
            z_base = st.number_input("Base Depth", value=max_d, min_value=min_d, max_value=max_d, step=1.0)
        with c3:
            plot_height = st.slider("Plot Height (px)", min_value=600, max_value=1400, value=900, step=50)
        with c4:
            bit_size = st.number_input("Bit Size (in)", value=8.5, step=0.125)

    # Generate multi-track plot
    params = st.session_state.params_manifest
    fig = create_multi_track_log_plot(
        df,
        depth_col=depth_col,
        top_depth=z_top,
        base_depth=z_base,
        bit_size=bit_size,
        gr_clean=params.get("gr_clean"),
        gr_shale=params.get("gr_shale"),
        sw_cutoff=params.get("sw_cutoff"),
        phie_cutoff=params.get("phie_cutoff"),
        height=plot_height,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Compact Interpretation Panel
    st.subheader("Interval Evidence & Diagnostic Summary")
    pay_res = st.session_state.pay_results
    if pay_res and pay_res.get("intervals"):
        st.dataframe(pd.DataFrame(pay_res["intervals"]), use_container_width=True, hide_index=True)
    else:
        st.info("Run Petrophysics and Pay Evaluation to generate interval evidence.")


# -----------------------------------------------------------------------------
# TAB 5: Petrophysics
# -----------------------------------------------------------------------------
def render_petrophysics_tab():
    st.header("⚙️ Petrophysical Interpretation Engine")

    if st.session_state.eval_df is None:
        st.warning("Please load and preprocess well data first.")
        return

    df = st.session_state.eval_df

    st.markdown(
        "Configure baseline petrophysical parameters and equations. "
        "Calculations are strictly transparent; no assumptions are hidden."
    )

    col1, col2 = st.columns(2)

    # 1. Shale Volume Form
    with col1:
        st.subheader("1. Shale Volume ($V_{sh}$)")
        gr_series = df["GR"].dropna() if "GR" in df.columns else pd.Series()
        suggested_gr_clean = float(np.percentile(gr_series, 5)) if len(gr_series) > 0 else 25.0
        suggested_gr_shale = float(np.percentile(gr_series, 95)) if len(gr_series) > 0 else 115.0

        gr_clean = st.number_input("GR Clean Sand Baseline (API)", value=round(suggested_gr_clean, 1))
        gr_shale = st.number_input("GR Pure Shale Baseline (API)", value=round(suggested_gr_shale, 1))

        vsh_model_key = st.selectbox(
            "Shale Volume Model",
            options=list(VSH_MODELS.keys()),
            format_func=lambda k: VSH_MODELS[k]["name"],
            help="Select linear benchmark or an empirical basin model.",
        )
        st.caption(f"**Equation:** {VSH_MODELS[vsh_model_key]['descr']}")
        st.caption(f"**Applicability:** {VSH_MODELS[vsh_model_key]['applicability']}")

    # 2. Porosity Form
    with col2:
        st.subheader(r"2. Porosity Engine ($\phi$)")
        matrix_key = st.selectbox(
            "Matrix Lithology",
            options=list(STANDARD_MATRICES.keys()),
            format_func=lambda k: f"{STANDARD_MATRICES[k]['name']} (ρma={STANDARD_MATRICES[k]['rhoma']} g/cm³)",
        )
        custom_rhoma = st.number_input(
            "Matrix Density RHOMA (g/cm³)",
            value=STANDARD_MATRICES[matrix_key]["rhoma"],
            step=0.01,
        )
        fluid_key = st.selectbox(
            "Pore Fluid Type",
            options=list(STANDARD_FLUIDS.keys()),
            format_func=lambda k: f"{STANDARD_FLUIDS[k]['name']} (ρf={STANDARD_FLUIDS[k]['rhof']} g/cm³)",
        )
        custom_rhof = st.number_input(
            "Fluid Density RHOF (g/cm³)",
            value=STANDARD_FLUIDS[fluid_key]["rhof"],
            step=0.01,
        )

        nphi_basis = st.selectbox("NPHI Log Calibration Basis", ["Limestone", "Sandstone", "Dolomite"])

        st.markdown("**Effective Porosity Model:**")
        phie_model_key = st.selectbox(
            "Model",
            options=list(PHIE_MODELS.keys()),
            format_func=lambda k: PHIE_MODELS[k]["name"],
        )

    st.markdown("---")
    col3, col4 = st.columns(2)

    # 3. Archie Water Saturation Form
    with col3:
        st.subheader("3. Archie Water Saturation ($S_w$)")
        rw = st.number_input(
            "Formation Water Resistivity Rw at Reservoir Temp (ohm.m)",
            value=0.05,
            min_value=0.001,
            step=0.005,
            format="%.4f",
        )
        c_a, c_m, c_n = st.columns(3)
        with c_a:
            arch_a = st.number_input("Tortuosity factor (a)", value=1.0, step=0.05)
        with c_m:
            arch_m = st.number_input("Cementation exponent (m)", value=2.0, step=0.05)
        with c_n:
            arch_n = st.number_input("Saturation exponent (n)", value=2.0, step=0.05)

        st.caption("Equation: $S_w = \\left(\\frac{a \\times R_w}{\\phi_e^m \\times R_t}\\right)^{1/n}$")

    # 4. Arp's Temperature Conversion Tool
    with col4:
        st.subheader("Optional: Arp's $R_w$ Temperature Adjuster")
        with st.expander("Calculate Rw at Formation Temperature"):
            rw_surface = st.number_input("Surface Measured Rw (ohm.m)", value=0.10, step=0.01, key="rw_surf")
            t_surface = st.number_input("Surface Temperature (°C)", value=20.0, step=1.0, key="t_surf")
            t_res = st.number_input("Formation Temperature (°C)", value=80.0, step=1.0, key="t_form")
            if st.button("Calculate Formation Rw"):
                rw_calc = arps_temperature_correction(rw_surface, t_surface, t_res, unit="DEGC")
                st.success(f"Temperature-corrected Rw = {rw_calc:.4f} ohm.m at {t_res}°C")

    st.markdown("---")
    if st.button("🚀 Calculate Petrophysical Properties", type="primary"):
        try:
            # 1. Shale Volume
            igr, gr_warnings = calculate_gamma_ray_index(df["GR"], gr_clean, gr_shale)
            vsh, vsh_meta = calculate_shale_volume(igr, model=vsh_model_key)
            df["IGR"] = igr
            df["VSH"] = vsh

            # 2. Porosity
            phid, phid_warnings = calculate_density_porosity(df["RHOB"], rhoma=custom_rhoma, rhof=custom_rhof)
            df["PHID"] = phid

            # Sonic Porosity if DT present
            if "DT" in df.columns and df["DT"].notna().any():
                phis, phis_warnings = calculate_sonic_porosity(
                    df["DT"],
                    dt_matrix=STANDARD_MATRICES[matrix_key]["dtma"],
                    dt_fluid=STANDARD_FLUIDS[fluid_key]["dtf"],
                )
                df["PHIS"] = phis

            # 3. Effective Porosity
            phie, phie_meta, phie_warnings = calculate_effective_porosity(
                phid, vsh, model=phie_model_key, phid_shale=0.05
            )
            df["PHIE"] = phie

            # 4. Archie Sw
            sw_raw, sw_clipped, sw_warnings, sw_diag = calculate_archie_sw(
                df["RT"], phie, rw=rw, a=arch_a, m=arch_m, n=arch_n, vsh=vsh
            )
            df["SW_RAW"] = sw_raw
            df["SW"] = sw_clipped

            # Save to session state
            st.session_state.eval_df = df
            st.session_state.params_manifest = {
                "gr_clean": gr_clean,
                "gr_shale": gr_shale,
                "vsh_model_name": VSH_MODELS[vsh_model_key]["name"],
                "vsh_equation": VSH_MODELS[vsh_model_key]["descr"],
                "rhoma": custom_rhoma,
                "rhof": custom_rhof,
                "phie_model_name": PHIE_MODELS[phie_model_key]["name"],
                "rw": rw,
                "a": arch_a,
                "m": arch_m,
                "n": arch_n,
            }

            st.success("Petrophysical calculations completed successfully!")

            # Display Warnings
            all_warnings = gr_warnings + phid_warnings + sw_warnings
            if all_warnings:
                st.subheader("Calculation Diagnostics & Caveats")
                for w in all_warnings:
                    st.warning(w)

            st.rerun()

        except Exception as e:
            st.error(f"Petrophysical calculation error: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 6: Pay Evaluation
# -----------------------------------------------------------------------------
def render_pay_evaluation_tab():
    st.header("🎯 Candidate Pay Evaluation & Sensitivity")

    if st.session_state.eval_df is None or "SW" not in st.session_state.eval_df.columns:
        st.warning("Please complete Petrophysics calculations first.")
        return

    df = st.session_state.eval_df

    st.subheader("Pay Cutoff Thresholds")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vsh_cut = st.slider("Shale Volume Cutoff (Vsh <=)", min_value=0.10, max_value=0.60, value=0.35, step=0.05)
    with c2:
        phie_cut = st.slider("Effective Porosity Cutoff (PHIE >=)", min_value=0.04, max_value=0.30, value=0.10, step=0.01)
    with c3:
        sw_cut = st.slider("Water Saturation Cutoff (Sw <=)", min_value=0.20, max_value=0.80, value=0.50, step=0.05)
    with c4:
        ex_bad = st.checkbox("Exclude Bad Hole / Washout", value=True)

    # Evaluate Candidate Pay
    eval_res = evaluate_candidate_pay(
        df,
        vsh_cutoff=vsh_cut,
        phie_cutoff=phie_cut,
        sw_cutoff=sw_cut,
        exclude_bad_hole=ex_bad,
    )
    st.session_state.pay_results = eval_res
    st.session_state.eval_df = eval_res["evaluated_df"]
    st.session_state.params_manifest["vsh_cutoff"] = vsh_cut
    st.session_state.params_manifest["phie_cutoff"] = phie_cut
    st.session_state.params_manifest["sw_cutoff"] = sw_cut

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Gross Interval Thickness", f"{eval_res['gross_thickness']:.2f} m")
    with m2:
        st.metric("Net Reservoir Thickness", f"{eval_res['reservoir_thickness']:.2f} m")
    with m3:
        st.metric("Net Qualifying Pay", f"{eval_res['net_pay_thickness']:.2f} m")
    with m4:
        st.metric("Net-to-Gross (NTG)", f"{eval_res['net_to_gross']:.1%}")

    # Candidate Pay Intervals Table
    st.subheader("✅ Qualified Candidate Pay Intervals")
    intervals = eval_res.get("intervals", [])
    if intervals:
        st.dataframe(pd.DataFrame(intervals), use_container_width=True, hide_index=True)
    else:
        st.warning("No depth intervals satisfy the selected cutoff criteria.")

    # Disqualified / Rejected Reservoir Intervals
    st.subheader("⚠️ Disqualified Prospective Reservoir Intervals")
    st.caption(
        "Candidate intervals that exhibited reservoir-like characteristics (e.g. good porosity or resistivity shows) "
        "but were rejected from Net Pay based on cutoffs or borehole QC:"
    )
    rejected = eval_res.get("rejected_intervals", [])
    if rejected:
        rej_df = pd.DataFrame(rejected)[["status", "top_depth", "base_depth", "net_pay_thickness", "avg_vsh", "avg_phie", "avg_sw", "rejection_reason"]]
        rej_df = rej_df.rename(columns={"net_pay_thickness": "thickness (m)"})
        st.dataframe(rej_df, use_container_width=True, hide_index=True)
    else:
        st.info("No near-miss candidate intervals were identified.")

    # Sensitivity Analysis
    st.markdown("---")
    st.subheader("Cutoff Sensitivity Analysis")
    st.caption("Inspect how Net Pay thickness responds to cutoff sweeps around current baseline values.")
    sens_data = compute_cutoff_sensitivity(
        df,
        base_vsh_cut=vsh_cut,
        base_phie_cut=phie_cut,
        base_sw_cut=sw_cut,
    )
    sens_fig = create_sensitivity_plot(sens_data)
    st.plotly_chart(sens_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 7: Report & Export
# -----------------------------------------------------------------------------
def render_report_tab():
    st.header("📄 Formation Evaluation Report & Data Export")

    if st.session_state.eval_df is None:
        st.warning("Please load data and complete petrophysical evaluation first.")
        return

    df = st.session_state.eval_df
    header = st.session_state.header
    params = st.session_state.params_manifest
    prep = st.session_state.preprocessed_data
    qc_summary = prep.get("qc_summary", {}) if prep else {}
    pay_results = st.session_state.pay_results

    st.subheader("Download Evaluation Deliverables")

    col1, col2, col3, col4 = st.columns(4)

    # 1. Download HTML Report
    with col1:
        log_fig = create_multi_track_log_plot(df, height=750)
        html_report = generate_html_report(header, params, qc_summary, pay_results, fig=log_fig)
        st.download_button(
            "🌐 Download HTML Report",
            data=html_report,
            file_name=f"WellScope_Report_{header.get('well_name', 'Well')}.html",
            mime="text/html",
            use_container_width=True,
        )

    # 2. Download PDF Report
    with col2:
        try:
            pdf_bytes = generate_pdf_report(header, params, qc_summary, pay_results)
            st.download_button(
                "📑 Download PDF Report",
                data=pdf_bytes,
                file_name=f"WellScope_Report_{header.get('well_name', 'Well')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")

    # 3. Download Curves CSV
    with col3:
        curves_csv = export_derived_curves_csv(df)
        st.download_button(
            "📊 Download Curves CSV",
            data=curves_csv,
            file_name=f"WellScope_Curves_{header.get('well_name', 'Well')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 4. Download Intervals CSV
    with col4:
        intervals_csv = export_intervals_csv(pay_results.get("all_intervals", pay_results.get("intervals", [])))
        st.download_button(
            "📋 Download Intervals CSV",
            data=intervals_csv,
            file_name=f"WellScope_Intervals_{header.get('well_name', 'Well')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Parameter Manifest (JSON)")
    manifest_json = export_parameter_manifest({
        "software": "WellScope",
        "version": "1.0.0",
        "well_header": header,
        "parameters": params,
        "qc_summary": qc_summary,
        "pay_summary": {k: v for k, v in pay_results.items() if k != "evaluated_df"},
    })
    st.download_button(
        "💾 Download Parameter Manifest (JSON)",
        data=manifest_json,
        file_name="parameter_manifest.json",
        mime="application/json",
    )
    st.json(manifest_json)
