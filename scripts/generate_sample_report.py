"""
Sample Report Generator for WellScope.
Runs the complete formation evaluation pipeline on the synthetic well dataset
and saves sample HTML and PDF reports to reports/.
"""

import os
import sys

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from wellscope.io.las_reader import read_las
from wellscope.io.curve_mapping import suggest_curve_mapping
from wellscope.core.preprocessing import preprocess_well_data
from wellscope.petrophysics.gamma_ray import calculate_gamma_ray_index, calculate_shale_volume
from wellscope.petrophysics.porosity import calculate_density_porosity, calculate_sonic_porosity
from wellscope.petrophysics.saturation import calculate_archie_sw
from wellscope.petrophysics.pay_evaluation import calculate_effective_porosity, evaluate_candidate_pay
from wellscope.visualization.log_tracks import create_multi_track_log_plot
from wellscope.reporting.report_builder import generate_html_report, generate_pdf_report
from wellscope.reporting.exports import export_derived_curves_csv, export_intervals_csv, export_parameter_manifest

def generate_sample_outputs():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    las_path = os.path.join(base_dir, "data", "sample_synthetic_well.las")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("1. Ingesting synthetic LAS well...")
    res = read_las(las_path)
    header = res["header"]
    mapping = suggest_curve_mapping(list(res["df"].columns))

    print("2. Preprocessing & Borehole QC...")
    prep = preprocess_well_data(res["df"], mapping, depth_unit=header.get("depth_unit", "M"))
    df = prep["clean_df"]
    qc_summary = prep["qc_summary"]

    print("3. Computing Petrophysics...")
    # Baseline parameters
    gr_clean, gr_shale = 25.0, 115.0
    igr, _ = calculate_gamma_ray_index(df["GR"], gr_clean, gr_shale)
    vsh, vsh_meta = calculate_shale_volume(igr, model="larionov_older")
    df["IGR"] = igr
    df["VSH"] = vsh

    rhoma, rhof = 2.65, 1.00
    phid, _ = calculate_density_porosity(df["RHOB"], rhoma=rhoma, rhof=rhof)
    df["PHID"] = phid

    if "DT" in df.columns:
        phis, _ = calculate_sonic_porosity(df["DT"], dt_matrix=55.5, dt_fluid=189.0)
        df["PHIS"] = phis

    phie, _, _ = calculate_effective_porosity(phid, vsh, model="proportional")
    df["PHIE"] = phie

    rw = 0.05
    sw_raw, sw_clipped, _, _ = calculate_archie_sw(df["RT"], phie, rw=rw, a=1.0, m=2.0, n=2.0, vsh=vsh)
    df["SW_RAW"] = sw_raw
    df["SW"] = sw_clipped

    print("4. Evaluating Candidate Pay...")
    vsh_cut, phie_cut, sw_cut = 0.35, 0.10, 0.50
    pay_res = evaluate_candidate_pay(
        df,
        vsh_cutoff=vsh_cut,
        phie_cutoff=phie_cut,
        sw_cutoff=sw_cut,
        exclude_bad_hole=True,
    )
    df = pay_res["evaluated_df"]

    params_manifest = {
        "gr_clean": gr_clean,
        "gr_shale": gr_shale,
        "vsh_model_name": "Larionov (Older / Mesozoic)",
        "vsh_equation": "Vsh = 0.33 * (2^(2.0 * IGR) - 1)",
        "rhoma": rhoma,
        "rhof": rhof,
        "phie_model_name": "Proportional (PHIE = PHID * (1 - Vsh))",
        "rw": rw,
        "a": 1.0,
        "m": 2.0,
        "n": 2.0,
        "vsh_cutoff": vsh_cut,
        "phie_cutoff": phie_cut,
        "sw_cutoff": sw_cut,
    }

    print("5. Rendering Multi-Track Log Figure...")
    fig = create_multi_track_log_plot(
        df,
        depth_col="DEPTH",
        gr_clean=gr_clean,
        gr_shale=gr_shale,
        sw_cutoff=sw_cut,
        phie_cutoff=phie_cut,
        height=900,
    )

    print("6. Saving Deliverables...")
    # HTML Report
    html_path = os.path.join(reports_dir, "sample_evaluation_report.html")
    html_content = generate_html_report(header, params_manifest, qc_summary, pay_res, fig=fig)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   -> Saved HTML Report: {html_path}")

    # PDF Report
    pdf_path = os.path.join(reports_dir, "sample_evaluation_report.pdf")
    pdf_bytes = generate_pdf_report(header, params_manifest, qc_summary, pay_res)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"   -> Saved PDF Report: {pdf_path}")

    # Curves CSV
    curves_csv_path = os.path.join(reports_dir, "sample_derived_curves.csv")
    with open(curves_csv_path, "w", encoding="utf-8") as f:
        f.write(export_derived_curves_csv(df))
    print(f"   -> Saved Derived Curves CSV: {curves_csv_path}")

    # Intervals CSV (includes both qualified pay and rejected candidate zones)
    intervals_csv_path = os.path.join(reports_dir, "sample_pay_intervals.csv")
    with open(intervals_csv_path, "w", encoding="utf-8") as f:
        f.write(export_intervals_csv(pay_res.get("all_intervals", pay_res.get("intervals", []))))
    print(f"   -> Saved Pay Intervals CSV: {intervals_csv_path}")

    # Parameter Manifest JSON
    manifest_path = os.path.join(reports_dir, "sample_parameter_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(export_parameter_manifest({
            "software": "WellScope",
            "version": "1.0.0",
            "well_header": header,
            "parameters": params_manifest,
            "qc_summary": qc_summary,
            "pay_summary": {k: v for k, v in pay_res.items() if k != "evaluated_df"},
        }))
    print(f"   -> Saved Parameter Manifest: {manifest_path}")

    print("\n[SUCCESS] All sample reports and data exports generated successfully!")

if __name__ == "__main__":
    generate_sample_outputs()
