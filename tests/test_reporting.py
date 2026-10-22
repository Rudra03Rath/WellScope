"""
Unit tests for report generation (HTML & PDF) and data exports.
"""

import pytest
import pandas as pd
from wellscope.reporting.exports import (
    export_derived_curves_csv,
    export_intervals_csv,
    export_parameter_manifest,
)
from wellscope.reporting.report_builder import generate_html_report, generate_pdf_report

def test_exports_csv_and_manifest():
    df = pd.DataFrame({
        "DEPTH": [2000.0, 2001.0],
        "GR": [30.0, 45.0],
        "VSH": [0.10, 0.20],
        "PHIE": [0.22, 0.18],
        "SW": [0.25, 0.40],
        "CANDIDATE_PAY": [True, True],
    })

    csv_data = export_derived_curves_csv(df)
    assert "DEPTH,GR,VSH" in csv_data
    assert "2000.0" in csv_data

    intervals = [{
        "top_depth": 2000.0,
        "base_depth": 2025.0,
        "net_pay_thickness": 25.0,
        "avg_vsh": 0.15,
        "avg_phie": 0.21,
        "avg_sw": 0.28,
    }]
    int_csv = export_intervals_csv(intervals)
    assert "2000.0,2025.0,25.0" in int_csv

    manifest = {"well_name": "TEST", "rw": 0.05}
    json_str = export_parameter_manifest(manifest)
    assert '"well_name": "TEST"' in json_str

def test_generate_html_report():
    header = {"well_name": "WELLSCOPE-01", "field": "TEST_FIELD", "depth_unit": "M"}
    params = {"vsh_model_name": "Linear", "gr_clean": 25.0, "gr_shale": 115.0, "rw": 0.05}
    qc = {"bit_size_in": 8.5, "washout_threshold_in": 10.0, "washout_sample_count": 0, "bad_hole_sample_count": 0}
    pay = {"gross_thickness": 100.0, "reservoir_thickness": 40.0, "net_pay_thickness": 30.0, "net_to_gross": 0.30, "intervals": []}

    html = generate_html_report(header, params, qc, pay)
    assert "<!DOCTYPE html>" in html
    assert "WELLSCOPE-01" in html
    assert "Net-To-Gross (NTG)" in html
    assert "Disclaimer:" in html

def test_generate_pdf_report():
    header = {"well_name": "WELLSCOPE-01", "field": "TEST_FIELD"}
    params = {"vsh_model_name": "Linear", "gr_clean": 25.0, "gr_shale": 115.0, "rw": 0.05, "rhoma": 2.65, "rhof": 1.00, "a": 1.0, "m": 2.0, "n": 2.0}
    qc = {"bit_size_in": 8.5, "washout_threshold_in": 10.0, "washout_sample_count": 0, "bad_hole_sample_count": 0}
    pay = {
        "gross_thickness": 100.0,
        "reservoir_thickness": 40.0,
        "net_pay_thickness": 30.0,
        "net_to_gross": 0.30,
        "vsh_cutoff": 0.35,
        "phie_cutoff": 0.10,
        "sw_cutoff": 0.50,
        "intervals": [{
            "top_depth": 2025.0,
            "base_depth": 2055.0,
            "net_pay_thickness": 30.0,
            "avg_vsh": 0.15,
            "avg_phie": 0.22,
            "avg_sw": 0.25,
        }],
    }

    pdf_bytes = generate_pdf_report(header, params, qc, pay)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
