"""
Unit tests for LAS and CSV I/O, curve mapping, and metadata extraction.
"""

import os
import pytest
import numpy as np
import pandas as pd
from wellscope.io.las_reader import read_las
from wellscope.io.csv_reader import read_csv_log
from wellscope.io.curve_mapping import suggest_curve_mapping, validate_mapping, apply_mapping

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SYNTHETIC_LAS = os.path.join(DATA_DIR, "sample_synthetic_well.las")
SYNTHETIC_CSV = os.path.join(DATA_DIR, "sample_synthetic_well.csv")

def test_read_synthetic_las():
    assert os.path.exists(SYNTHETIC_LAS), f"Missing test file: {SYNTHETIC_LAS}"
    result = read_las(SYNTHETIC_LAS)

    # Check header
    header = result["header"]
    assert header["well_name"] == "WELLSCOPE-SYNTH-01"
    assert header["depth_unit"] == "M"
    assert header["start_depth"] == 2000.0
    assert header["stop_depth"] == 2150.0

    # Check curves info
    curves = [c["mnemonic"] for c in result["curves_info"]]
    for expected in ["DEPTH", "GR", "RHOB", "NPHI", "DT", "RT", "RXO", "CALI", "SP", "DRHO"]:
        assert expected in curves

    # Check DataFrame
    df = result["df"]
    assert len(df) == 1501
    assert "DEPTH" in df.columns
    assert "GR" in df.columns

    # Verify null gap in RHOB was converted to NaN
    # synthetic data introduced nulls between 2040.5 and 2041.0
    null_rows = df[(df["DEPTH"] >= 2040.5) & (df["DEPTH"] <= 2041.0)]
    assert null_rows["RHOB"].isna().all()

def test_read_synthetic_csv():
    assert os.path.exists(SYNTHETIC_CSV), f"Missing test file: {SYNTHETIC_CSV}"
    result = read_csv_log(SYNTHETIC_CSV)

    assert result["header"]["start_depth"] == 2000.0
    assert result["header"]["stop_depth"] == 2150.0
    df = result["df"]
    assert len(df) == 1501
    assert "DEPTH" in df.columns
    assert "GR" in df.columns

def test_suggest_curve_mapping():
    available = ["DEPT", "GAM", "RHOZ", "TNPH", "AC", "RLLD", "MSFL", "CAL", "SP"]
    mapping = suggest_curve_mapping(available)

    assert mapping["DEPTH"] == "DEPT"
    assert mapping["GR"] == "GAM"
    assert mapping["RHOB"] == "RHOZ"
    assert mapping["NPHI"] == "TNPH"
    assert mapping["DT"] == "AC"
    assert mapping["RT"] == "RLLD"
    assert mapping["RXO"] == "MSFL"
    assert mapping["CALI"] == "CAL"
    assert mapping["SP"] == "SP"

def test_validate_mapping_missing_required():
    available = ["GR", "RHOB"]
    mapping = suggest_curve_mapping(available)
    is_valid, warnings = validate_mapping(mapping, available)
    # DEPTH is missing
    assert not is_valid
    assert any("DEPTH" in w for w in warnings)

def test_validate_mapping_missing_rhob():
    # File with missing RHOB should not crash; density porosity is disabled with warning
    available = ["DEPTH", "GR", "RT"]
    mapping = suggest_curve_mapping(available)
    is_valid, warnings = validate_mapping(mapping, available)
    assert is_valid  # Still valid to load and calculate Vsh, etc.
    assert any("Bulk Density (RHOB) is unmapped" in w for w in warnings)

def test_apply_mapping():
    df = pd.DataFrame({"DEPT": [2000.0, 2001.0], "GAM": [45.0, 50.0]})
    mapping = {"DEPTH": "DEPT", "GR": "GAM", "RHOB": None}
    mapped_df = apply_mapping(df, mapping)

    # Canonical columns present
    assert "DEPTH" in mapped_df.columns
    assert "GR" in mapped_df.columns
    assert "RHOB" in mapped_df.columns
    assert mapped_df["DEPTH"].iloc[0] == 2000.0
    assert mapped_df["GR"].iloc[0] == 45.0
    assert mapped_df["RHOB"].isna().all()
    # Original columns preserved
    assert "DEPT" in mapped_df.columns
    assert "GAM" in mapped_df.columns
