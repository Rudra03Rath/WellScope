"""
Unit tests for data validation, depth integrity checks, and QC flags.
"""

import pytest
import numpy as np
import pandas as pd
from wellscope.core.validation import validate_depth_series, validate_curve_ranges, find_data_gaps
from wellscope.core.qc_flags import compute_qc_flags
from wellscope.core.preprocessing import preprocess_well_data

def test_depth_validation_clean():
    clean_depth = pd.Series(np.arange(1000.0, 1010.0, 0.5))
    res = validate_depth_series(clean_depth)
    assert res["is_valid"]
    assert res["duplicate_count"] == 0
    assert res["inversion_count"] == 0
    assert len(res["errors"]) == 0

def test_depth_validation_duplicate_and_inversion():
    bad_depth = pd.Series([1000.0, 1000.5, 1000.5, 999.0, 1001.0])
    res = validate_depth_series(bad_depth)
    assert not res["is_valid"]
    assert res["duplicate_count"] == 1
    assert res["inversion_count"] == 2  # 1000.5 -> 1000.5 (diff=0), 1000.5 -> 999.0 (diff=-1.5)
    assert any("not monotonically increasing" in e for e in res["errors"])

def test_curve_ranges_outliers():
    df = pd.DataFrame({
        "GR": [50.0, 350.0, -10.0],  # 350 and -10 are outliers
        "RHOB": [2.25, 4.5, 1.2],    # 4.5 and 1.2 are outliers
    })
    res = validate_curve_ranges(df)
    assert res["GR"]["total_outliers"] == 2
    assert res["RHOB"]["total_outliers"] == 2
    assert res["GR"]["warning"] is not None

def test_find_data_gaps():
    depths = pd.Series(np.arange(1000.0, 1010.0, 1.0))
    # introduce a 4-sample gap at indices 3, 4, 5, 6
    gr = pd.Series([50.0, 52.0, 55.0, np.nan, np.nan, np.nan, np.nan, 60.0, 62.0, 65.0])
    gaps = find_data_gaps(gr, depths, min_gap_samples=3)
    assert len(gaps) == 1
    assert gaps[0]["sample_count"] == 4
    assert gaps[0]["start_depth"] == 1003.0
    assert gaps[0]["end_depth"] == 1006.0

def test_compute_qc_flags():
    df = pd.DataFrame({
        "DEPTH": [2000.0, 2001.0, 2002.0],
        "CALI": [8.5, 12.0, 8.5],    # 12.0 is washout (bit_size 8.5 + 1.5 = 10.0)
        "DRHO": [0.01, 0.08, 0.02],  # 0.08 is high DRHO
        "RHOB": [2.35, 1.70, 2.38],
    })
    flags_df, summary = compute_qc_flags(df, bit_size=8.5, washout_diff=1.5, drho_threshold=0.05)
    assert not flags_df["QC_WASHOUT"].iloc[0]
    assert flags_df["QC_WASHOUT"].iloc[1]
    assert flags_df["QC_BAD_HOLE"].iloc[1]
    assert not flags_df["QC_VALID_POROSITY"].iloc[1]  # invalid due to bad hole
    assert flags_df["QC_VALID_POROSITY"].iloc[0]
