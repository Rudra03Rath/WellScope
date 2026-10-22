"""
Unit tests for Effective Porosity, Pay Evaluation, and Cutoff Sensitivity.
"""

import pytest
import numpy as np
import pandas as pd
from wellscope.petrophysics.pay_evaluation import (
    calculate_effective_porosity,
    evaluate_candidate_pay,
    compute_cutoff_sensitivity,
)

def test_effective_porosity_models():
    phit = pd.Series([0.20, 0.20])
    vsh = pd.Series([0.0, 0.25])

    # 1. Proportional: PHIE = PHIT * (1 - Vsh)
    # At Vsh=0 -> 0.20; at Vsh=0.25 -> 0.20 * 0.75 = 0.15
    phie_prop, meta_prop, _ = calculate_effective_porosity(phit, vsh, model="proportional")
    assert np.isclose(phie_prop.iloc[0], 0.20)
    assert np.isclose(phie_prop.iloc[1], 0.15)

    # 2. Density shale subtraction: PHIE = PHIT - Vsh * PHID_shale
    # At Vsh=0.25, phid_shale=0.08 -> 0.20 - 0.25 * 0.08 = 0.20 - 0.02 = 0.18
    phie_sub, meta_sub, _ = calculate_effective_porosity(phit, vsh, model="density_shale_subtraction", phid_shale=0.08)
    assert np.isclose(phie_sub.iloc[0], 0.20)
    assert np.isclose(phie_sub.iloc[1], 0.18)

def test_evaluate_candidate_pay_thickness_and_ntg():
    # Construct 10 depth samples at 1.0 m spacing (2000 to 2009 m)
    depths = np.arange(2000.0, 2010.0, 1.0)
    df = pd.DataFrame({
        "DEPTH": depths,
        # 5 samples pay (2000-2004), 3 samples wet sand (2005-2007), 2 samples shale (2008-2009)
        "VSH":  [0.15]*5 + [0.15]*3 + [0.70]*2,
        "PHIE": [0.20]*5 + [0.22]*3 + [0.06]*2,
        "SW":   [0.25]*5 + [0.85]*3 + [0.80]*2,
        "GR":   [30.0]*8 + [110.0]*2,
        "QC_BAD_HOLE": [False]*10,
    })

    res = evaluate_candidate_pay(
        df,
        vsh_cutoff=0.35,
        phie_cutoff=0.10,
        sw_cutoff=0.50,
        exclude_bad_hole=True,
    )

    # 10 samples with 1.0 m step = 9 intervals or ~10m delta_z
    assert res["gross_thickness"] > 8.0
    # Exactly 5 samples meet pay criteria
    eval_df = res["evaluated_df"]
    assert eval_df["CANDIDATE_PAY"].sum() == 5
    assert np.isclose(res["avg_sw"], 0.25)
    assert np.isclose(res["avg_phie"], 0.20)
    assert np.isclose(res["avg_vsh"], 0.15)
    assert len(res["intervals"]) == 1
    assert res["intervals"][0]["top_depth"] == 2000.0
    assert res["intervals"][0]["base_depth"] == 2004.0
    assert res["intervals"][0]["status"] == "QUALIFIED_PAY"

    # Verify rejected prospective reservoir intervals (the 3 wet sand samples at 2005-2007)
    assert len(res["rejected_intervals"]) == 1
    assert res["rejected_intervals"][0]["status"] == "REJECTED_WATER_BEARING"
    assert res["rejected_intervals"][0]["top_depth"] == 2005.0
    assert res["rejected_intervals"][0]["base_depth"] == 2007.0
    assert "Water-bearing" in res["rejected_intervals"][0]["rejection_reason"]
    assert len(res["all_intervals"]) == 2

def test_pay_evaluation_bad_hole_exclusion():
    # Sample meets petro criteria but fails QC
    df = pd.DataFrame({
        "DEPTH": [2000.0, 2001.0],
        "VSH": [0.10, 0.10],
        "PHIE": [0.22, 0.22],
        "SW": [0.20, 0.20],
        "QC_BAD_HOLE": [False, True],  # Sample 2 is bad hole / washout
    })

    res = evaluate_candidate_pay(df, exclude_bad_hole=True)
    eval_df = res["evaluated_df"]
    assert eval_df["CANDIDATE_PAY"].iloc[0]
    assert not eval_df["CANDIDATE_PAY"].iloc[1]  # Excluded by QC!

def test_cutoff_sensitivity():
    df = pd.DataFrame({
        "DEPTH": np.arange(2000.0, 2010.0, 1.0),
        "VSH":  [0.15]*5 + [0.40]*5,
        "PHIE": [0.20]*5 + [0.12]*5,
        "SW":   [0.25]*5 + [0.55]*5,
    })

    sens = compute_cutoff_sensitivity(df)
    assert "vsh_sensitivity" in sens
    assert "phie_sensitivity" in sens
    assert "sw_sensitivity" in sens
    assert len(sens["vsh_sensitivity"]) == 9
    assert "net_pay" in sens["vsh_sensitivity"].columns
