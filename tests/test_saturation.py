"""
Unit tests for Archie water saturation and Arp's temperature correction.
"""

import pytest
import numpy as np
import pandas as pd
from wellscope.petrophysics.saturation import (
    calculate_archie_sw,
    arps_temperature_correction,
)

def test_arps_temperature_correction_celsius():
    # Rw=0.10 at 25 C -> at 75 C: 0.10 * (25 + 21.5)/(75 + 21.5) = 0.10 * 46.5 / 96.5 = 0.0481865
    rw_75 = arps_temperature_correction(0.10, t1=25.0, t2=75.0, unit="DEGC")
    assert np.isclose(rw_75, 0.0481865, atol=1e-5)

def test_arps_temperature_correction_fahrenheit():
    # Rw=0.10 at 77 F -> at 167 F: 0.10 * (77 + 6.77)/(167 + 6.77) = 0.048207
    rw_167 = arps_temperature_correction(0.10, t1=77.0, t2=167.0, unit="DEGF")
    assert np.isclose(rw_167, 0.048207, atol=1e-5)

def test_archie_sw_hand_calculated_hydrocarbon_and_wet():
    # Hydrocarbon sand: a=1, m=2, n=2, Rw=0.05, phie=0.25, Rt=20.0 -> Sw = sqrt(0.05 / (0.0625 * 20)) = sqrt(0.04) = 0.20
    # Clean 100% water sand: Rt=0.80 -> Sw = sqrt(0.05 / (0.0625 * 0.8)) = sqrt(1.0) = 1.00
    phie = pd.Series([0.25, 0.25])
    rt = pd.Series([20.0, 0.80])

    sw_raw, sw_clipped, warnings, diag = calculate_archie_sw(
        rt=rt, phie=phie, rw=0.05, a=1.0, m=2.0, n=2.0
    )

    assert np.isclose(sw_raw.iloc[0], 0.20, atol=1e-5)
    assert np.isclose(sw_raw.iloc[1], 1.00, atol=1e-5)
    assert np.isclose(sw_clipped.iloc[0], 0.20, atol=1e-5)
    assert np.isclose(sw_clipped.iloc[1], 1.00, atol=1e-5)
    assert diag["samples_sw_gt_1"] == 0

def test_archie_sw_unclipped_diagnostics():
    # Low Rt = 0.20 -> Sw_raw = sqrt(0.05 / (0.0625 * 0.2)) = sqrt(4.0) = 2.0 (supersaturated)
    phie = pd.Series([0.25])
    rt = pd.Series([0.20])

    sw_raw, sw_clipped, warnings, diag = calculate_archie_sw(
        rt=rt, phie=phie, rw=0.05, a=1.0, m=2.0, n=2.0
    )

    assert np.isclose(sw_raw.iloc[0], 2.0, atol=1e-5)
    assert np.isclose(sw_clipped.iloc[0], 1.0, atol=1e-5)
    assert diag["samples_sw_gt_1"] == 1
    assert any("unclipped Archie Sw > 1.0" in w for w in warnings)

def test_archie_sw_shaly_warning():
    phie = pd.Series([0.20])
    rt = pd.Series([10.0])
    vsh = pd.Series([0.35])  # Exceeds 0.20 threshold

    _, _, warnings, _ = calculate_archie_sw(
        rt=rt, phie=phie, rw=0.05, a=1.0, m=2.0, n=2.0, vsh=vsh, vsh_shaly_threshold=0.20
    )
    assert any("Shaly formation warning" in w for w in warnings)

def test_archie_sw_invalid_rw():
    phie = pd.Series([0.20])
    rt = pd.Series([10.0])
    with pytest.raises(ValueError):
        calculate_archie_sw(rt, phie, rw=-0.05)
