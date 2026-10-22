"""
Unit tests for petrophysical unit conversions.
"""

import pytest
import numpy as np
import pandas as pd
from wellscope.core.units import (
    convert_depth,
    convert_density,
    convert_sonic,
    normalize_porosity_fraction,
    convert_temperature,
)

def test_convert_depth():
    meters = np.array([1000.0, 2000.0, 3000.0])
    feet = convert_depth(meters, "M", "FT")
    # 1 m = 3.28084 ft
    assert np.isclose(feet[0], 3280.84, atol=0.01)

    back_to_m = convert_depth(feet, "FT", "M")
    assert np.allclose(meters, back_to_m)

    # Same unit
    assert np.array_equal(meters, convert_depth(meters, "M", "M"))

    # Unsupported unit
    with pytest.raises(ValueError):
        convert_depth(meters, "YARDS", "M")

def test_convert_density():
    g_cm3 = np.array([2.65, 2.71, 2.87])
    kg_m3 = convert_density(g_cm3, "g/cm3", "kg/m3")
    assert np.array_equal(kg_m3, np.array([2650.0, 2710.0, 2870.0]))

    back = convert_density(kg_m3, "kg/m3", "g/cm3")
    assert np.allclose(g_cm3, back)

def test_convert_sonic():
    us_ft = np.array([40.0, 100.0])
    # 100 us/ft * 1/0.3048 = 328.084 us/m
    us_m = convert_sonic(us_ft, "us/ft", "us/m")
    assert np.isclose(us_m[1], 328.084, atol=0.01)

    back = convert_sonic(us_m, "us/m", "us/ft")
    assert np.allclose(us_ft, back)

def test_normalize_porosity_fraction():
    # Percent input
    s_pct = pd.Series([15.0, 25.0, 30.0])
    norm_pct, warn = normalize_porosity_fraction(s_pct, declared_unit="%")
    assert np.isclose(norm_pct.iloc[0], 0.15)
    assert warn is None

    # Already decimal
    s_dec = pd.Series([0.15, 0.25, 0.30])
    norm_dec, warn = normalize_porosity_fraction(s_dec, declared_unit="v/v")
    assert np.isclose(norm_dec.iloc[0], 0.15)
    assert warn is None

    # Ambiguous unit with values > 1.0 -> should convert and warn
    norm_ambig, warn = normalize_porosity_fraction(s_pct, declared_unit=None)
    assert np.isclose(norm_ambig.iloc[0], 0.15)
    assert warn is not None
    assert "divided by 100" in warn or "percent" in warn

def test_convert_temperature():
    # 100 deg C = 212 deg F
    assert np.isclose(convert_temperature(100.0, "degC", "degF"), 212.0)
    assert np.isclose(convert_temperature(212.0, "degF", "degC"), 100.0)
    # 0 deg C = 32 deg F
    assert np.isclose(convert_temperature(0.0, "degC", "degF"), 32.0)
