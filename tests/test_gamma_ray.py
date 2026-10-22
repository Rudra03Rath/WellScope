"""
Unit tests for Gamma-Ray index and Shale Volume models with hand calculations.
"""

import pytest
import numpy as np
import pandas as pd
from wellscope.petrophysics.gamma_ray import (
    calculate_gamma_ray_index,
    calculate_shale_volume,
    VSH_MODELS,
)

def test_gamma_ray_index_basic():
    gr = pd.Series([20.0, 70.0, 120.0])
    gr_clean = 20.0
    gr_shale = 120.0
    igr, warnings = calculate_gamma_ray_index(gr, gr_clean, gr_shale)

    assert np.isclose(igr.iloc[0], 0.0)
    assert np.isclose(igr.iloc[1], 0.5)
    assert np.isclose(igr.iloc[2], 1.0)
    assert len(warnings) == 0

def test_gamma_ray_index_out_of_bounds_diagnostics():
    gr = pd.Series([10.0, 150.0])
    gr_clean = 20.0
    gr_shale = 120.0
    igr, warnings = calculate_gamma_ray_index(gr, gr_clean, gr_shale)

    # Raw IGR is preserved without clipping
    assert igr.iloc[0] < 0.0
    assert igr.iloc[1] > 1.0
    assert len(warnings) == 2
    assert any("GR < GR_clean" in w for w in warnings)
    assert any("GR > GR_shale" in w for w in warnings)

def test_gamma_ray_index_invalid_baselines():
    gr = pd.Series([50.0])
    with pytest.raises(ValueError):
        calculate_gamma_ray_index(gr, gr_clean=100.0, gr_shale=50.0)
    with pytest.raises(ValueError):
        calculate_gamma_ray_index(gr, gr_clean=50.0, gr_shale=50.0)

def test_shale_volume_models_hand_calculated():
    igr = pd.Series([0.0, 0.5, 1.0])

    # 1. Linear
    vsh_lin, _ = calculate_shale_volume(igr, model="linear")
    assert np.isclose(vsh_lin.iloc[0], 0.0)
    assert np.isclose(vsh_lin.iloc[1], 0.50)
    assert np.isclose(vsh_lin.iloc[2], 1.0)

    # 2. Larionov Tertiary (hand calculation: at 0.5 is 0.083*(2^1.85 - 1) = 0.2162)
    vsh_tert, _ = calculate_shale_volume(igr, model="larionov_tertiary")
    assert np.isclose(vsh_tert.iloc[0], 0.0)
    assert np.isclose(vsh_tert.iloc[1], 0.2162, atol=0.005)
    assert np.isclose(vsh_tert.iloc[2], 1.0, atol=0.01)

    # 3. Larionov Older (hand calculation: at 0.5 is 0.33*(2^1.0 - 1) = 0.33)
    vsh_old, _ = calculate_shale_volume(igr, model="larionov_older")
    assert np.isclose(vsh_old.iloc[0], 0.0)
    assert np.isclose(vsh_old.iloc[1], 0.33, atol=0.005)
    assert np.isclose(vsh_old.iloc[2], 1.0, atol=0.01)

    # 4. Clavier (hand calculation: at 0.5 is 1.7 - sqrt(3.38 - 1.44) = 1.7 - 1.3928 = 0.3072)
    vsh_clav, _ = calculate_shale_volume(igr, model="clavier")
    assert np.isclose(vsh_clav.iloc[0], 0.0)
    assert np.isclose(vsh_clav.iloc[1], 0.3072, atol=0.005)
    assert np.isclose(vsh_clav.iloc[2], 1.0, atol=0.005)

    # 5. Stieber (hand calculation: at 0.5 is 0.5 / (3 - 1) = 0.25)
    vsh_stieb, _ = calculate_shale_volume(igr, model="stieber")
    assert np.isclose(vsh_stieb.iloc[0], 0.0)
    assert np.isclose(vsh_stieb.iloc[1], 0.25, atol=0.005)
    assert np.isclose(vsh_stieb.iloc[2], 1.0, atol=0.005)

def test_unknown_vsh_model():
    igr = pd.Series([0.5])
    with pytest.raises(ValueError):
        calculate_shale_volume(igr, model="non_existent_model")
