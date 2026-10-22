"""
Unit tests for porosity calculations (Density, Sonic Wyllie, Neutron-Density combinations).
"""

import pytest
import numpy as np
import pandas as pd
from wellscope.petrophysics.porosity import (
    calculate_density_porosity,
    calculate_sonic_porosity,
    calculate_neutron_density_porosity,
)

def test_density_porosity_hand_calc():
    # RHOMA=2.65, RHOF=1.00, RHOB=2.20 -> PHID = (2.65-2.20)/(2.65-1.00) = 0.45/1.65 = 0.272727
    rhob = pd.Series([2.20, 2.65])
    phid, warnings = calculate_density_porosity(rhob, rhoma=2.65, rhof=1.00)

    assert np.isclose(phid.iloc[0], 0.272727, atol=1e-5)
    assert np.isclose(phid.iloc[1], 0.0, atol=1e-5)
    assert len(warnings) == 0

def test_density_porosity_outliers_warning():
    # RHOB=2.80 (> 2.65 -> negative por), RHOB=1.80 (< 1.9 -> high por 0.515)
    rhob = pd.Series([2.80, 1.80])
    phid, warnings = calculate_density_porosity(rhob, rhoma=2.65, rhof=1.00)

    assert phid.iloc[0] < 0.0
    assert phid.iloc[1] > 0.45
    assert len(warnings) == 2
    assert any("negative density porosity" in w for w in warnings)
    assert any("unusually high density porosity" in w for w in warnings)

def test_density_porosity_invalid_params():
    rhob = pd.Series([2.30])
    with pytest.raises(ValueError):
        calculate_density_porosity(rhob, rhoma=1.0, rhof=1.0)
    with pytest.raises(ValueError):
        calculate_density_porosity(rhob, rhoma=0.9, rhof=1.0)

def test_sonic_porosity_hand_calc():
    # DTMA=55.5, DTF=189.0, DT=88.875 -> PHIS = (88.875 - 55.5) / (189 - 55.5) = 33.375 / 133.5 = 0.250
    dt = pd.Series([88.875, 55.5])
    phis, warnings = calculate_sonic_porosity(dt, dt_matrix=55.5, dt_fluid=189.0)

    assert np.isclose(phis.iloc[0], 0.250, atol=1e-5)
    assert np.isclose(phis.iloc[1], 0.0, atol=1e-5)

def test_neutron_density_combination():
    nphi = pd.Series([0.16])
    phid = pd.Series([0.24])

    phi_nd, phi_gas, warnings = calculate_neutron_density_porosity(
        nphi, phid, nphi_matrix_basis="limestone", target_matrix="sandstone"
    )

    # phi_nd = (0.16 + 0.24) / 2 = 0.20
    assert np.isclose(phi_nd.iloc[0], 0.20)

    # phi_gas = sqrt((0.16^2 + 0.24^2) / 2) = sqrt((0.0256 + 0.0576)/2) = sqrt(0.0416) = 0.20396078
    assert np.isclose(phi_gas.iloc[0], 0.20396, atol=1e-4)

    # Warning for matrix mismatch
    assert len(warnings) == 1
    assert "Lithology basis mismatch" in warnings[0]
