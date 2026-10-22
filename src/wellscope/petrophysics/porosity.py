"""
Porosity Calculation and Cross-Check Module.
Implements:
1. Density Porosity (PHID)
2. Wyllie Time-Average Sonic Porosity (PHIS)
3. Neutron-Density Combination Porosities (Standard and Gas-corrected)
4. Out-of-bounds, lithology mismatch, and borehole effect diagnostics.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Standard Lithology Matrix Constants
STANDARD_MATRICES = {
    "sandstone": {
        "name": "Sandstone (Quartz)",
        "rhoma": 2.65,  # g/cm3
        "dtma": 55.5,   # µs/ft
        "descr": "Pure quartz sandstone matrix",
    },
    "limestone": {
        "name": "Limestone (Calcite)",
        "rhoma": 2.71,  # g/cm3
        "dtma": 47.5,   # µs/ft
        "descr": "Pure calcite limestone matrix (standard logging calibration basis)",
    },
    "dolomite": {
        "name": "Dolomite",
        "rhoma": 2.87,  # g/cm3
        "dtma": 43.5,   # µs/ft
        "descr": "Dense dolomite matrix",
    },
    "anhydrite": {
        "name": "Anhydrite",
        "rhoma": 2.98,
        "dtma": 50.0,
        "descr": "Dense evaporite",
    },
}

STANDARD_FLUIDS = {
    "fresh_water": {"name": "Fresh Water Mud", "rhof": 1.00, "dtf": 189.0},
    "salt_water": {"name": "Salt Water Mud (Brine)", "rhof": 1.10, "dtf": 185.0},
    "oil_base": {"name": "Oil-Based Mud (OBM)", "rhof": 0.85, "dtf": 200.0},
}

def calculate_density_porosity(
    rhob: pd.Series,
    rhoma: float = 2.65,
    rhof: float = 1.00,
) -> Tuple[pd.Series, List[str]]:
    """
    Calculates Density Porosity:
    PHID = (RHOMA - RHOB) / (RHOMA - RHOF)
    
    Does NOT silently clip or conceal negative or extreme values.
    Returns (phid_series, warnings_list).
    """
    warnings: List[str] = []

    if np.isclose(rhoma, rhof):
        raise ValueError(f"Matrix density ({rhoma}) cannot equal fluid density ({rhof}). Division by zero.")
    if rhoma <= rhof:
        raise ValueError(f"Matrix density ({rhoma}) must be strictly greater than fluid density ({rhof}).")

    denom = rhoma - rhof
    phid = (rhoma - rhob) / denom

    # Diagnostic checks
    valid = phid.dropna()
    neg_count = (valid < 0.0).sum()
    high_count = (valid > 0.45).sum()

    if neg_count > 0:
        warnings.append(
            f"{neg_count} sample(s) had negative density porosity (PHID < 0). "
            f"Possible causes: heavy minerals (pyrite, siderite), dense matrix (dolomite/anhydrite), or incorrect RHOMA."
        )
    if high_count > 0:
        warnings.append(
            f"{high_count} sample(s) had unusually high density porosity (PHID > 0.45). "
            f"Possible causes: gas effect (low density gas), borehole washout, or unconsolidated sand."
        )

    return phid, warnings

def calculate_sonic_porosity(
    dt: pd.Series,
    dt_matrix: float = 55.5,
    dt_fluid: float = 189.0,
) -> Tuple[pd.Series, List[str]]:
    """
    Calculates Sonic Porosity using Wyllie Time-Average:
    PHIS = (DT_log - DT_matrix) / (DT_fluid - DT_matrix)
    
    Returns (phis_series, warnings_list).
    """
    warnings: List[str] = []

    if np.isclose(dt_fluid, dt_matrix):
        raise ValueError(f"DT_fluid ({dt_fluid}) cannot equal DT_matrix ({dt_matrix}). Division by zero.")
    if dt_fluid <= dt_matrix:
        raise ValueError(f"DT_fluid ({dt_fluid}) must be strictly greater than DT_matrix ({dt_matrix}).")

    denom = dt_fluid - dt_matrix
    phis = (dt - dt_matrix) / denom

    valid = phis.dropna()
    neg_count = (valid < 0.0).sum()
    high_count = (valid > 0.45).sum()

    if neg_count > 0:
        warnings.append(f"{neg_count} sample(s) had negative sonic porosity (DT < DT_matrix).")
    if high_count > 0:
        warnings.append(f"{high_count} sample(s) had high sonic porosity (PHIS > 0.45). Wyllie overestimates in uncompacted or gas sands.")

    return phis, warnings

def calculate_neutron_density_porosity(
    nphi: pd.Series,
    phid: pd.Series,
    nphi_matrix_basis: str = "limestone",
    target_matrix: str = "sandstone",
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """
    Computes neutron-density combination porosity:
    1. Standard clean average: PHI_ND = (NPHI + PHID) / 2
    2. Gas-corrected combination: PHI_ND_GAS = sqrt((NPHI^2 + PHID^2) / 2)
    
    Emits warnings if NPHI matrix calibration differs from target lithology.
    """
    warnings: List[str] = []

    if nphi_matrix_basis.lower() != target_matrix.lower():
        warnings.append(
            f"Lithology basis mismatch: NPHI is calibrated on a '{nphi_matrix_basis}' matrix, "
            f"whereas target interpretation is '{target_matrix}'. "
            f"Crossover and porosity values may contain lithology-induced shifts."
        )

    # Standard arithmetic average
    phi_nd = (nphi + phid) / 2.0

    # Gas-corrected (Gaymard-Poupon formula):
    # Clips negative inputs to 0 before squaring for numerical stability
    nphi_safe = nphi.clip(lower=0.0)
    phid_safe = phid.clip(lower=0.0)
    phi_nd_gas = np.sqrt((np.square(nphi_safe) + np.square(phid_safe)) / 2.0)

    return phi_nd, phi_nd_gas, warnings
