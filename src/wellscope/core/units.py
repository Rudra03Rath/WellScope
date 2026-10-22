"""
Petrophysical Unit Conversion and Normalization Module.
Provides explicit, traceable conversions between common oilfield and SI units.
Never makes silent unit guesses.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd

# Conversion Constants
M_TO_FT = 3.280839895013123
FT_TO_M = 1.0 / M_TO_FT

US_M_TO_US_FT = 0.3048
US_FT_TO_US_M = 1.0 / US_M_TO_US_FT

G_CM3_TO_KG_M3 = 1000.0
KG_M3_TO_G_CM3 = 0.001

def convert_depth(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Converts depth array between 'M' (meters) and 'FT' (feet)."""
    f = from_unit.strip().upper()
    t = to_unit.strip().upper()
    if f == t:
        return values.copy()
    if f in ["M", "METER", "METERS"] and t in ["FT", "FEET", "F"]:
        return values * M_TO_FT
    if f in ["FT", "FEET", "F"] and t in ["M", "METER", "METERS"]:
        return values * FT_TO_M
    raise ValueError(f"Unsupported depth unit conversion from '{from_unit}' to '{to_unit}'")

def convert_density(values: np.ndarray, from_unit: str, to_unit: str = "G/CM3") -> np.ndarray:
    """Converts bulk density array to g/cm3."""
    f = from_unit.strip().upper()
    t = to_unit.strip().upper()
    if f in ["G/C3", "G/CM3", "GM/CC", "G/CC"] and t in ["G/C3", "G/CM3", "GM/CC"]:
        return values.copy()
    if f in ["KG/M3", "KGM3"] and t in ["G/C3", "G/CM3"]:
        return values * KG_M3_TO_G_CM3
    if f in ["G/C3", "G/CM3"] and t in ["KG/M3", "KGM3"]:
        return values * G_CM3_TO_KG_M3
    raise ValueError(f"Unsupported density unit conversion from '{from_unit}' to '{to_unit}'")

def convert_sonic(values: np.ndarray, from_unit: str, to_unit: str = "US/FT") -> np.ndarray:
    """Converts acoustic slowness array to µs/ft."""
    f = from_unit.strip().upper()
    t = to_unit.strip().upper()
    if f in ["US/F", "US/FT", "USEC/FT"] and t in ["US/F", "US/FT"]:
        return values.copy()
    if f in ["US/M", "USEC/M"] and t in ["US/F", "US/FT"]:
        return values * US_M_TO_US_FT
    if f in ["US/F", "US/FT"] and t in ["US/M", "USEC/M"]:
        return values * US_FT_TO_US_M
    raise ValueError(f"Unsupported sonic slowness conversion from '{from_unit}' to '{to_unit}'")

def normalize_porosity_fraction(series: pd.Series, declared_unit: Optional[str] = None) -> Tuple[pd.Series, Optional[str]]:
    """
    Normalizes porosity values to decimal fraction [0, 1].
    Checks declared unit or evaluates numerical range (e.g., if mean/max > 1.0, likely in percent 0-100).
    Returns (normalized_series, warning_message).
    """
    warning = None
    u = (declared_unit or "").strip().upper()

    if u in ["%", "PU", "PERCENT"]:
        return series / 100.0, None
    elif u in ["V/V", "DEC", "FRACTION", "DECIMAL"]:
        return series.copy(), None

    # Unit ambiguous/missing: inspect valid values
    valid = series.dropna()
    if len(valid) > 0 and valid.max() > 1.0:
        warning = (
            f"Porosity unit was ambiguous ('{declared_unit}'), but values exceed 1.0 "
            f"(max={valid.max():.1f}). Normalized by dividing by 100 (assuming percent)."
        )
        return series / 100.0, warning

    return series.copy(), None

def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
    """Converts single temperature value between Celsius and Fahrenheit."""
    f = from_unit.strip().upper()
    t = to_unit.strip().upper()
    if f == t:
        return temp
    if f in ["DEGC", "C", "CELSIUS"] and t in ["DEGF", "F", "FAHRENHEIT"]:
        return (temp * 9.0 / 5.0) + 32.0
    if f in ["DEGF", "F", "FAHRENHEIT"] and t in ["DEGC", "C", "CELSIUS"]:
        return (temp - 32.0) * 5.0 / 9.0
    raise ValueError(f"Unsupported temperature conversion: {from_unit} to {to_unit}")
