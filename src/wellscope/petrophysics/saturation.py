"""
Water Saturation (Archie) and Formation Water Resistivity (Rw) Module.
Implements the classic Archie (1942) equation and Arp's temperature adjustment.
Adheres strictly to petrophysical transparency:
- Preserves raw Sw for diagnostics
- Warns if Sw exceeds physical limits [0, 1]
- Emits explicit warnings when applied to shaly intervals (where Archie overestimates Sw)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

def arps_temperature_correction(
    rw_at_t1: float,
    t1: float,
    t2: float,
    unit: str = "DEGC",
) -> float:
    """
    Adjusts water resistivity (Rw) from temperature T1 to temperature T2 using Arp's empirical formula:
    - For Celsius: Rw2 = Rw1 * (T1 + 21.5) / (T2 + 21.5)
    - For Fahrenheit: Rw2 = Rw1 * (T1 + 6.77) / (T2 + 6.77)
    """
    if rw_at_t1 <= 0:
        raise ValueError(f"Rw must be positive (> 0), got {rw_at_t1}")

    u = unit.strip().upper()
    if u in ["DEGC", "C", "CELSIUS"]:
        if t1 < -21.5 or t2 < -21.5:
            raise ValueError("Temperatures in Celsius must be > -21.5 °C for Arp's formula.")
        return rw_at_t1 * (t1 + 21.5) / (t2 + 21.5)
    elif u in ["DEGF", "F", "FAHRENHEIT"]:
        if t1 < -6.77 or t2 < -6.77:
            raise ValueError("Temperatures in Fahrenheit must be > -6.77 °F for Arp's formula.")
        return rw_at_t1 * (t1 + 6.77) / (t2 + 6.77)
    else:
        raise ValueError(f"Unsupported temperature unit '{unit}'. Use 'DEGC' or 'DEGF'.")

def calculate_archie_sw(
    rt: pd.Series,
    phie: pd.Series,
    rw: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    vsh: Optional[pd.Series] = None,
    vsh_shaly_threshold: float = 0.20,
) -> Tuple[pd.Series, pd.Series, List[str], Dict[str, Any]]:
    """
    Calculates water saturation using Archie's equation:
    Sw = ((a * Rw) / (phie^m * Rt))^(1 / n)
    
    Parameters:
        rt: Deep resistivity proxy (ohm.m)
        phie: Effective porosity (fraction)
        rw: Formation water resistivity at reservoir temperature (ohm.m)
        a: Tortuosity / lithology factor (typical 0.6 - 1.2)
        m: Cementation exponent (typical 1.8 - 2.2)
        n: Saturation exponent (typical 1.8 - 2.2)
        vsh: Optional shale volume series for shaly-sand qualification warnings
        vsh_shaly_threshold: Vsh limit above which Archie Sw warnings are emitted
        
    Returns:
        (sw_raw, sw_clipped, warnings_list, diagnostics_dict)
    """
    warnings: List[str] = []

    # Validate parameters
    if rw <= 0:
        raise ValueError(f"Formation water resistivity Rw must be strictly positive (> 0), got {rw}")
    if a <= 0:
        raise ValueError(f"Tortuosity factor 'a' must be positive, got {a}")
    if m <= 0:
        raise ValueError(f"Cementation exponent 'm' must be positive, got {m}")
    if n <= 0:
        raise ValueError(f"Saturation exponent 'n' must be positive, got {n}")

    # Check for non-standard Archie ranges
    if not (0.5 <= a <= 1.5):
        warnings.append(f"Tortuosity factor a={a} is outside standard empirical range [0.5, 1.5].")
    if not (1.3 <= m <= 2.8):
        warnings.append(f"Cementation exponent m={m} is outside typical range [1.3, 2.8].")
    if not (1.3 <= n <= 2.8):
        warnings.append(f"Saturation exponent n={n} is outside typical range [1.3, 2.8].")

    # Safe evaluation mask
    # phie must be > 0 and rt must be > 0
    valid_mask = (phie > 0.0) & (rt > 0.0) & phie.notna() & rt.notna()

    # Pre-allocate output series
    sw_raw = pd.Series(index=rt.index, dtype=float)

    # Vectorized calculation on valid mask
    p = phie[valid_mask]
    r = rt[valid_mask]

    denom = np.power(p, m) * r
    f_factor = (a * rw) / denom
    # Ensure no negative values inside root
    f_factor = np.clip(f_factor, 0.0, None)
    sw_raw.loc[valid_mask] = np.power(f_factor, 1.0 / n)

    # Diagnostic inspection of raw Sw
    valid_sw = sw_raw.dropna()
    over_1_count = int((valid_sw > 1.0).sum())
    under_0_count = int((valid_sw < 0.0).sum())

    if over_1_count > 0:
        warnings.append(
            f"{over_1_count} sample(s) had unclipped Archie Sw > 1.0 (max raw={valid_sw.max():.2f}). "
            f"Possible causes: assumed Rw is too high, porosity is underestimated, or formation contains low-salinity water."
        )

    # Shaly-sand qualification warning
    if vsh is not None:
        shaly_samples = ((vsh > vsh_shaly_threshold) & valid_mask).sum()
        if shaly_samples > 0:
            warnings.append(
                f"Shaly formation warning: {shaly_samples} sample(s) have Vsh > {vsh_shaly_threshold:.2f}. "
                f"Archie's model does not account for clay electrical conductivity and may overestimate Sw "
                f"(underestimating hydrocarbon pay). Use results with caution."
            )

    # Display curve clipped to [0, 1]
    sw_clipped = sw_raw.clip(lower=0.0, upper=1.0)

    diagnostics = {
        "a": a,
        "m": m,
        "n": n,
        "rw": rw,
        "total_evaluated": int(len(valid_sw)),
        "samples_sw_gt_1": over_1_count,
        "samples_sw_lt_0": under_0_count,
        "mean_raw_sw": float(valid_sw.mean()) if len(valid_sw) > 0 else np.nan,
        "mean_clipped_sw": float(sw_clipped.dropna().mean()) if len(sw_clipped.dropna()) > 0 else np.nan,
    }

    return sw_raw, sw_clipped, warnings, diagnostics
