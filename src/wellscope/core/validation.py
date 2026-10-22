"""
Data Validation and Quality Control Preprocessing Module.
Checks depth integrity, duplicate depths, irregular step intervals, missing value gaps, and physical range limits.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

PHYSICAL_LIMITS = {
    "GR": (0.0, 300.0, "API", "Natural gamma ray outside typical sedimentary range"),
    "RHOB": (1.50, 3.20, "g/cm3", "Bulk density outside sedimentary rock / heavy mineral bounds"),
    "NPHI": (-0.05, 0.60, "v/v", "Neutron porosity outside plausible clean matrix / high-porosity limits"),
    "DT": (40.0, 180.0, "µs/ft", "Sonic transit time outside dense limestone to unconsolidated shales"),
    "RT": (0.05, 20000.0, "ohm.m", "Resistivity non-positive or above extreme dielectric limit"),
    "RXO": (0.05, 20000.0, "ohm.m", "Flushed zone resistivity non-positive or extreme"),
    "CALI": (4.0, 26.0, "in", "Borehole caliper outside standard bit and breakout range"),
    "DRHO": (-0.20, 0.20, "g/cm3", "Density correction DRHO outside spine-and-ribs calibration window"),
}

def validate_depth_series(depth: pd.Series) -> Dict[str, Any]:
    """
    Validates depth indexing:
    - Null depths
    - Duplicate depth values
    - Monotonicity (depth inversions)
    - Irregular sampling intervals
    """
    results: Dict[str, Any] = {
        "is_valid": True,
        "total_rows": len(depth),
        "null_depth_count": int(depth.isna().sum()),
        "duplicate_count": 0,
        "inversion_count": 0,
        "step_median": None,
        "irregular_step_count": 0,
        "warnings": [],
        "errors": [],
    }

    if results["null_depth_count"] > 0:
        results["is_valid"] = False
        results["errors"].append(f"Depth column contains {results['null_depth_count']} null (NaN) values.")

    valid_depth = depth.dropna()
    if len(valid_depth) < 2:
        results["is_valid"] = False
        results["errors"].append("Insufficient valid depth samples (< 2).")
        return results

    # Check duplicates
    dups = valid_depth.duplicated()
    dup_count = int(dups.sum())
    results["duplicate_count"] = dup_count
    if dup_count > 0:
        results["warnings"].append(f"Found {dup_count} duplicate depth values. Depths must be unique.")

    # Check monotonicity
    diffs = valid_depth.diff().dropna()
    inversions = diffs[diffs <= 0]
    inv_count = int(len(inversions))
    results["inversion_count"] = inv_count
    if inv_count > 0:
        results["is_valid"] = False
        results["errors"].append(f"Depth is not monotonically increasing: found {inv_count} negative or zero steps.")

    # Check step consistency
    median_step = float(diffs.median())
    results["step_median"] = median_step
    if median_step > 0:
        # Steps deviating > 50% from median
        irregular = diffs[(diffs < 0.5 * median_step) | (diffs > 1.5 * median_step)]
        results["irregular_step_count"] = int(len(irregular))
        if len(irregular) > 0:
            results["warnings"].append(
                f"Irregular depth sampling detected in {len(irregular)} intervals "
                f"(expected ~{median_step:.3f})."
            )

    return results

def validate_curve_ranges(df: pd.DataFrame, curves_to_check: List[str] = None) -> Dict[str, Any]:
    """
    Checks curves against physical limits without silently clipping.
    Returns per-curve statistics and warnings for out-of-range values.
    """
    if curves_to_check is None:
        curves_to_check = [c for c in df.columns if c in PHYSICAL_LIMITS]

    report: Dict[str, Any] = {}
    for col in curves_to_check:
        if col not in df.columns or col not in PHYSICAL_LIMITS:
            continue
        min_lim, max_lim, unit, descr = PHYSICAL_LIMITS[col]
        series = df[col].dropna()
        if len(series) == 0:
            continue

        low_outliers = int((series < min_lim).sum())
        high_outliers = int((series > max_lim).sum())
        total_outliers = low_outliers + high_outliers

        report[col] = {
            "min_observed": float(series.min()),
            "max_observed": float(series.max()),
            "expected_min": min_lim,
            "expected_max": max_lim,
            "unit": unit,
            "total_outliers": total_outliers,
            "low_outliers": low_outliers,
            "high_outliers": high_outliers,
            "warning": (
                f"{col}: {total_outliers} sample(s) outside physical range [{min_lim}, {max_lim}] {unit}. {descr}"
                if total_outliers > 0
                else None
            ),
        }

    return report

def find_data_gaps(series: pd.Series, depth: pd.Series, min_gap_samples: int = 3) -> List[Dict[str, Any]]:
    """
    Identifies contiguous null intervals in a curve to avoid silent interpolation across large gaps.
    """
    is_null = series.isna()
    if not is_null.any():
        return []

    # Find runs of True
    gaps = []
    in_gap = False
    start_idx = 0

    for i, val in enumerate(is_null):
        if val and not in_gap:
            in_gap = True
            start_idx = i
        elif not val and in_gap:
            in_gap = False
            gap_len = i - start_idx
            if gap_len >= min_gap_samples:
                gaps.append({
                    "start_depth": float(depth.iloc[start_idx]),
                    "end_depth": float(depth.iloc[i - 1]),
                    "sample_count": gap_len,
                    "thickness": float(abs(depth.iloc[i - 1] - depth.iloc[start_idx])),
                })
    if in_gap:
        gap_len = len(series) - start_idx
        if gap_len >= min_gap_samples:
            gaps.append({
                "start_depth": float(depth.iloc[start_idx]),
                "end_depth": float(depth.iloc[-1]),
                "sample_count": gap_len,
                "thickness": float(abs(depth.iloc[-1] - depth.iloc[start_idx])),
            })

    return gaps
