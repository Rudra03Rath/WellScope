"""
Preprocessing and QC Pipeline Module.
Combines data validation, curve normalization, gap detection, and borehole QC.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from wellscope.core.units import convert_depth, convert_density, convert_sonic, normalize_porosity_fraction
from wellscope.core.validation import validate_depth_series, validate_curve_ranges, find_data_gaps
from wellscope.core.qc_flags import compute_qc_flags
from wellscope.io.curve_mapping import apply_mapping, validate_mapping

def preprocess_well_data(
    raw_df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    depth_unit: str = "M",
    target_depth_unit: str = "M",
    bit_size: float = 8.5,
    washout_diff: float = 1.5,
) -> Dict[str, Any]:
    """
    Executes the full validation and preprocessing sequence:
    1. Curve mapping
    2. Depth integrity validation (duplicates, monotonicity)
    3. Unit conversions (if requested)
    4. Outlier detection across physical limits
    5. Missing-data gap detection
    6. Borehole QC flags computation (washout, high DRHO, bad hole)
    
    Returns structured results:
        - "clean_df": pd.DataFrame with mapped curves and QC flags
        - "depth_validation": Dict from validate_depth_series
        - "curve_validation": Dict from validate_curve_ranges
        - "gaps": Dict of gaps per curve
        - "qc_summary": Dict of borehole condition metrics
        - "warnings": List of string warnings
    """
    warnings: List[str] = []

    # 1. Apply Curve Mapping
    df_mapped = apply_mapping(raw_df, mapping)
    is_valid_map, map_warnings = validate_mapping(mapping, list(raw_df.columns))
    warnings.extend(map_warnings)

    if not is_valid_map:
        return {
            "clean_df": df_mapped,
            "depth_validation": {"is_valid": False, "errors": ["Mapping missing valid DEPTH."]},
            "curve_validation": {},
            "gaps": {},
            "qc_summary": {},
            "warnings": warnings,
        }

    # 2. Validate Depth
    depth_val = validate_depth_series(df_mapped["DEPTH"])
    warnings.extend(depth_val["warnings"])
    warnings.extend(depth_val["errors"])

    # 3. Unit Conversions
    if depth_unit.upper() != target_depth_unit.upper():
        try:
            df_mapped["DEPTH"] = convert_depth(df_mapped["DEPTH"].values, depth_unit, target_depth_unit)
        except Exception as e:
            warnings.append(f"Depth unit conversion failed: {str(e)}")

    # Normalize NPHI if in percent (e.g. max > 1.0)
    if "NPHI" in df_mapped.columns and df_mapped["NPHI"].notna().any():
        df_mapped["NPHI"], nphi_warn = normalize_porosity_fraction(df_mapped["NPHI"])
        if nphi_warn:
            warnings.append(nphi_warn)

    # 4. Physical Curve Limits
    curve_val = validate_curve_ranges(df_mapped)
    for col, res in curve_val.items():
        if res.get("warning"):
            warnings.append(res["warning"])

    # 5. Data Gaps
    gaps = {}
    for col in ["GR", "RHOB", "NPHI", "RT", "DT"]:
        if col in df_mapped.columns and df_mapped[col].notna().any():
            col_gaps = find_data_gaps(df_mapped[col], df_mapped["DEPTH"], min_gap_samples=3)
            if col_gaps:
                gaps[col] = col_gaps
                warnings.append(f"{col} has {len(col_gaps)} missing-data gap(s) exceeding 3 samples.")

    # 6. Borehole QC Flags
    flags_df, qc_summary = compute_qc_flags(
        df_mapped,
        bit_size=bit_size,
        washout_diff=washout_diff,
        depth_col="DEPTH",
    )

    # Attach QC flags to DataFrame
    for col in flags_df.columns:
        df_mapped[col] = flags_df[col]

    return {
        "clean_df": df_mapped,
        "depth_validation": depth_val,
        "curve_validation": curve_val,
        "gaps": gaps,
        "qc_summary": qc_summary,
        "warnings": warnings,
    }
