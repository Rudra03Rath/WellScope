"""
Quality Control (QC) Flagging and Borehole Integrity Module.
Flags hole washouts, severe mudcake correction, bad sensor contact, and invalid measurement intervals.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

def compute_qc_flags(
    df: pd.DataFrame,
    bit_size: float = 8.5,
    washout_diff: float = 1.5,
    drho_threshold: float = 0.05,
    depth_col: str = "DEPTH",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes depth-indexed boolean QC flags:
    - QC_WASHOUT: Caliper significantly exceeds bit size (CALI > bit_size + washout_diff)
    - QC_HIGH_DRHO: Density correction absolute value exceeds threshold (|DRHO| > drho_threshold)
    - QC_BAD_HOLE: Washout or high DRHO indicating potential loss of pad contact
    - QC_VALID_POROSITY: RHOB is present, within physical limits, and free from bad hole effects
    
    Returns:
        (df_with_flags, qc_summary_dict)
    """
    flags_df = pd.DataFrame(index=df.index)

    # 1. Caliper Washout
    if "CALI" in df.columns and df["CALI"].notna().any():
        flags_df["QC_WASHOUT"] = (df["CALI"] > (bit_size + washout_diff))
    else:
        flags_df["QC_WASHOUT"] = False

    # 2. High DRHO (Density Correction)
    if "DRHO" in df.columns and df["DRHO"].notna().any():
        flags_df["QC_HIGH_DRHO"] = (df["DRHO"].abs() > drho_threshold)
    else:
        flags_df["QC_HIGH_DRHO"] = False

    # 3. Bad Hole condition (combination of washout and DRHO)
    flags_df["QC_BAD_HOLE"] = flags_df["QC_WASHOUT"] | flags_df["QC_HIGH_DRHO"]

    # 4. Valid Density Porosity Input
    if "RHOB" in df.columns and df["RHOB"].notna().any():
        valid_rhob = (df["RHOB"] >= 1.5) & (df["RHOB"] <= 3.2) & df["RHOB"].notna()
        flags_df["QC_VALID_POROSITY"] = valid_rhob & (~flags_df["QC_BAD_HOLE"])
    else:
        flags_df["QC_VALID_POROSITY"] = False

    # Summarize QC intervals
    depth = df[depth_col] if depth_col in df.columns else pd.Series(df.index)
    washout_intervals = extract_flagged_intervals(flags_df["QC_WASHOUT"], depth, "Caliper Washout")
    bad_hole_intervals = extract_flagged_intervals(flags_df["QC_BAD_HOLE"], depth, "Bad Borehole / DRHO Warning")

    summary = {
        "bit_size_in": bit_size,
        "washout_threshold_in": bit_size + washout_diff,
        "drho_threshold": drho_threshold,
        "washout_sample_count": int(flags_df["QC_WASHOUT"].sum()),
        "bad_hole_sample_count": int(flags_df["QC_BAD_HOLE"].sum()),
        "washout_intervals": washout_intervals,
        "bad_hole_intervals": bad_hole_intervals,
        "total_depth_samples": len(df),
    }

    return flags_df, summary

def extract_flagged_intervals(
    flag_series: pd.Series,
    depth_series: pd.Series,
    flag_name: str,
    min_samples: int = 2,
) -> List[Dict[str, Any]]:
    """Identifies contiguous depth zones where a boolean QC flag is True."""
    intervals = []
    in_interval = False
    start_idx = 0

    for i, flag in enumerate(flag_series):
        if flag and not in_interval:
            in_interval = True
            start_idx = i
        elif not flag and in_interval:
            in_interval = False
            count = i - start_idx
            if count >= min_samples:
                top = float(depth_series.iloc[start_idx])
                base = float(depth_series.iloc[i - 1])
                intervals.append({
                    "flag": flag_name,
                    "top_depth": min(top, base),
                    "base_depth": max(top, base),
                    "thickness": float(abs(base - top)),
                    "sample_count": count,
                })
    if in_interval:
        count = len(flag_series) - start_idx
        if count >= min_samples:
            top = float(depth_series.iloc[start_idx])
            base = float(depth_series.iloc[-1])
            intervals.append({
                "flag": flag_name,
                "top_depth": min(top, base),
                "base_depth": max(top, base),
                "thickness": float(abs(base - top)),
                "sample_count": count,
            })

    return intervals
