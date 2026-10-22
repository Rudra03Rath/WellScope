"""
Data Export Module.
Generates CSV tables for derived petrophysical curves and candidate pay intervals,
as well as a JSON parameter manifest for audit-ready reproducibility.
"""

from typing import Dict, Any, List
import json
import pandas as pd

def export_derived_curves_csv(df: pd.DataFrame) -> str:
    """Exports DataFrame containing raw and derived curves to CSV string."""
    cols_priority = [
        "DEPTH", "GR", "VSH", "RHOB", "PHID", "NPHI", "DT", "PHIS", "PHIE",
        "RT", "RXO", "SW", "CANDIDATE_PAY", "QC_WASHOUT", "QC_BAD_HOLE"
    ]
    present_cols = [c for c in cols_priority if c in df.columns]
    other_cols = [c for c in df.columns if c not in present_cols]
    final_cols = present_cols + other_cols

    export_df = df[final_cols].copy()
    return export_df.to_csv(index=False)

def export_intervals_csv(intervals: List[Dict[str, Any]]) -> str:
    """Exports candidate pay and evaluated intervals table to CSV string."""
    if not intervals:
        return "status,top_depth,base_depth,net_pay_thickness,avg_vsh,avg_phie,avg_sw,rejection_reason\n"
    df_int = pd.DataFrame(intervals)
    cols_order = ["status", "top_depth", "base_depth", "net_pay_thickness", "avg_vsh", "avg_phie", "avg_sw", "rejection_reason"]
    existing_cols = [c for c in cols_order if c in df_int.columns]
    other_cols = [c for c in df_int.columns if c not in existing_cols]
    df_int = df_int[existing_cols + other_cols]
    return df_int.to_csv(index=False)

def export_parameter_manifest(manifest: Dict[str, Any]) -> str:
    """Serializes interpretation parameters and software metadata into formatted JSON."""
    return json.dumps(manifest, indent=2, default=str)
