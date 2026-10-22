"""
CSV Well Log Ingestion Module.
Parses CSV files, detects depth column, and extracts curve inventories.
"""

from typing import Dict, Any, List
import io
import numpy as np
import pandas as pd

class CSVReaderError(Exception):
    """Custom exception for CSV parsing failures."""
    pass

def read_csv_log(source: Any, depth_col_override: str = None) -> Dict[str, Any]:
    """
    Parses a CSV file containing depth-indexed wireline logs.
    
    Returns:
        Dict with header, curves_info, df, warnings, and source_type.
    """
    warnings: List[str] = []
    
    try:
        if isinstance(source, (str, bytes)):
            if isinstance(source, str) and not ("\n" in source or "\r" in source):
                df = pd.read_csv(source)
            else:
                df = pd.read_csv(io.StringIO(source if isinstance(source, str) else source.decode("utf-8")))
        else:
            content = source.read()
            if isinstance(content, bytes):
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("latin-1")
                    warnings.append("CSV file decoded with latin-1 fallback.")
                df = pd.read_csv(io.StringIO(text))
            else:
                df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        raise CSVReaderError(f"Failed to read CSV log file: {str(e)}") from e

    # Replace common null value sentinels with NaN
    df = df.replace([-999.25, -999.0, -9999.0, -9999.25, "-999.25", "-999", "NULL", "null", "NaN", "nan"], np.nan)

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect depth column
    candidate_depth_cols = ["DEPTH", "DEPT", "MD", "DEPTH_M", "DEPTH_FT", "DEPTH.M", "DEPTH.FT"]
    depth_col = None

    if depth_col_override and depth_col_override in df.columns:
        depth_col = depth_col_override
    else:
        for c in df.columns:
            if c.strip().upper() in candidate_depth_cols:
                depth_col = c
                break

    if depth_col is None:
        warnings.append("Could not automatically identify a depth column (e.g., DEPTH, DEPT, MD). Please specify manually.")
        start_d, stop_d, step_d = None, None, None
    else:
        # Move depth column to first position
        cols = [depth_col] + [c for c in df.columns if c != depth_col]
        df = df[cols]
        valid_depths = df[depth_col].dropna()
        start_d = float(valid_depths.iloc[0]) if len(valid_depths) > 0 else None
        stop_d = float(valid_depths.iloc[-1]) if len(valid_depths) > 0 else None
        step_d = float(valid_depths.diff().median()) if len(valid_depths) > 1 else None

    header = {
        "well_name": "CSV_IMPORTED_WELL",
        "field": "",
        "location": "",
        "service_company": "",
        "date": "",
        "null_value": np.nan,
        "start_depth": start_d,
        "stop_depth": stop_d,
        "step": step_d,
        "depth_unit": "M",  # Default candidate unit, requires user confirmation
    }

    curves_info = []
    for col in df.columns:
        series = df[col]
        valid_vals = series.dropna()
        curves_info.append({
            "mnemonic": col,
            "unit": "",
            "descr": "Imported CSV column",
            "count": int(len(series)),
            "valid_count": int(len(valid_vals)),
            "null_count": int(series.isna().sum()),
            "min": float(valid_vals.min()) if len(valid_vals) > 0 else np.nan,
            "max": float(valid_vals.max()) if len(valid_vals) > 0 else np.nan,
            "mean": float(valid_vals.mean()) if len(valid_vals) > 0 else np.nan,
        })

    return {
        "header": header,
        "params": {},
        "curves_info": curves_info,
        "df": df,
        "warnings": warnings,
        "source_type": "CSV",
    }
