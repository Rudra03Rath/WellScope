"""
LAS 2.0 Ingestion Module using lasio.
Extracts well header metadata, curve inventory, units, and depth-indexed data.
"""

from typing import Dict, Any, List, Optional
import io
import lasio
import numpy as np
import pandas as pd

class LASReaderError(Exception):
    """Custom exception for LAS parsing failures."""
    pass

def read_las(source: Any) -> Dict[str, Any]:
    """
    Parses a LAS file from a filepath, string, or file-like object / bytes.
    
    Returns a dictionary with:
        - "header": Dict of well metadata (well_name, field, location, null_value, etc.)
        - "params": Dict of logging parameters (bit_size, rmf, bht, etc.)
        - "curves_info": List of dicts for each curve (mnemonic, unit, descr, min, max, null_count)
        - "df": pandas DataFrame with original mnemonics and null values replaced with NaN
        - "warnings": List of string warnings encountered during parsing
    """
    warnings: List[str] = []
    
    try:
        if isinstance(source, (str, bytes)):
            if isinstance(source, str) and not ("\n" in source or "\r" in source):
                # File path
                las = lasio.read(source)
            else:
                # String or bytes content
                las = lasio.read(source)
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            content = source.read()
            if isinstance(content, bytes):
                # Try decoding utf-8, fallback to latin-1
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("latin-1")
                    warnings.append("LAS file decoded with latin-1 fallback encoding.")
                las = lasio.read(io.StringIO(text))
            else:
                las = lasio.read(io.StringIO(content))
    except Exception as e:
        raise LASReaderError(f"Failed to parse LAS file: {str(e)}") from e

    # Extract well metadata
    def get_item(section, mnemonic, default=None):
        try:
            item = section[mnemonic]
            return item.value if item.value is not None else default
        except KeyError:
            return default

    header = {
        "well_name": str(get_item(las.well, "WELL", "UNKNOWN_WELL")),
        "field": str(get_item(las.well, "FLD", "")),
        "location": str(get_item(las.well, "LOC", "")),
        "service_company": str(get_item(las.well, "SRVC", "")),
        "date": str(get_item(las.well, "DATE", "")),
        "null_value": float(get_item(las.well, "NULL", -999.25)),
        "start_depth": get_item(las.well, "STRT", None),
        "stop_depth": get_item(las.well, "STOP", None),
        "step": get_item(las.well, "STEP", None),
        "depth_unit": str(las.well["STRT"].unit if "STRT" in las.well and las.well["STRT"].unit else "M"),
    }

    # Extract parameters
    params = {}
    for p in las.params:
        params[p.mnemonic] = {
            "value": p.value,
            "unit": p.unit or "",
            "descr": p.descr or "",
        }

    # Extract DataFrame
    df = las.df().reset_index()
    null_val = header["null_value"]

    # In lasio, null values are typically converted to NaN automatically, but let's be exhaustive
    for col in df.columns:
        if null_val is not None and not np.isnan(null_val):
            df[col] = df[col].replace(null_val, np.nan)
        # Also handle common alternative null flags if present
        df[col] = df[col].replace([-999.25, -999.0, -9999.0, -9999.25], np.nan)

    # Extract Curve Inventory
    curves_info = []
    for curve in las.curves:
        mnem = curve.mnemonic
        unit = (curve.unit or "").strip().upper()
        descr = (curve.descr or "").strip()
        
        series = df[mnem] if mnem in df.columns else None
        if series is not None:
            valid_vals = series.dropna()
            c_info = {
                "mnemonic": mnem,
                "unit": unit,
                "descr": descr,
                "count": int(len(series)),
                "valid_count": int(len(valid_vals)),
                "null_count": int(series.isna().sum()),
                "min": float(valid_vals.min()) if len(valid_vals) > 0 else np.nan,
                "max": float(valid_vals.max()) if len(valid_vals) > 0 else np.nan,
                "mean": float(valid_vals.mean()) if len(valid_vals) > 0 else np.nan,
            }
        else:
            c_info = {
                "mnemonic": mnem,
                "unit": unit,
                "descr": descr,
                "count": 0,
                "valid_count": 0,
                "null_count": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
            }
        curves_info.append(c_info)

    if df.empty:
        warnings.append("LAS file contains no data rows.")

    return {
        "header": header,
        "params": params,
        "curves_info": curves_info,
        "df": df,
        "warnings": warnings,
        "source_type": "LAS",
    }
