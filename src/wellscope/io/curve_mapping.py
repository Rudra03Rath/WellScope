"""
Curve Mapping and Mnemonic Standardization Module.
Maps vendor-specific log mnemonics to canonical petrophysical curves with explicit user confirmation.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Canonical petrophysical curves
CANONICAL_CURVES = {
    "DEPTH": {
        "name": "Depth Index",
        "required": True,
        "default_units": ["M", "FT"],
        "description": "Measured depth or true vertical depth index",
        "aliases": ["DEPTH", "DEPT", "MD", "TVD", "TVDSS", "INDEX"],
    },
    "GR": {
        "name": "Gamma Ray",
        "required": True,  # Required for Vsh
        "default_units": ["GAPI", "API"],
        "description": "Total natural gamma radiation (shale indicator)",
        "aliases": ["GR", "GAM", "SGR", "CGR", "GAMMA", "GRD", "ECGR", "HSGR"],
    },
    "RHOB": {
        "name": "Bulk Density",
        "required": False,  # Required for density porosity, but app shouldn't crash if missing
        "default_units": ["G/C3", "G/CM3", "GM/CC", "KG/M3"],
        "description": "Formation bulk density from gamma-gamma scattering",
        "aliases": ["RHOB", "RHOZ", "DEN", "DENS", "RHO8", "RHOM", "BDEN"],
    },
    "NPHI": {
        "name": "Neutron Porosity",
        "required": False,
        "default_units": ["V/V", "DEC", "%", "PU"],
        "description": "Neutron porosity index (hydrogen index)",
        "aliases": ["NPHI", "TNPH", "NPOR", "CNC", "NEUT", "NPHI_LS", "NPHI_SS", "NPRL"],
    },
    "DT": {
        "name": "Sonic Slowness",
        "required": False,
        "default_units": ["US/F", "US/FT", "US/M"],
        "description": "Compressional acoustic travel time / slowness",
        "aliases": ["DT", "DTC", "AC", "DTCO", "SONIC", "DT24", "DT4P", "DTS"],
    },
    "RT": {
        "name": "Deep Resistivity (Rt proxy)",
        "required": False,  # Required for Archie Sw
        "default_units": ["OHMM", "OHM-M", "OHM*M", "MOHM"],
        "description": "Deep induction/laterolog resistivity proxy for uninvaded zone (Rt)",
        "aliases": ["RT", "RILD", "RLLD", "AT90", "ILD", "LLD", "AHT90", "RD", "RDEP", "RES_DEEP", "HLLD", "IDPH"],
    },
    "RXO": {
        "name": "Shallow Resistivity (Rxo)",
        "required": False,
        "default_units": ["OHMM", "OHM-M", "OHM*M"],
        "description": "Shallow/flushed zone resistivity",
        "aliases": ["RXO", "RLA1", "MSFL", "AT10", "ILS", "LLS", "AHT10", "RS", "RSHAL", "RES_SHAL", "SFLU"],
    },
    "CALI": {
        "name": "Caliper",
        "required": False,
        "default_units": ["IN", "INCH", "MM", "CM"],
        "description": "Borehole diameter measurement",
        "aliases": ["CALI", "CAL", "C1", "C2", "HCAL", "CALX", "HDAR"],
    },
    "SP": {
        "name": "Spontaneous Potential",
        "required": False,
        "default_units": ["MV", "V"],
        "description": "Natural electric potential difference across mudcake/formation",
        "aliases": ["SP", "SPC"],
    },
    "DRHO": {
        "name": "Density Correction",
        "required": False,
        "default_units": ["G/C3", "G/CM3"],
        "description": "Spine-and-ribs density correction for mudcake/washout QC",
        "aliases": ["DRHO", "DCOR", "HDRA", "RHOC"],
    },
}

def suggest_curve_mapping(available_mnemonics: List[str]) -> Dict[str, Optional[str]]:
    """
    Analyzes available mnemonics and returns a dictionary mapping canonical names to matched mnemonics.
    Does not finalize without user confirmation.
    """
    mapping: Dict[str, Optional[str]] = {canon: None for canon in CANONICAL_CURVES}
    used_mnemonics = set()

    for canon, info in CANONICAL_CURVES.items():
        aliases = [a.upper() for a in info["aliases"]]
        # Exact match first
        for m in available_mnemonics:
            m_clean = m.strip().upper()
            if m_clean in aliases and m not in used_mnemonics:
                mapping[canon] = m
                used_mnemonics.add(m)
                break
        
        # Substring heuristic if not matched
        if mapping[canon] is None:
            for m in available_mnemonics:
                m_clean = m.strip().upper()
                if m not in used_mnemonics and any(a in m_clean for a in aliases if len(a) >= 3):
                    mapping[canon] = m
                    used_mnemonics.add(m)
                    break

    return mapping

def validate_mapping(mapping: Dict[str, Optional[str]], available_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Validates curve mapping against requirements.
    Emits warnings if required curves or key petrophysical inputs are absent.
    """
    warnings: List[str] = []
    is_valid = True

    # DEPTH is strictly required
    if not mapping.get("DEPTH") or mapping["DEPTH"] not in available_cols:
        warnings.append("Critical: DEPTH curve is not mapped or missing from dataset.")
        is_valid = False

    # Check petrophysical prerequisites
    if not mapping.get("GR"):
        warnings.append("Warning: Gamma Ray (GR) is unmapped. Shale volume (Vsh) calculations will be unavailable.")

    if not mapping.get("RHOB"):
        warnings.append("Warning: Bulk Density (RHOB) is unmapped. Density porosity (PHID) will be unavailable.")

    if not mapping.get("RT"):
        warnings.append("Warning: True Formation Resistivity (RT) is unmapped. Archie water saturation (Sw) will be unavailable.")

    if not mapping.get("CALI"):
        warnings.append("Note: Caliper (CALI) is unmapped. Borehole washout detection will be disabled.")

    return is_valid, warnings

def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Returns a copy of the dataframe with canonical column names added for ease of downstream calculation.
    Preserves original columns untouched to prevent data loss.
    """
    df_mapped = df.copy()
    for canon, src_col in mapping.items():
        if src_col and src_col in df.columns:
            df_mapped[canon] = df[src_col]
        else:
            df_mapped[canon] = pd.Series(index=df.index, dtype=float)
    return df_mapped
