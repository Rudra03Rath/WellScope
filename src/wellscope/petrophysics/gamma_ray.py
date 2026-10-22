"""
Gamma-Ray Index (IGR) and Shale Volume (Vsh) Calculation Module.
Implements traceable, documented petrophysical equations:
1. Linear: Vsh = IGR
2. Larionov (Tertiary / Unconsolidated): Vsh = 0.083 * (2^(3.7 * IGR) - 1)
3. Larionov (Older / Mesozoic & Paleozoic): Vsh = 0.33 * (2^(2.0 * IGR) - 1)
4. Clavier et al. (1971): Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)
5. Stieber (1970): Vsh = IGR / (3 - 2 * IGR)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

VSH_MODELS = {
    "linear": {
        "name": "Linear (Vsh = IGR)",
        "descr": "Upper-bound conservative estimate; standard benchmark when clay mineralogy is uncalibrated.",
        "applicability": "General default / conservative screening across all basins.",
    },
    "larionov_tertiary": {
        "name": "Larionov (Tertiary / Young)",
        "descr": "Vsh = 0.083 * (2^(3.7 * IGR) - 1)",
        "applicability": "Unconsolidated, younger Cenozoic rocks (e.g. Gulf of Mexico, Niger Delta).",
    },
    "larionov_older": {
        "name": "Larionov (Older / Mesozoic)",
        "descr": "Vsh = 0.33 * (2^(2.0 * IGR) - 1)",
        "applicability": "Consolidated, older Mesozoic and Paleozoic formations (e.g. North Sea Brent).",
    },
    "clavier": {
        "name": "Clavier et al. (1971)",
        "descr": "Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)",
        "applicability": "Widely cited empirical non-linear response for dispersed shaly sands.",
    },
    "stieber": {
        "name": "Stieber (1970)",
        "descr": "Vsh = IGR / (3.0 - 2.0 * IGR)",
        "applicability": "Soft to moderately consolidated sands; popular in Tertiary deltaic settings.",
    },
}

def calculate_gamma_ray_index(
    gr: pd.Series,
    gr_clean: float,
    gr_shale: float,
) -> Tuple[pd.Series, List[str]]:
    """
    Calculates the unclipped Gamma Ray Index (IGR):
    IGR = (GR_log - GR_clean) / (GR_shale - GR_clean)
    
    Preserves raw unbounded IGR for diagnostics.
    Emits warnings if baselines are physically invalid or if readings violate bounds.
    """
    warnings: List[str] = []

    if gr_shale <= gr_clean:
        raise ValueError(f"Invalid baselines: GR_shale ({gr_shale}) must be strictly greater than GR_clean ({gr_clean}).")

    denom = gr_shale - gr_clean
    igr_raw = (gr - gr_clean) / denom

    # Diagnostic checks
    below_clean = (gr < gr_clean).sum()
    above_shale = (gr > gr_shale).sum()

    if below_clean > 0:
        warnings.append(
            f"{below_clean} sample(s) had GR < GR_clean ({gr_clean} API). "
            f"May indicate ultra-clean quartz, carbonate/evaporite beds, or under-estimated clean baseline."
        )
    if above_shale > 0:
        warnings.append(
            f"{above_shale} sample(s) had GR > GR_shale ({gr_shale} API). "
            f"May indicate radioactive sand (uranium/potassium/feldspar), organic-rich source rock, or over-estimated shale baseline."
        )

    return igr_raw, warnings

def calculate_shale_volume(
    igr: pd.Series,
    model: str = "linear",
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Calculates shale volume (Vsh) from Gamma Ray Index using the selected empirical model.
    The input IGR is clipped to [0, 1] strictly for the empirical models, while preserving metadata.
    
    Returns:
        (vsh_series, metadata_dict)
    """
    model_key = model.lower().strip()
    if model_key not in VSH_MODELS:
        raise ValueError(f"Unknown Vsh model '{model}'. Supported models: {list(VSH_MODELS.keys())}")

    # Clip IGR to [0, 1] for empirical model stability
    igr_clipped = igr.clip(lower=0.0, upper=1.0)

    if model_key == "linear":
        vsh = igr_clipped
    elif model_key == "larionov_tertiary":
        vsh = 0.083 * (np.power(2.0, 3.7 * igr_clipped) - 1.0)
    elif model_key == "larionov_older":
        vsh = 0.33 * (np.power(2.0, 2.0 * igr_clipped) - 1.0)
    elif model_key == "clavier":
        # Argument of sqrt: 3.38 - (IGR + 0.7)^2
        # Max argument at IGR=0 is 3.38 - 0.49 = 2.89; at IGR=1 is 3.38 - 2.89 = 0.49
        term = 3.38 - np.square(igr_clipped + 0.7)
        term = np.clip(term, 0.0, None)
        vsh = 1.7 - np.sqrt(term)
    elif model_key == "stieber":
        denom = 3.0 - 2.0 * igr_clipped
        vsh = igr_clipped / denom
    else:
        vsh = igr_clipped

    # Guarantee final bounds [0, 1]
    vsh = vsh.clip(lower=0.0, upper=1.0)

    metadata = {
        "model_key": model_key,
        "model_name": VSH_MODELS[model_key]["name"],
        "equation": VSH_MODELS[model_key]["descr"],
        "applicability": VSH_MODELS[model_key]["applicability"],
        "mean_vsh": float(vsh.dropna().mean()) if len(vsh.dropna()) > 0 else np.nan,
    }

    return vsh, metadata
