"""
Effective Porosity and Candidate Pay Interval Classification Module.
Implements:
1. Documented Effective Porosity (PHIE) models
2. Multi-criterion Pay Flagging (Vsh, PHIE, Sw cutoffs + QC exclusions)
3. Irregular sampling-aware gross/net thickness and Net-to-Gross (NTG) calculations
4. Discrete interval zoning and property averaging
5. Cutoff sensitivity analysis engine
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

PHIE_MODELS = {
    "proportional": {
        "name": "Proportional Model: PHIE = PHIT * (1 - Vsh)",
        "descr": "Reduces total porosity in proportion to clean sand fraction (1 - Vsh).",
        "equation": "PHIE = PHIT * (1 - Vsh)",
    },
    "density_shale_subtraction": {
        "name": "Density Shale Subtraction: PHIE = PHID - (Vsh * PHID_shale)",
        "descr": "Subtracts the pore volume associated with clay bound water in shale.",
        "equation": "PHIE = PHID - (Vsh * PHID_shale)",
    },
}

def calculate_effective_porosity(
    total_porosity: pd.Series,
    vsh: pd.Series,
    model: str = "proportional",
    phid_shale: float = 0.05,
) -> Tuple[pd.Series, Dict[str, Any], List[str]]:
    """
    Calculates Effective Porosity (PHIE) using the selected documented model.
    Guarantees bounds [0.0, 1.0].
    """
    warnings: List[str] = []
    m = model.lower().strip()

    if m not in PHIE_MODELS:
        raise ValueError(f"Unknown PHIE model '{model}'. Supported: {list(PHIE_MODELS.keys())}")

    # Clip Vsh to [0, 1] for safe calculation
    vsh_clean = vsh.clip(lower=0.0, upper=1.0)

    if m == "proportional":
        phie = total_porosity * (1.0 - vsh_clean)
    elif m == "density_shale_subtraction":
        phie = total_porosity - (vsh_clean * phid_shale)
    else:
        phie = total_porosity * (1.0 - vsh_clean)

    # Physical lower bound is 0.0 (cannot have negative effective porosity)
    phie_bounded = phie.clip(lower=0.0, upper=1.0)

    metadata = {
        "model_key": m,
        "model_name": PHIE_MODELS[m]["name"],
        "equation": PHIE_MODELS[m]["equation"],
        "phid_shale": phid_shale if m == "density_shale_subtraction" else None,
        "mean_phie": float(phie_bounded.dropna().mean()) if len(phie_bounded.dropna()) > 0 else np.nan,
    }

    return phie_bounded, metadata, warnings

def evaluate_candidate_pay(
    df: pd.DataFrame,
    vsh_col: str = "VSH",
    phie_col: str = "PHIE",
    sw_col: str = "SW",
    depth_col: str = "DEPTH",
    vsh_cutoff: float = 0.35,
    phie_cutoff: float = 0.10,
    sw_cutoff: float = 0.50,
    top_depth: Optional[float] = None,
    base_depth: Optional[float] = None,
    exclude_bad_hole: bool = True,
    qc_bad_hole_col: str = "QC_BAD_HOLE",
) -> Dict[str, Any]:
    """
    Evaluates Candidate Pay criteria across depth intervals:
    - Pass Vsh: VSH <= vsh_cutoff
    - Pass Porosity: PHIE >= phie_cutoff
    - Pass Saturation: SW <= sw_cutoff
    - Pass QC: not QC_BAD_HOLE (if exclude_bad_hole is True)
    
    Accounts for irregular depth sample spacing when integrating thickness.
    """
    # Slice to target depth window
    sub_df = df.copy()
    if top_depth is not None:
        sub_df = sub_df[sub_df[depth_col] >= top_depth]
    if base_depth is not None:
        sub_df = sub_df[sub_df[depth_col] <= base_depth]

    if sub_df.empty:
        raise ValueError("No data found within the selected depth interval.")

    # Calculate depth step delta_z for each sample
    # Using half-interval backward and forward diff for exact trapezoidal/sample coverage
    depths = sub_df[depth_col].values
    n = len(depths)

    if n > 1:
        # Step spacing per sample
        diffs = np.diff(depths)
        # Handle depth steps: for sample i, thickness is approximately mean of adjacent steps
        delta_z = np.zeros(n)
        delta_z[0] = diffs[0]
        delta_z[-1] = diffs[-1]
        delta_z[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
        # Absolute value in case depths were sorted in reverse
        delta_z = np.abs(delta_z)
    else:
        delta_z = np.array([0.1])

    sub_df["DELTA_Z"] = delta_z

    # Criteria evaluation
    # Required curves must not be NaN
    valid_data_mask = (
        sub_df[vsh_col].notna() &
        sub_df[phie_col].notna() &
        sub_df[sw_col].notna()
    )

    pass_vsh = (sub_df[vsh_col] <= vsh_cutoff) & valid_data_mask
    pass_phie = (sub_df[phie_col] >= phie_cutoff) & valid_data_mask
    pass_sw = (sub_df[sw_col] <= sw_cutoff) & valid_data_mask

    # Reservoir flag (clean sand + porosity)
    is_reservoir = pass_vsh & pass_phie

    # QC qualification
    if exclude_bad_hole and qc_bad_hole_col in sub_df.columns:
        pass_qc = ~sub_df[qc_bad_hole_col].fillna(False)
    else:
        pass_qc = pd.Series(True, index=sub_df.index)

    # Candidate Pay flag: meets all petrophysical cutoffs and passes borehole QC
    is_candidate_pay = is_reservoir & pass_sw & pass_qc

    sub_df["PASS_VSH"] = pass_vsh
    sub_df["PASS_PHIE"] = pass_phie
    sub_df["PASS_SW"] = pass_sw
    sub_df["PASS_QC"] = pass_qc
    sub_df["IS_RESERVOIR"] = is_reservoir
    sub_df["CANDIDATE_PAY"] = is_candidate_pay

    # Thickness calculations
    gross_thickness = float(sub_df["DELTA_Z"].sum())
    reservoir_thickness = float(sub_df.loc[is_reservoir, "DELTA_Z"].sum())
    net_pay_thickness = float(sub_df.loc[is_candidate_pay, "DELTA_Z"].sum())

    ntg = (net_pay_thickness / gross_thickness) if gross_thickness > 0 else 0.0

    # Summary averages in candidate pay
    pay_samples = sub_df[is_candidate_pay]
    if len(pay_samples) > 0:
        # Thickness-weighted averages
        w = pay_samples["DELTA_Z"]
        w_sum = w.sum()
        avg_vsh = float((pay_samples[vsh_col] * w).sum() / w_sum)
        avg_phie = float((pay_samples[phie_col] * w).sum() / w_sum)
        avg_sw = float((pay_samples[sw_col] * w).sum() / w_sum)
        avg_gr = float((pay_samples["GR"] * w).sum() / w_sum) if "GR" in pay_samples.columns else np.nan
        avg_rt = float((pay_samples["RT"] * w).sum() / w_sum) if "RT" in pay_samples.columns else np.nan
    else:
        avg_vsh, avg_phie, avg_sw, avg_gr, avg_rt = np.nan, np.nan, np.nan, np.nan, np.nan

    # Extract distinct contiguous pay intervals
    intervals = extract_pay_zones(sub_df, depth_col, vsh_col, phie_col, sw_col)

    # Extract candidate reservoir zones that were rejected by cutoffs or QC
    rejected_intervals = extract_rejected_zones(
        sub_df,
        depth_col=depth_col,
        vsh_col=vsh_col,
        phie_col=phie_col,
        sw_col=sw_col,
        vsh_cutoff=vsh_cutoff,
        phie_cutoff=phie_cutoff,
        sw_cutoff=sw_cutoff,
    )

    all_intervals = intervals + rejected_intervals
    all_intervals.sort(key=lambda x: x["top_depth"])

    results = {
        "gross_thickness": gross_thickness,
        "reservoir_thickness": reservoir_thickness,
        "net_pay_thickness": net_pay_thickness,
        "net_to_gross": ntg,
        "vsh_cutoff": vsh_cutoff,
        "phie_cutoff": phie_cutoff,
        "sw_cutoff": sw_cutoff,
        "top_depth": float(depths.min()),
        "base_depth": float(depths.max()),
        "avg_vsh": avg_vsh,
        "avg_phie": avg_phie,
        "avg_sw": avg_sw,
        "avg_gr": avg_gr,
        "avg_rt": avg_rt,
        "intervals": intervals,
        "rejected_intervals": rejected_intervals,
        "all_intervals": all_intervals,
        "evaluated_df": sub_df,
    }

    return results

def extract_pay_zones(
    df: pd.DataFrame,
    depth_col: str,
    vsh_col: str,
    phie_col: str,
    sw_col: str,
) -> List[Dict[str, Any]]:
    """Groups contiguous candidate pay samples into discrete reservoir pay zones."""
    is_pay = df["CANDIDATE_PAY"].values
    depths = df[depth_col].values
    dz = df["DELTA_Z"].values

    intervals = []
    in_zone = False
    start_idx = 0

    for i, flag in enumerate(is_pay):
        if flag and not in_zone:
            in_zone = True
            start_idx = i
        elif not flag and in_zone:
            in_zone = False
            zone_df = df.iloc[start_idx:i]
            z_weights = dz[start_idx:i]
            total_w = z_weights.sum()
            intervals.append({
                "status": "QUALIFIED_PAY",
                "top_depth": float(depths[start_idx]),
                "base_depth": float(depths[i - 1]),
                "net_pay_thickness": float(total_w),
                "avg_vsh": float((zone_df[vsh_col] * z_weights).sum() / total_w) if total_w > 0 else np.nan,
                "avg_phie": float((zone_df[phie_col] * z_weights).sum() / total_w) if total_w > 0 else np.nan,
                "avg_sw": float((zone_df[sw_col] * z_weights).sum() / total_w) if total_w > 0 else np.nan,
                "rejection_reason": "N/A - Qualified Candidate Pay",
            })
    if in_zone:
        zone_df = df.iloc[start_idx:]
        z_weights = dz[start_idx:]
        total_w = z_weights.sum()
        intervals.append({
            "status": "QUALIFIED_PAY",
            "top_depth": float(depths[start_idx]),
            "base_depth": float(depths[-1]),
            "net_pay_thickness": float(total_w),
            "avg_vsh": float((zone_df[vsh_col] * z_weights).sum() / total_w) if total_w > 0 else np.nan,
            "avg_phie": float((zone_df[phie_col] * z_weights).sum() / total_w) if total_w > 0 else np.nan,
            "avg_sw": float((zone_df[sw_col] * z_weights).sum() / total_w) if total_w > 0 else np.nan,
            "rejection_reason": "N/A - Qualified Candidate Pay",
        })

    return intervals

def extract_rejected_zones(
    df: pd.DataFrame,
    depth_col: str,
    vsh_col: str,
    phie_col: str,
    sw_col: str,
    vsh_cutoff: float,
    phie_cutoff: float,
    sw_cutoff: float,
    min_samples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Identifies and categorizes candidate zones that appeared to be potential reservoir rock,
    but were disqualified from net pay due to specific cutoffs or borehole QC issues.
    
    Categories:
    1. REJECTED_WATER_BEARING: Good reservoir rock (Vsh <= cut, PHIE >= cut), but failed Sw (wet sand)
    2. REJECTED_BOREHOLE_QC: Meets petrophysical cutoffs, but failed Borehole QC (washout/bad hole)
    3. REJECTED_TIGHT: Clean rock with low Sw (hydrocarbon show), but failed PHIE cutoff (tight matrix)
    4. REJECTED_SHALY_SAND: Porous rock with high resistivity, but failed Vsh cutoff (shaly sand)
    """
    # Create categorical classification for each sample
    class_series = pd.Series("OTHER_NON_PAY", index=df.index)

    # 1. Qualified Pay
    class_series[df["CANDIDATE_PAY"]] = "QUALIFIED_PAY"

    # 2. Wet Sand: Reservoir rock (clean + porous) that fails Sw
    is_wet = df["IS_RESERVOIR"] & (~df["PASS_SW"]) & df["PASS_QC"]
    class_series[is_wet] = "REJECTED_WATER_BEARING"

    # 3. Washout exclusion: Meets all petrophysical cutoffs, but fails QC
    is_washout_disqual = df["IS_RESERVOIR"] & df["PASS_SW"] & (~df["PASS_QC"])
    class_series[is_washout_disqual] = "REJECTED_BOREHOLE_QC"

    # 4. Tight Show: Clean and low Sw, but fails PHIE
    is_tight = df["PASS_VSH"] & df["PASS_SW"] & (~df["PASS_PHIE"]) & df["PASS_QC"]
    class_series[is_tight] = "REJECTED_TIGHT"

    # 5. Shaly Sand: Porous and low Sw, but exceeds Vsh
    is_shaly = (~df["PASS_VSH"]) & df["PASS_PHIE"] & df["PASS_SW"] & df["PASS_QC"]
    class_series[is_shaly] = "REJECTED_SHALY_SAND"

    depths = df[depth_col].values
    dz = df["DELTA_Z"].values
    classes = class_series.values

    rejected_intervals = []
    current_class = "OTHER_NON_PAY"
    start_idx = 0

    for i, c in enumerate(classes):
        if c.startswith("REJECTED_") and c != current_class:
            if current_class.startswith("REJECTED_"):
                # End previous rejected interval
                count = i - start_idx
                if count >= min_samples:
                    zone_df = df.iloc[start_idx:i]
                    z_w = dz[start_idx:i]
                    thk = float(z_w.sum())
                    avg_v = float((zone_df[vsh_col] * z_w).sum() / thk) if thk > 0 else np.nan
                    avg_p = float((zone_df[phie_col] * z_w).sum() / thk) if thk > 0 else np.nan
                    avg_s = float((zone_df[sw_col] * z_w).sum() / thk) if thk > 0 else np.nan
                    reason = get_rejection_reason(current_class, avg_v, avg_p, avg_s, vsh_cutoff, phie_cutoff, sw_cutoff)
                    rejected_intervals.append({
                        "status": current_class,
                        "top_depth": float(depths[start_idx]),
                        "base_depth": float(depths[i - 1]),
                        "net_pay_thickness": thk,
                        "avg_vsh": avg_v,
                        "avg_phie": avg_p,
                        "avg_sw": avg_s,
                        "rejection_reason": reason,
                    })
            current_class = c
            start_idx = i
        elif not c.startswith("REJECTED_") and current_class.startswith("REJECTED_"):
            # End previous rejected interval
            count = i - start_idx
            if count >= min_samples:
                zone_df = df.iloc[start_idx:i]
                z_w = dz[start_idx:i]
                thk = float(z_w.sum())
                avg_v = float((zone_df[vsh_col] * z_w).sum() / thk) if thk > 0 else np.nan
                avg_p = float((zone_df[phie_col] * z_w).sum() / thk) if thk > 0 else np.nan
                avg_s = float((zone_df[sw_col] * z_w).sum() / thk) if thk > 0 else np.nan
                reason = get_rejection_reason(current_class, avg_v, avg_p, avg_s, vsh_cutoff, phie_cutoff, sw_cutoff)
                rejected_intervals.append({
                    "status": current_class,
                    "top_depth": float(depths[start_idx]),
                    "base_depth": float(depths[i - 1]),
                    "net_pay_thickness": thk,
                    "avg_vsh": avg_v,
                    "avg_phie": avg_p,
                    "avg_sw": avg_s,
                    "rejection_reason": reason,
                })
            current_class = "OTHER_NON_PAY"

    if current_class.startswith("REJECTED_"):
        count = len(classes) - start_idx
        if count >= min_samples:
            zone_df = df.iloc[start_idx:]
            z_w = dz[start_idx:]
            thk = float(z_w.sum())
            avg_v = float((zone_df[vsh_col] * z_w).sum() / thk) if thk > 0 else np.nan
            avg_p = float((zone_df[phie_col] * z_w).sum() / thk) if thk > 0 else np.nan
            avg_s = float((zone_df[sw_col] * z_w).sum() / thk) if thk > 0 else np.nan
            reason = get_rejection_reason(current_class, avg_v, avg_p, avg_s, vsh_cutoff, phie_cutoff, sw_cutoff)
            rejected_intervals.append({
                "status": current_class,
                "top_depth": float(depths[start_idx]),
                "base_depth": float(depths[-1]),
                "net_pay_thickness": thk,
                "avg_vsh": avg_v,
                "avg_phie": avg_p,
                "avg_sw": avg_s,
                "rejection_reason": reason,
            })

    return rejected_intervals

def get_rejection_reason(
    status: str,
    avg_vsh: float,
    avg_phie: float,
    avg_sw: float,
    vsh_cut: float,
    phie_cut: float,
    sw_cut: float,
) -> str:
    """Formats explicit petroleum engineering justification for interval rejection."""
    if status == "REJECTED_WATER_BEARING":
        return f"Water-bearing reservoir sand: Sw avg ({avg_sw:.2f}) > cutoff ({sw_cut:.2f})"
    elif status == "REJECTED_BOREHOLE_QC":
        return "Borehole washout / bad hole flag: density pad contact loss invalidates calculation"
    elif status == "REJECTED_TIGHT":
        return f"Tight hydrocarbon show: PHIE avg ({avg_phie:.3f}) < cutoff ({phie_cut:.2f})"
    elif status == "REJECTED_SHALY_SAND":
        return f"Shaly formation: Vsh avg ({avg_vsh:.2f}) > cutoff ({vsh_cut:.2f})"
    return "Failed benchmark criteria"

def compute_cutoff_sensitivity(
    df: pd.DataFrame,
    vsh_col: str = "VSH",
    phie_col: str = "PHIE",
    sw_col: str = "SW",
    depth_col: str = "DEPTH",
    base_vsh_cut: float = 0.35,
    base_phie_cut: float = 0.10,
    base_sw_cut: float = 0.50,
) -> Dict[str, pd.DataFrame]:
    """
    Computes 1D sensitivity curves for Net Pay thickness by sweeping:
    1. Vsh cutoff across [0.15, 0.55]
    2. PHIE cutoff across [0.05, 0.25]
    3. Sw cutoff across [0.30, 0.70]
    """
    # 1. Sweep Vsh
    vsh_steps = np.linspace(0.15, 0.55, 9)
    vsh_results = []
    for cut in vsh_steps:
        eval_res = evaluate_candidate_pay(
            df, vsh_col, phie_col, sw_col, depth_col,
            vsh_cutoff=cut, phie_cutoff=base_phie_cut, sw_cutoff=base_sw_cut
        )
        vsh_results.append({
            "cutoff_value": round(cut, 3),
            "net_pay": round(eval_res["net_pay_thickness"], 2),
            "ntg": round(eval_res["net_to_gross"], 4),
        })

    # 2. Sweep PHIE
    phie_steps = np.linspace(0.05, 0.25, 9)
    phie_results = []
    for cut in phie_steps:
        eval_res = evaluate_candidate_pay(
            df, vsh_col, phie_col, sw_col, depth_col,
            vsh_cutoff=base_vsh_cut, phie_cutoff=cut, sw_cutoff=base_sw_cut
        )
        phie_results.append({
            "cutoff_value": round(cut, 3),
            "net_pay": round(eval_res["net_pay_thickness"], 2),
            "ntg": round(eval_res["net_to_gross"], 4),
        })

    # 3. Sweep Sw
    sw_steps = np.linspace(0.30, 0.70, 9)
    sw_results = []
    for cut in sw_steps:
        eval_res = evaluate_candidate_pay(
            df, vsh_col, phie_col, sw_col, depth_col,
            vsh_cutoff=base_vsh_cut, phie_cutoff=base_phie_cut, sw_cutoff=cut
        )
        sw_results.append({
            "cutoff_value": round(cut, 3),
            "net_pay": round(eval_res["net_pay_thickness"], 2),
            "ntg": round(eval_res["net_to_gross"], 4),
        })

    return {
        "vsh_sensitivity": pd.DataFrame(vsh_results),
        "phie_sensitivity": pd.DataFrame(phie_results),
        "sw_sensitivity": pd.DataFrame(sw_results),
    }
