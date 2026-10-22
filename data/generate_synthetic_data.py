"""
Synthetic Well Generator for WellScope.
Generates deterministic synthetic LAS (LAS 2.0) and CSV datasets with 5 distinct petrophysical facies:
1. Shaly formation (high GR, high NPHI, moderate RT)
2. Clean hydrocarbon gas/oil sand (low GR, high porosity with N-D crossover, high RT, low Sw - candidate pay)
3. Clean water-bearing sand (low GR, high porosity, low RT, high Sw)
4. Bad-hole washout interval (enlarged caliper > bit size, artificially low RHOB, QC flagged)
5. Tight carbonate formation (low GR, low porosity, high RT, high density)
"""

import os
import numpy as np
import pandas as pd
import lasio

def generate_synthetic_well():
    np.random.seed(42)  # Deterministic generation

    depth_start = 2000.0  # meters
    depth_stop = 2150.0   # meters
    step = 0.1            # meters
    depths = np.arange(depth_start, depth_stop + step / 2, step)
    depths = np.round(depths, 2)
    n = len(depths)

    # Initialize curve arrays
    gr = np.zeros(n)
    rhob = np.zeros(n)
    nphi = np.zeros(n)
    dt = np.zeros(n)
    rt = np.zeros(n)
    rxo = np.zeros(n)
    cali = np.zeros(n)
    sp = np.zeros(n)
    drho = np.zeros(n)

    bit_size = 8.5  # inches

    for i, d in enumerate(depths):
        # Noise
        noise = np.random.normal(0, 1)

        if d < 2025.0:
            # Facies 1: Shaly interval (2000 - 2025 m)
            gr[i] = 105.0 + 8.0 * noise
            rhob[i] = 2.58 + 0.03 * noise
            nphi[i] = 0.38 + 0.02 * noise
            dt[i] = 95.0 + 3.0 * noise
            rt[i] = 4.2 + 0.5 * noise
            rxo[i] = 3.8 + 0.4 * noise
            cali[i] = bit_size + 0.2 + 0.1 * noise
            sp[i] = -15.0 + 2.0 * noise
            drho[i] = 0.01 + 0.005 * noise

        elif d < 2060.0:
            # Facies 2: Clean Gas/Oil Sand (Candidate Pay!) (2025 - 2060 m)
            # Gas effect: NPHI decreases (0.16), RHOB decreases (2.23) -> classic crossover!
            gr[i] = 28.0 + 4.0 * noise
            rhob[i] = 2.23 + 0.02 * noise
            nphi[i] = 0.16 + 0.015 * noise
            dt[i] = 82.0 + 2.5 * noise
            rt[i] = 65.0 + 10.0 * noise
            rxo[i] = 18.0 + 3.0 * noise
            cali[i] = bit_size + 0.05 * noise
            sp[i] = -65.0 + 3.0 * noise
            drho[i] = 0.005 + 0.003 * noise

        elif d < 2080.0:
            # Transition / Silty zone (2060 - 2080 m)
            gr[i] = 68.0 + 6.0 * noise
            rhob[i] = 2.45 + 0.03 * noise
            nphi[i] = 0.27 + 0.02 * noise
            dt[i] = 88.0 + 2.0 * noise
            rt[i] = 9.5 + 1.2 * noise
            rxo[i] = 7.0 + 1.0 * noise
            cali[i] = bit_size + 0.15 + 0.05 * noise
            sp[i] = -35.0 + 2.0 * noise
            drho[i] = 0.015 + 0.005 * noise

        elif d < 2110.0:
            # Facies 3: Clean Water Sand (2080 - 2110 m)
            # High porosity, but low RT (high Sw)
            gr[i] = 25.0 + 3.5 * noise
            rhob[i] = 2.32 + 0.02 * noise
            nphi[i] = 0.22 + 0.015 * noise
            dt[i] = 84.0 + 2.0 * noise
            rt[i] = 1.6 + 0.2 * noise
            rxo[i] = 3.2 + 0.3 * noise
            cali[i] = bit_size + 0.05 * noise
            sp[i] = -70.0 + 3.0 * noise
            drho[i] = 0.008 + 0.004 * noise

        elif d < 2130.0:
            # Facies 4: Bad-hole / Washout Zone (2110 - 2130 m)
            # Caliper enlarged, pad tools lose contact: RHOB erroneously low, DRHO spikes
            cal_enlarge = 3.8 + 0.8 * abs(noise)
            cali[i] = bit_size + cal_enlarge
            gr[i] = 85.0 + 7.0 * noise
            rhob[i] = 1.82 + 0.06 * noise  # Erroneous low density
            nphi[i] = 0.44 + 0.03 * noise
            dt[i] = 110.0 + 5.0 * noise
            rt[i] = 3.5 + 0.4 * noise
            rxo[i] = 2.8 + 0.3 * noise
            sp[i] = -25.0 + 2.0 * noise
            drho[i] = 0.09 + 0.03 * abs(noise)  # High correction flag

        else:
            # Facies 5: Tight Carbonate (2130 - 2150 m)
            # High density, very low porosity, high RT
            gr[i] = 18.0 + 3.0 * noise
            rhob[i] = 2.74 + 0.02 * noise
            nphi[i] = 0.03 + 0.01 * noise
            dt[i] = 52.0 + 1.5 * noise
            rt[i] = 180.0 + 25.0 * noise
            rxo[i] = 90.0 + 15.0 * noise
            cali[i] = bit_size + 0.02 * noise
            sp[i] = -10.0 + 2.0 * noise
            drho[i] = 0.005 + 0.002 * noise

    # Clean clip bounds
    gr = np.clip(gr, 0, 250)
    rt = np.clip(rt, 0.1, 1000)
    rxo = np.clip(rxo, 0.1, 1000)
    nphi = np.clip(nphi, -0.05, 0.60)
    cali = np.clip(cali, 6.0, 16.0)

    # Add a realistic small null gap at 2040.5 - 2041.0 m to test null gap handling
    null_mask = (depths >= 2040.5) & (depths <= 2041.0)
    rhob[null_mask] = np.nan
    nphi[null_mask] = np.nan
    dt[null_mask] = np.nan

    # Create DataFrame
    df = pd.DataFrame({
        "DEPTH": np.round(depths, 2),
        "GR": np.round(gr, 2),
        "RHOB": np.round(rhob, 3),
        "NPHI": np.round(nphi, 4),
        "DT": np.round(dt, 2),
        "RT": np.round(rt, 2),
        "RXO": np.round(rxo, 2),
        "CALI": np.round(cali, 2),
        "SP": np.round(sp, 2),
        "DRHO": np.round(drho, 3),
    })

    return df

def save_synthetic_datasets(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    df = generate_synthetic_well()

    # 1. Save CSV
    csv_path = os.path.join(output_dir, "sample_synthetic_well.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic CSV to {csv_path}")

    # 2. Save LAS 2.0 via lasio
    las = lasio.LASFile()
    las.well.WELL = lasio.HeaderItem(mnemonic="WELL", value="WELLSCOPE-SYNTH-01", descr="Synthetic Test Well")
    las.well.FLD = lasio.HeaderItem(mnemonic="FLD", value="NORTH SEA SYNTHETIC", descr="Field Name")
    las.well.LOC = lasio.HeaderItem(mnemonic="LOC", value="BLOCK 16/1", descr="Location")
    las.well.SRVC = lasio.HeaderItem(mnemonic="SRVC", value="WELLSCOPE PETROPHYSICS", descr="Service Company")
    las.well.DATE = lasio.HeaderItem(mnemonic="DATE", value="2026-10-21", descr="Log Date")
    las.well.NULL = lasio.HeaderItem(mnemonic="NULL", value=-999.25, descr="Null Value")
    las.well.STRT = lasio.HeaderItem(mnemonic="STRT", unit="M", value=df["DEPTH"].iloc[0], descr="Start Depth")
    las.well.STOP = lasio.HeaderItem(mnemonic="STOP", unit="M", value=df["DEPTH"].iloc[-1], descr="Stop Depth")
    las.well.STEP = lasio.HeaderItem(mnemonic="STEP", unit="M", value=0.1, descr="Step Depth")

    # Parameters
    las.params.BS = lasio.HeaderItem(mnemonic="BS", unit="IN", value=8.5, descr="Bit Size")
    las.params.RMF = lasio.HeaderItem(mnemonic="RMF", unit="OHMM", value=0.85, descr="Mud Filtrate Resistivity")
    las.params.BHT = lasio.HeaderItem(mnemonic="BHT", unit="DEGC", value=85.0, descr="Bottom Hole Temperature")

    # Add Curves with descriptions & units
    las.append_curve("DEPTH", df["DEPTH"].values, unit="M", descr="Measured Depth")
    las.append_curve("GR", df["GR"].values, unit="GAPI", descr="Gamma Ray Log")
    las.append_curve("RHOB", df["RHOB"].fillna(-999.25).values, unit="G/C3", descr="Bulk Density")
    las.append_curve("NPHI", df["NPHI"].fillna(-999.25).values, unit="V/V", descr="Neutron Porosity (Limestone Matrix)")
    las.append_curve("DT", df["DT"].fillna(-999.25).values, unit="US/F", descr="Compressional Sonic Slowness")
    las.append_curve("RT", df["RT"].values, unit="OHMM", descr="Deep True Resistivity")
    las.append_curve("RXO", df["RXO"].values, unit="OHMM", descr="Shallow Flushed Zone Resistivity")
    las.append_curve("CALI", df["CALI"].values, unit="IN", descr="Borehole Caliper")
    las.append_curve("SP", df["SP"].values, unit="MV", descr="Spontaneous Potential")
    las.append_curve("DRHO", df["DRHO"].values, unit="G/C3", descr="Density Correction")

    las_path = os.path.join(output_dir, "sample_synthetic_well.las")
    with open(las_path, "w") as f:
        las.write(f, version=2.0)
    print(f"Saved synthetic LAS to {las_path}")

if __name__ == "__main__":
    save_synthetic_datasets()
