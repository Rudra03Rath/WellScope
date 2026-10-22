# WellScope: Wireline Log Interpretation & Integrated Formation Evaluation

WellScope is a core Petroleum Engineering application developed for conventional wireline log visualization, log data quality control (QC), petrophysical interpretation, candidate pay-zone assessment, and audit-ready formation evaluation reporting.

> **Disclaimer:**
> WellScope is an educational and portfolio project for conventional wireline log visualization and formation evaluation. Calculated petrophysical properties depend on input quality, curve calibration, lithology, borehole conditions, and user-selected assumptions. Outputs are indicative and must not be used as the sole basis for operational, reserves, completion, or investment decisions.

---

## Key Petroleum Engineering Capabilities

1. **Log Data Quality Control (QC):**
   - Verification of depth monotonicity, duplicate depth identification, and irregular depth step detection.
   - Sensor sanity checks against physical sedimentary rock bounds.
   - Contiguous missing data gap detection without silent interpolation.
   - Borehole environmental quality control: Caliper washout detection ($CALI > \text{Bit Size} + \Delta$) and spine-and-ribs density correction ($|\Delta\rho| > 0.05\text{ g/cm}^3$) to prevent false porosity calculations caused by lost pad contact.
2. **Lithology & Shale Volume ($V_{sh}$):**
   - Gamma Ray Index ($IGR$) calculation with unclipped raw values preserved for diagnostics.
   - Multiple selectable shale volume models: Linear benchmark, Larionov for Tertiary (unconsolidated) rocks, Larionov for Mesozoic/Paleozoic (consolidated) rocks, Clavier et al. (1971), and Stieber (1970).
3. **Porosity Interpretation:**
   - Density Porosity ($\phi_D$) based on matrix density ($\rho_{ma}$) and fluid density ($\rho_f$).
   - Wyllie time-average Sonic Porosity ($\phi_S$) based on matrix and fluid acoustic transit times.
   - Neutron-Density combination and gas crossover diagnostic shading.
   - Documented Effective Porosity ($\phi_e$) models accounting for clay volume without double-counting.
4. **Fluid Saturation:**
   - Archie (1942) clean-sand water saturation ($S_w$) with unclipped diagnostics and shaly-sand qualification warnings.
   - Arp's temperature adjustment equation for formation water resistivity ($R_w$).
5. **Candidate Pay Evaluation & Sensitivity:**
   - Rule-based multi-criteria qualification ($V_{sh} \le V_{sh\_cut} \land \phi_e \ge \phi_{e\_cut} \land S_w \le S_{w\_cut} \land \text{Valid QC}$).
   - Integration of gross thickness, net reservoir thickness, net qualifying pay thickness, and Net-to-Gross ($NTG$) ratio with depth-spacing awareness.
   - Contiguous pay zone grouping with thickness-weighted average properties.
   - 1D cutoff sensitivity analysis curves sweeping $V_{sh}$, $\phi_e$, and $S_w$ cutoffs.
6. **Audit-Ready Deliverables:**
   - Standalone interactive HTML report with embedded multi-track log plots.
   - Professional PDF evaluation report generated via ReportLab.
   - CSV exports for depth-indexed interpreted curves and pay interval summaries.
   - JSON parameter manifest capturing exact assumptions, formulas, and cutoffs.

---

## Petrophysical Equations & Engineering Basis

### 1. Gamma-Ray Index and Shale Volume
$$IGR = \frac{GR_{log} - GR_{clean}}{GR_{shale} - GR_{clean}}$$

- **Linear Baseline:** $V_{sh} = IGR$
- **Larionov (Tertiary / Young Rocks):** $V_{sh} = 0.083 \times (2^{3.7 \times IGR} - 1)$
- **Larionov (Older / Mesozoic Rocks):** $V_{sh} = 0.33 \times (2^{2.0 \times IGR} - 1)$
- **Clavier et al. (1971):** $V_{sh} = 1.7 - \sqrt{3.38 - (IGR + 0.7)^2}$
- **Stieber (1970):** $V_{sh} = \frac{IGR}{3.0 - 2.0 \times IGR}$

*Limitations:* Gamma ray index assumes clay minerals are the sole radioactive source. Potassium feldspars, micas, or heavy uranium salts can elevate GR in clean sands without indicating shale.

### 2. Density Porosity
$$\phi_D = \frac{\rho_{ma} - \rho_b}{\rho_{ma} - \rho_f}$$

- Standard Matrix Densities ($\rho_{ma}$): Sandstone $2.65\text{ g/cm}^3$, Limestone $2.71\text{ g/cm}^3$, Dolomite $2.87\text{ g/cm}^3$, Anhydrite $2.98\text{ g/cm}^3$.
- Standard Fluid Densities ($\rho_f$): Fresh water $1.00\text{ g/cm}^3$, Saline mud/brine $1.10\text{ g/cm}^3$, Oil-based mud $0.85\text{ g/cm}^3$.
- *Diagnostics:* $\phi_D < 0$ flags heavy minerals or incorrect matrix choice; $\phi_D > 0.45$ flags gas effect or borehole washouts.

### 3. Sonic Wyllie Time-Average Porosity (Optional)
$$\phi_S = \frac{\Delta t_{log} - \Delta t_{ma}}{\Delta t_f - \Delta t_{ma}}$$

- Standard Matrix Travel Times ($\Delta t_{ma}$): Sandstone $55.5\text{ }\mu\text{s/ft}$, Limestone $47.5\text{ }\mu\text{s/ft}$, Dolomite $43.5\text{ }\mu\text{s/ft}$.
- Fluid Travel Times ($\Delta t_f$): Fresh water $189\text{ }\mu\text{s/ft}$, Brine $185\text{ }\mu\text{s/ft}$.
- *Limitations:* Overestimates porosity in unconsolidated, vuggy, fractured, or gas-bearing intervals.

### 4. Effective Porosity
- **Proportional Model:** $\phi_e = \phi_D \times (1 - V_{sh})$
- **Shale Density Subtraction:** $\phi_e = \phi_D - (V_{sh} \times \phi_{D,sh})$

### 5. Water Saturation (Archie 1942)
$$S_w = \left(\frac{a \times R_w}{\phi_e^m \times R_t}\right)^{1/n}$$

- $R_w$: Formation water resistivity at formation temperature ($\Omega\cdot\text{m}$).
- $R_t$: True formation resistivity measured by deep reading tool ($\Omega\cdot\text{m}$).
- $a$: Tortuosity factor (default $1.0$).
- $m$: Cementation exponent (default $2.0$; sandstone $\approx 2.0$, carbonates $\approx 2.0 - 2.5$).
- $n$: Saturation exponent (default $2.0$; water-wet $\approx 2.0$, oil-wet $> 2.0$).
- **Arp's Temperature Adjustment:**
  $$R_{w, T_2} = R_{w, T_1} \times \frac{T_1 + 21.5}{T_2 + 21.5} \quad (^\circ\text{C})$$
  $$R_{w, T_2} = R_{w, T_1} \times \frac{T_1 + 6.77}{T_2 + 6.77} \quad (^\circ\text{F})$$
- *Limitations:* Archie's equation assumes the rock matrix and clay are electrically non-conductive. In shaly formations ($V_{sh} > 0.20$), clay cation exchange capacity conducts electricity, leading Archie to overestimate $S_w$ and underestimate hydrocarbon pay.

### 6. Candidate Pay Qualification & Rejection Analysis
$$\text{Candidate Pay} = (V_{sh} \le V_{sh\_cut}) \land (\phi_e \ge \phi_{e\_cut}) \land (S_w \le S_{w\_cut}) \land (\neg\text{QC\_BAD\_HOLE})$$
$$\text{Net-to-Gross (NTG)} = \frac{h_{net\_pay}}{h_{gross}}$$

- **Transparent Disqualification Rationale:** In addition to qualified net pay, WellScope identifies, calculates, and reports all prospective reservoir intervals that were disqualified from pay, detailing the exact geological or mechanical cause:
  - **Water-Bearing Reservoir Sand ($S_w > S_{w\_cut}$):** Clean, porous sands that fail the saturation cutoff (defining the water leg / aquifer).
  - **Borehole Washout / QC Excluded:** Intervals meeting petrophysical cutoffs where caliper breakout or $\Delta\rho$ spikes invalidate pad tool measurements.
  - **Tight Hydrocarbon Shows ($\phi_e < \phi_{e\_cut}$):** Hydrocarbon-bearing zones lacking commercial reservoir storage capacity.
  - **Shaly Sand ($V_{sh} > V_{sh\_cut}$):** Laminated or dispersed shaly sand candidates.

---

## Repository Structure

```
WellScope/
├── app.py                             # Main Streamlit web application
├── requirements.txt                   # Dependency manifest
├── pytest.ini                         # Pytest configuration
├── README.md                          # Comprehensive documentation
├── data/                              # Test data directory
│   ├── generate_synthetic_data.py     # Deterministic synthetic LAS/CSV generator
│   ├── sample_synthetic_well.las      # 5-facies benchmark LAS 2.0 file
│   └── sample_synthetic_well.csv      # CSV export of synthetic well
├── reports/                           # Output directory for evaluation reports
│   ├── sample_evaluation_report.html
│   ├── sample_evaluation_report.pdf
│   ├── sample_derived_curves.csv
│   ├── sample_pay_intervals.csv
│   └── sample_parameter_manifest.json
├── scripts/
│   └── generate_sample_report.py      # Automated report generation script
├── src/
│   └── wellscope/
│       ├── io/                        # LAS/CSV readers and curve mapping
│       │   ├── las_reader.py
│       │   ├── csv_reader.py
│       │   └── curve_mapping.py
│       ├── core/                      # Normalization, validation, QC flags
│       │   ├── units.py
│       │   ├── validation.py
│       │   ├── qc_flags.py
│       │   └── preprocessing.py
│       ├── petrophysics/              # Core engineering calculation modules
│       │   ├── gamma_ray.py
│       │   ├── porosity.py
│       │   ├── saturation.py
│       │   └── pay_evaluation.py
│       ├── visualization/             # Plotly multi-track viewer & sensitivity
│       │   ├── log_tracks.py
│       │   └── sensitivity.py
│       ├── reporting/                 # HTML, PDF, and CSV report builders
│       │   ├── report_builder.py
│       │   └── exports.py
│       └── ui/                        # Streamlit pages and state management
│           └── pages.py
└── tests/                             # Automated pytest test suite (41 tests)
    ├── test_smoke.py
    ├── test_units.py
    ├── test_validation.py
    ├── test_las_io.py
    ├── test_log_tracks.py
    ├── test_gamma_ray.py
    ├── test_porosity.py
    ├── test_saturation.py
    └── test_pay_evaluation.py
```

---

## Windows Installation & Run Instructions

### 1. Prerequisites
- Python 3.11+ installed (tested on Python 3.14 on Windows 11).

### 2. Setup
Open PowerShell in the project directory:
```powershell
cd c:\Users\rudra\.gemini\antigravity\scratch\WellScope

# (Optional) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
Execute the comprehensive unit test suite:
```powershell
python -m pytest
```

### 4. Launch the Web Application
Launch the local Streamlit user interface:
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### 5. Generate Standalone Sample Reports
Run the headless report generator to produce sample HTML and PDF deliverables in `reports/`:
```powershell
python scripts/generate_sample_report.py
```

---

## Dataset Strategy & Testing Instructions

### 1. Bundled Synthetic Well (`sample_synthetic_well.las`)
The repository includes a deterministic synthetic well representing a 150-meter section (2000.0 m to 2150.0 m) designed to evaluate all critical petrophysical and QC edge cases:
- **Interval 1 (2000.0 - 2025.0 m) — Shaly Overburden:** High GR (~105 API), high NPHI (~0.38), moderate RT (~4.2 $\Omega\cdot\text{m}$). Correctly classified as non-reservoir.
- **Interval 2 (2025.0 - 2060.0 m) — Clean Gas/Oil Sand (Candidate Pay):** Low GR (~28 API), low RHOB (~2.23 $\text{g/cm}^3$), low NPHI (~0.16) creating a diagnostic neutron-density crossover, high RT (~65 $\Omega\cdot\text{m}$), and low $S_w \approx 0.11$. Contains an intentional null gap (2040.5 - 2041.0 m) to confirm the app does not silently interpolate across missing data.
- **Interval 3 (2060.0 - 2080.0 m) — Silty Sand Transition:** Moderate GR (~68 API), moderate porosity.
- **Interval 4 (2080.0 - 2110.0 m) — Clean Water Sand:** Low GR (~25 API), high porosity ($\phi \approx 0.20$), but low resistivity ($R_t \approx 1.6\ \Omega\cdot\text{m}$), resulting in $S_w \approx 1.0$ (100% water saturation). Correctly identified as reservoir rock but rejected from net pay.
- **Interval 5 (2110.0 - 2130.0 m) — Borehole Washout:** Caliper expands from 8.5 in to >12 in, causing density pad contact loss with artificially low RHOB and DRHO spikes. Correctly flagged and excluded by QC.
- **Interval 6 (2130.0 - 2150.0 m) — Tight Carbonate:** Low GR (~18 API), high bulk density (~2.74 $\text{g/cm}^3$), very low porosity ($\phi < 0.04$), high resistivity ($R_t \approx 180\ \Omega\cdot\text{m}$). Correctly rejected by porosity cutoff.

### 2. Testing with Open Real Field Data (Equinor Volve)
WellScope accepts standard LAS 2.0 files from public field datasets such as the **Equinor Volve Data Release**:
1. Download an open-hole wireline LAS file (e.g., well `NO 15/9-19 BT2` or `15/9-F-1 B`) from the [Equinor Volve Portal](https://www.equinor.com/energy/volve-data-sharing).
2. Ensure the log contains at least `DEPTH`, `GR`, `RHOB`, and `RT` (or equivalent deep resistivity mnemonics like `RDEP`, `AT90`, or `ILD`).
3. In WellScope, navigate to **Import & Curve Mapping**, upload the file, confirm the mnemonic mappings, and proceed with evaluation.

---
