"""
Petrophysical Report Generation Module.
Produces comprehensive HTML and PDF evaluation reports with well metadata,
equations, parameter values, QC findings, candidate intervals, and engineering caveats.
"""

from typing import Dict, Any, List, Optional
import io
import plotly.graph_objects as go
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

def generate_html_report(
    header: Dict[str, Any],
    parameters: Dict[str, Any],
    qc_summary: Dict[str, Any],
    pay_results: Dict[str, Any],
    fig: Optional[go.Figure] = None,
    software_version: str = "1.0.0",
) -> str:
    """Generates an audit-ready, standalone HTML petrophysical evaluation report."""
    plot_html = ""
    if fig is not None:
        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    intervals_html = ""
    intervals = pay_results.get("intervals", [])
    if intervals:
        intervals_html += """
        <table class="report-table">
            <thead>
                <tr>
                    <th>Zone</th>
                    <th>Top Depth (m)</th>
                    <th>Base Depth (m)</th>
                    <th>Net Pay (m)</th>
                    <th>Avg Vsh</th>
                    <th>Avg PHIE</th>
                    <th>Avg Sw</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        for i, zone in enumerate(intervals, 1):
            intervals_html += f"""
                <tr>
                    <td>Zone {i}</td>
                    <td>{zone['top_depth']:.2f}</td>
                    <td>{zone['base_depth']:.2f}</td>
                    <td>{zone['net_pay_thickness']:.2f}</td>
                    <td>{zone['avg_vsh']:.3f}</td>
                    <td>{zone['avg_phie']:.3f}</td>
                    <td>{zone['avg_sw']:.3f}</td>
                    <td><span style="color: #28a745; font-weight: bold;">Candidate Pay</span></td>
                </tr>
            """
        intervals_html += "</tbody></table>"
    else:
        intervals_html = "<p><em>No intervals satisfied the candidate pay cutoffs in the evaluated section.</em></p>"

    # Rejected intervals HTML
    rejected_html = ""
    rejected = pay_results.get("rejected_intervals", [])
    if rejected:
        rejected_html += """
        <div class="section-title">Disqualified / Rejected Candidate Reservoir Intervals</div>
        <p style="font-size: 13px; color: #555;">
            The following prospective reservoir intervals exhibited porous or permeable characteristics, 
            but were rejected from net pay based on the user's cutoff thresholds or borehole QC exclusions:
        </p>
        <table class="report-table">
            <thead>
                <tr>
                    <th>Zone</th>
                    <th>Top (m)</th>
                    <th>Base (m)</th>
                    <th>Gross Thk (m)</th>
                    <th>Avg Vsh</th>
                    <th>Avg PHIE</th>
                    <th>Avg Sw</th>
                    <th>Disqualification / Rejection Rationale</th>
                </tr>
            </thead>
            <tbody>
        """
        for j, r_zone in enumerate(rejected, 1):
            status_color = "#dc3545" if "QC" in r_zone.get("status", "") else "#fd7e14"
            rejected_html += f"""
                <tr>
                    <td>Excl {j}</td>
                    <td>{r_zone['top_depth']:.2f}</td>
                    <td>{r_zone['base_depth']:.2f}</td>
                    <td>{r_zone['net_pay_thickness']:.2f}</td>
                    <td>{r_zone['avg_vsh']:.3f}</td>
                    <td>{r_zone['avg_phie']:.3f}</td>
                    <td>{r_zone['avg_sw']:.3f}</td>
                    <td><strong style="color: {status_color};">{r_zone.get('status', 'REJECTED')}</strong>: {r_zone.get('rejection_reason', 'Failed benchmark')}</td>
                </tr>
            """
        rejected_html += "</tbody></table>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WellScope Petrophysical Evaluation Report - {header.get('well_name', 'Well')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #222;
            line-height: 1.5;
            margin: 0;
            padding: 30px;
            background-color: #fafafa;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .header-title {{
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 12px;
            margin-bottom: 25px;
        }}
        .header-title h1 {{
            margin: 0;
            color: #1f77b4;
            font-size: 26px;
        }}
        .header-title .subtitle {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
        .section-title {{
            font-size: 18px;
            color: #333;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 6px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 13px;
        }}
        .report-table th, .report-table td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        .report-table th {{
            background-color: #f2f7fb;
            color: #333;
            font-weight: 600;
        }}
        .metric-cards {{
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
        }}
        .metric-card {{
            flex: 1;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 22px;
            font-weight: 700;
            color: #1f77b4;
            margin-top: 5px;
        }}
        .metric-card .label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
        }}
        .disclaimer {{
            margin-top: 40px;
            padding: 15px;
            background-color: #fff8e1;
            border-left: 4px solid #ffc107;
            font-size: 12px;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-title">
            <h1>WELLSCOPE PETROPHYSICAL FORMATION EVALUATION REPORT</h1>
            <div class="subtitle">Well: <strong>{header.get('well_name', 'Unknown')}</strong> | Field: {header.get('field', 'N/A')} | Date: {header.get('date', 'N/A')} | WellScope v{software_version}</div>
        </div>

        <div class="section-title">Executive Evaluation Summary</div>
        <div class="metric-cards">
            <div class="metric-card">
                <div class="label">Gross Interval</div>
                <div class="value">{pay_results.get('gross_thickness', 0.0):.1f} m</div>
            </div>
            <div class="metric-card">
                <div class="label">Reservoir Rock</div>
                <div class="value">{pay_results.get('reservoir_thickness', 0.0):.1f} m</div>
            </div>
            <div class="metric-card">
                <div class="label">Net Pay Thickness</div>
                <div class="value">{pay_results.get('net_pay_thickness', 0.0):.1f} m</div>
            </div>
            <div class="metric-card">
                <div class="label">Net-To-Gross (NTG)</div>
                <div class="value">{pay_results.get('net_to_gross', 0.0):.1%}</div>
            </div>
        </div>

        <div class="section-title">Well & Ingestion Metadata</div>
        <table class="report-table">
            <tr>
                <th>Well Name</th><td>{header.get('well_name', 'N/A')}</td>
                <th>Field / Location</th><td>{header.get('field', 'N/A')} / {header.get('location', 'N/A')}</td>
            </tr>
            <tr>
                <th>Start Depth</th><td>{header.get('start_depth', 'N/A')} {header.get('depth_unit', 'M')}</td>
                <th>Stop Depth</th><td>{header.get('stop_depth', 'N/A')} {header.get('depth_unit', 'M')}</td>
            </tr>
            <tr>
                <th>Service Company</th><td>{header.get('service_company', 'N/A')}</td>
                <th>Sampling Step</th><td>{header.get('step', 'N/A')} {header.get('depth_unit', 'M')}</td>
            </tr>
        </table>

        <div class="section-title">Petrophysical Parameters & Models</div>
        <table class="report-table">
            <thead>
                <tr>
                    <th>Domain</th>
                    <th>Selected Model</th>
                    <th>Key Input Parameters</th>
                    <th>Equation / Basis</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Shale Volume (Vsh)</strong></td>
                    <td>{parameters.get('vsh_model_name', 'Linear')}</td>
                    <td>GR_clean = {parameters.get('gr_clean', 25.0)} API, GR_shale = {parameters.get('gr_shale', 115.0)} API</td>
                    <td><code>{parameters.get('vsh_equation', 'IGR = (GR-GR_clean)/(GR_shale-GR_clean)')}</code></td>
                </tr>
                <tr>
                    <td><strong>Porosity (PHID/PHIE)</strong></td>
                    <td>{parameters.get('phie_model_name', 'Proportional')}</td>
                    <td>RHOMA = {parameters.get('rhoma', 2.65)} g/cm³, RHOF = {parameters.get('rhof', 1.00)} g/cm³</td>
                    <td><code>PHID = (RHOMA - RHOB)/(RHOMA - RHOF); PHIE = PHID*(1-Vsh)</code></td>
                </tr>
                <tr>
                    <td><strong>Water Saturation (Sw)</strong></td>
                    <td>Archie (1942)</td>
                    <td>Rw = {parameters.get('rw', 0.05)} ohm.m, a = {parameters.get('a', 1.0)}, m = {parameters.get('m', 2.0)}, n = {parameters.get('n', 2.0)}</td>
                    <td><code>Sw = ((a * Rw) / (PHIE^m * Rt))^(1/n)</code></td>
                </tr>
                <tr>
                    <td><strong>Pay Cutoffs</strong></td>
                    <td>Rule-based deterministic</td>
                    <td>Vsh &le; {pay_results.get('vsh_cutoff', 0.35)}, PHIE &ge; {pay_results.get('phie_cutoff', 0.10)}, Sw &le; {pay_results.get('sw_cutoff', 0.50)}</td>
                    <td>Pass Vsh AND Pass PHIE AND Pass Sw AND Pass QC</td>
                </tr>
            </tbody>
        </table>

        <div class="section-title">Quality Control & Borehole Findings</div>
        <table class="report-table">
            <tr>
                <th>Bit Size</th><td>{qc_summary.get('bit_size_in', 8.5)} in</td>
                <th>Washout Threshold</th><td>{qc_summary.get('washout_threshold_in', 10.0)} in</td>
            </tr>
            <tr>
                <th>Washout Samples</th><td>{qc_summary.get('washout_sample_count', 0)} samples</td>
                <th>Bad Hole Exclusions</th><td>{qc_summary.get('bad_hole_sample_count', 0)} samples</td>
            </tr>
        </table>

        <div class="section-title">Candidate Pay Intervals</div>
        {intervals_html}

        {rejected_html}

        <div class="section-title">Multi-Track Petrophysical Composite Log</div>
        <div style="margin-top: 20px;">
            {plot_html}
        </div>

        <div class="disclaimer">
            <strong>Disclaimer:</strong> WellScope is an educational and portfolio project for conventional wireline log visualization and formation evaluation. Calculated petrophysical properties depend on input quality, curve calibration, lithology, borehole conditions, and user-selected assumptions. Outputs are indicative and must not be used as the sole basis for operational, reserves, completion, or investment decisions.
        </div>
    </div>
</body>
</html>
"""
    return html_content

def generate_pdf_report(
    header: Dict[str, Any],
    parameters: Dict[str, Any],
    qc_summary: Dict[str, Any],
    pay_results: Dict[str, Any],
    software_version: str = "1.0.0",
) -> bytes:
    """Generates an executive PDF petrophysical report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#222222"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#856404"),
    )

    story = []

    # Title
    story.append(Paragraph("WELLSCOPE PETROPHYSICAL FORMATION EVALUATION", title_style))
    story.append(
        Paragraph(
            f"Well: <b>{header.get('well_name', 'Unknown')}</b> | Field: {header.get('field', 'N/A')} | "
            f"Date: {header.get('date', 'N/A')} | Software Version: {software_version}",
            subtitle_style,
        )
    )

    # Key Metrics Table
    story.append(Paragraph("Executive Summary & Net Pay", h2_style))
    metrics_data = [
        ["Gross Interval (m)", "Reservoir Rock (m)", "Net Pay Thickness (m)", "Net-to-Gross (NTG)"],
        [
            f"{pay_results.get('gross_thickness', 0.0):.2f}",
            f"{pay_results.get('reservoir_thickness', 0.0):.2f}",
            f"{pay_results.get('net_pay_thickness', 0.0):.2f}",
            f"{pay_results.get('net_to_gross', 0.0):.1%}",
        ],
    ]
    t_metrics = Table(metrics_data, colWidths=[130, 130, 140, 140])
    t_metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # Parameters Table
    story.append(Paragraph("Interpretation Parameters & Assumptions", h2_style))
    params_data = [
        ["Domain", "Model", "Parameters", "Cutoff / Value"],
        ["Shale Volume (Vsh)", parameters.get("vsh_model_name", "Linear"), f"GR_cl={parameters.get('gr_clean')}, GR_sh={parameters.get('gr_shale')}", f"Vsh <= {pay_results.get('vsh_cutoff', 0.35)}"],
        ["Porosity", "Density Porosity", f"RHOMA={parameters.get('rhoma')}, RHOF={parameters.get('rhof')}", f"PHIE >= {pay_results.get('phie_cutoff', 0.10)}"],
        ["Saturation", "Archie (1942)", f"Rw={parameters.get('rw')}, a={parameters.get('a')}, m={parameters.get('m')}, n={parameters.get('n')}", f"Sw <= {pay_results.get('sw_cutoff', 0.50)}"],
    ]
    t_params = Table(params_data, colWidths=[100, 120, 200, 120])
    t_params.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 10))

    # Candidate Pay Intervals Table
    story.append(Paragraph("Candidate Pay Intervals", h2_style))
    intervals = pay_results.get("intervals", [])
    if intervals:
        int_data = [["Zone", "Top (m)", "Base (m)", "Net Pay (m)", "Avg Vsh", "Avg PHIE", "Avg Sw"]]
        for i, z in enumerate(intervals, 1):
            int_data.append([
                f"Zone {i}",
                f"{z['top_depth']:.2f}",
                f"{z['base_depth']:.2f}",
                f"{z['net_pay_thickness']:.2f}",
                f"{z['avg_vsh']:.3f}",
                f"{z['avg_phie']:.3f}",
                f"{z['avg_sw']:.3f}",
            ])
        t_int = Table(int_data, colWidths=[60, 80, 80, 80, 80, 80, 80])
        t_int.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_int)
    else:
        story.append(Paragraph("No intervals met candidate pay cutoffs in evaluated section.", body_style))

    # Rejected / Disqualified prospective zones
    rejected = pay_results.get("rejected_intervals", [])
    if rejected:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Disqualified / Rejected Prospective Intervals", h2_style))
        rej_data = [["Zone", "Top (m)", "Base (m)", "Thk (m)", "Avg Vsh", "Avg PHIE", "Avg Sw", "Rejection Rationale"]]
        for j, r in enumerate(rejected, 1):
            rej_data.append([
                f"Excl {j}",
                f"{r['top_depth']:.2f}",
                f"{r['base_depth']:.2f}",
                f"{r['net_pay_thickness']:.2f}",
                f"{r['avg_vsh']:.3f}",
                f"{r['avg_phie']:.3f}",
                f"{r['avg_sw']:.3f}",
                Paragraph(r.get("rejection_reason", "Failed cutoff"), body_style),
            ])
        t_rej = Table(rej_data, colWidths=[40, 50, 50, 45, 45, 50, 50, 210])
        t_rej.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fff3cd")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-2, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_rej)

    story.append(Spacer(1, 15))

    # QC Findings
    story.append(Paragraph("Quality Control & Borehole Diagnostics", h2_style))
    story.append(
        Paragraph(
            f"Bit Size: {qc_summary.get('bit_size_in', 8.5)} in | "
            f"Washout Threshold: {qc_summary.get('washout_threshold_in', 10.0)} in | "
            f"Washout Samples: {qc_summary.get('washout_sample_count', 0)} | "
            f"Bad Hole Exclusions: {qc_summary.get('bad_hole_sample_count', 0)}",
            body_style,
        )
    )

    story.append(Spacer(1, 20))

    # Disclaimer Box
    disclaimer_text = (
        "<b>Disclaimer:</b> WellScope is an educational and portfolio project for conventional wireline log "
        "visualization and formation evaluation. Calculated petrophysical properties depend on input quality, "
        "curve calibration, lithology, borehole conditions, and user-selected assumptions. Outputs are indicative "
        "and must not be used as the sole basis for operational, reserves, completion, or investment decisions."
    )
    t_disc = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[540])
    t_disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e1")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ffc107")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_disc)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
