"""
Offline PDF report generation (reportlab). No network required.
Returns PDF bytes so Streamlit can offer a download button.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)

import config
import education


def _esc(s: str) -> str:
    """Escape characters that are special to reportlab's mini-markup."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _img_flowable(rgb: np.ndarray, width_mm: float) -> Image:
    buf = io.BytesIO()
    PILImage.fromarray(rgb.astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    w = width_mm * mm
    return Image(buf, width=w, height=w)  # images are square in this app


def build_report(result: dict, quality: dict, patient: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="RetinaScan Report",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20, spaceAfter=2,
                        textColor=colors.HexColor("#0b6e4f"))
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey)
    body = styles["Normal"]
    story = []

    grade = int(result["grade"])
    grade_color = colors.HexColor(config.GRADE_COLORS[grade])
    ref = config.REFERRAL[grade]

    story.append(Paragraph("RetinaScan", h1))
    story.append(Paragraph(config.APP_TAGLINE, small))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
        f"Model: {result.get('source', 'n/a')}", small))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#0b6e4f"), spaceBefore=6, spaceAfter=8))

    # Patient
    pid = patient.get("id") or "-"
    pname = patient.get("name") or "-"
    peye = patient.get("eye") or "-"
    pnotes = patient.get("notes") or "-"
    ptable = Table(
        [["Patient ID", pid, "Name", pname],
         ["Eye", peye, "Notes", pnotes]],
        colWidths=[28 * mm, 55 * mm, 22 * mm, 55 * mm],
    )
    ptable.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ptable)
    story.append(Spacer(1, 8))

    # Result banner
    conf = result["confidence"] * 100
    banner = Table(
        [[Paragraph(f"<b>Grade {grade} &nbsp; {config.CLASS_NAMES[grade]}</b>",
                    ParagraphStyle("b", parent=body, fontSize=15, textColor=colors.white)),
          Paragraph(f"<b>{conf:.0f}%</b><br/><font size=8>confidence</font>",
                    ParagraphStyle("c", parent=body, fontSize=15, textColor=colors.white, alignment=2))]],
        colWidths=[120 * mm, 40 * mm],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), grade_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"<b>Recommendation:</b> {ref[0]} &mdash; {ref[1]} "
                           f"(urgency: {ref[2]})", body))
    story.append(Spacer(1, 8))

    # Reliability verdict — placed immediately under the grade, because it
    # governs whether the grade above should be acted on at all.
    decision = result.get("decision") or {}
    verdict = decision.get("verdict")
    if verdict:
        vcolor = {"confident": colors.HexColor("#1a9850"),
                  "borderline": colors.HexColor("#f0ad4e"),
                  "inconclusive": colors.HexColor("#d73027")}.get(
                      verdict, colors.grey)
        vlabel = {"confident": "RELIABLE READ",
                  "borderline": "PROVISIONAL - FLAG FOR HUMAN REVIEW",
                  "inconclusive": "INCONCLUSIVE - GRADE NOT RELIABLE"}.get(
                      verdict, verdict.upper())
        if result.get("escalated"):
            vlabel = "ESCALATED FOR SAFETY - FLAG FOR HUMAN REVIEW"
        vcell = ParagraphStyle("vcell", parent=body, fontSize=8.5, leading=11.5)
        inner = f"<b>Reliability: {vlabel}</b>"
        if result.get("escalated"):
            inner += (f"<br/>Reported grade {grade} was raised from the model's "
                      f"top grade {result.get('model_grade')}: the two were "
                      f"nearly tied on opposite sides of the referral threshold, "
                      f"so the more urgent grade is reported.")
        if decision.get("action"):
            inner += f"<br/>{_esc(decision['action'])}"
        for r in decision.get("reasons", [])[:5]:
            inner += f"<br/>&bull; {_esc(r)}"
        vt = Table([[Paragraph(inner, vcell)]], colWidths=[160 * mm])
        vt.setStyle(TableStyle([
            ("LINEBEFORE", (0, 0), (0, 0), 3, vcolor),
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f7f9fa")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(vt)
        story.append(Spacer(1, 8))

    # Images
    try:
        imgs = Table(
            [[_img_flowable(result["enhanced"], 72), _img_flowable(result["explanation"], 72)],
             [Paragraph("Processed fundus", small), Paragraph(result["explanation_kind"], small)]],
            colWidths=[76 * mm, 76 * mm],
        )
        imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(imgs)
        story.append(Spacer(1, 8))
    except Exception:
        pass

    # Probabilities
    prob_rows = [["Grade", "Class", "Probability"]]
    for i, p in enumerate(result["probs"]):
        prob_rows.append([str(i), config.CLASS_NAMES[i], f"{p * 100:.1f}%"])
    ptbl = Table(prob_rows, colWidths=[20 * mm, 70 * mm, 40 * mm])
    ptbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b6e4f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("BACKGROUND", (0, grade + 1), (-1, grade + 1), colors.HexColor("#eef7f2")),
    ]))
    story.append(Paragraph("<b>Class probabilities</b>", body))
    story.append(Spacer(1, 3))
    story.append(ptbl)
    story.append(Spacer(1, 8))

    # Quality
    if quality:
        q = (f"Quality score {quality.get('score', '-')}/100 &nbsp;|&nbsp; "
             f"sharpness {quality.get('sharpness', '-')} &nbsp;|&nbsp; "
             f"brightness {quality.get('brightness', '-')} &nbsp;|&nbsp; "
             f"FOV {quality.get('fov_ratio', '-')}")
        story.append(Paragraph(f"<b>Capture quality:</b> {q}", small))
        for w in quality.get("warnings", []):
            story.append(Paragraph(f"&bull; {w}", small))
        story.append(Spacer(1, 8))

    # Uncertainty audit trail
    unc = result.get("uncertainty") or {}
    if unc:
        story.append(Paragraph("<b>Uncertainty audit</b>", body))
        bits = (
            f"Top-2 margin {unc.get('margin', 0)*100:.0f} pts &nbsp;|&nbsp; "
            f"normalised entropy {unc.get('entropy_norm', 0):.2f} &nbsp;|&nbsp; "
            f"expected grade {unc.get('expected_grade', '-')} "
            f"(&plusmn;{unc.get('grade_std', 0):.2f}) &nbsp;|&nbsp; "
            f"agreement across {unc.get('tta_views', 1)} augmented views "
            f"{unc.get('tta_agreement', 0)*100:.0f}%"
        )
        if unc.get("mc_dropout_std") is not None:
            bits += (f" &nbsp;|&nbsp; MC-dropout &sigma; "
                     f"{unc['mc_dropout_std']:.3f} over "
                     f"{unc.get('mc_samples', 0)} passes")
        bits += (" &nbsp;|&nbsp; confidence is "
                 + ("temperature-calibrated (T=%.3f)" % unc.get("temperature", 1.0)
                    if unc.get("calibrated") else
                    "UNCALIBRATED (T=1) - treat as a ranking, not a likelihood"))
        story.append(Paragraph(bits, small))
        ood_info = result.get("ood") or {}
        if ood_info.get("checked"):
            story.append(Paragraph(
                f"Distribution check: fundus-likelihood "
                f"{ood_info.get('likelihood')} ({ood_info.get('level', '-')}). "
                f"{_esc(ood_info.get('reason', ''))}", small))
        story.append(Spacer(1, 8))

    # Other conditions screened from the same image
    others = result.get("multi_disease")
    if others and others.get("findings"):
        h2m = ParagraphStyle("h2m", parent=styles["Heading2"], fontSize=12,
                             spaceAfter=4, textColor=colors.HexColor("#0b6e4f"))
        cellm = ParagraphStyle("cellm", parent=body, fontSize=8.5, leading=11.5)
        story.append(HRFlowable(width="100%", color=colors.lightgrey,
                                spaceBefore=4, spaceAfter=6))
        story.append(Paragraph("Other conditions screened from this image", h2m))

        status_color = {
            "refer": colors.HexColor("#d73027"),
            "borderline": colors.HexColor("#f0ad4e"),
            "normal": colors.HexColor("#1a9850"),
            "unavailable": colors.HexColor("#9aa4ae"),
        }
        hdr = ParagraphStyle("hdr", parent=cellm, textColor=colors.white)
        rows = [[Paragraph("<b>Condition</b>", hdr),
                 Paragraph("<b>Status</b>", hdr),
                 Paragraph("<b>Finding</b>", hdr)]]
        style_cmds = [
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b6e4f")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for i, f in enumerate(others["findings"], start=1):
            metrics_txt = ", ".join(f"{k}: {v}"
                                    for k, v in (f.get("metrics") or {}).items())
            detail = _esc(f.get("headline", "")) + "<br/>" + _esc(f.get("detail", ""))
            if metrics_txt:
                detail += f"<br/><font size=7 color='#7a828a'>{_esc(metrics_txt)}</font>"
            rows.append([
                Paragraph(_esc(f.get("condition", "-")), cellm),
                Paragraph(f.get("status", "-").upper(), cellm),
                Paragraph(detail, cellm),
            ])
            style_cmds.append(
                ("TEXTCOLOR", (1, i), (1, i),
                 status_color.get(f.get("status"), colors.grey)))
        mt = Table(rows, colWidths=[34 * mm, 20 * mm, 106 * mm])
        mt.setStyle(TableStyle(style_cmds))
        story.append(mt)
        story.append(Spacer(1, 5))

        try:
            story.append(_img_flowable(others["overlay"], 62))
            story.append(Paragraph(_esc(others.get("overlay_legend", "")), small))
            story.append(Spacer(1, 4))
        except Exception:
            pass
        story.append(Paragraph(_esc(others.get("disclaimer", "")), small))
        story.append(Spacer(1, 8))

    # Understanding your result — disease explanation, treatment, diet
    edu = education.get_education(grade)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        spaceAfter=4, textColor=colors.HexColor("#0b6e4f"))
    bullet = ParagraphStyle("bullet", parent=body, fontSize=9, leftIndent=10,
                            bulletIndent=0, spaceAfter=2, leading=12)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=12)

    story.append(HRFlowable(width="100%", color=colors.lightgrey, spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(
        f"Understanding your result &mdash; Grade {grade} &#183; {_esc(edu['stage'])}", h2))

    story.append(Paragraph("<b>What it means</b>", body))
    story.append(Paragraph(_esc(edu["what_it_is"]), body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Treatment &amp; management</b>", body))
    for t in edu["treatment"]:
        story.append(Paragraph(_esc(t), bullet, bulletText="•"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Diet guidance</b>", body))
    story.append(Paragraph(_esc(edu["diet_rationale"]), small))
    story.append(Spacer(1, 4))
    avoid_html = "<b>Foods to avoid</b><br/>" + "<br/>".join(
        f"&bull; {_esc(x)}" for x in edu["foods_to_avoid"])
    favor_html = "<b>Eye-friendly foods</b><br/>" + "<br/>".join(
        f"&bull; {_esc(x)}" for x in edu["foods_to_favor"])
    diet_tbl = Table(
        [[Paragraph(avoid_html, cell), Paragraph(favor_html, cell)]],
        colWidths=[84 * mm, 84 * mm],
    )
    diet_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#fdecea")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#e7f5ee")),
        ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#f5c6c0")),
        ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#b7e0cd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(diet_tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(_esc(edu["disclaimer"]), small))

    story.append(HRFlowable(width="100%", color=colors.lightgrey, spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(config.DISCLAIMER, small))

    doc.build(story)
    buf.seek(0)
    return buf.read()
