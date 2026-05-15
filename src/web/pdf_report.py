import io
import os
import base64
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image as RLImage, HRFlowable
)


STATUS_COLOR = {
    "OK":              colors.HexColor("#22c55e"),
    "DEFECTIVE":       colors.HexColor("#ef4444"),
    "OK_WITH_WARNING": colors.HexColor("#f59e0b"),
}


def build_pdf(slots_ctx):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("t", parent=styles["Title"], fontSize=16, spaceAfter=4)
    sub_s   = ParagraphStyle("s", parent=styles["Normal"], fontSize=9,
                              textColor=colors.grey, spaceAfter=10)
    h2_s    = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceBefore=8)
    normal  = styles["Normal"]

    story = []
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("PCB Inspection Report", title_s))
    story.append(Paragraph(f"Generated: {ts}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 5*mm))

    # ── Summary table ──
    ok_n  = sum(1 for s in slots_ctx if s["report"]["status"] == "OK")
    df_n  = sum(1 for s in slots_ctx if s["report"]["status"] == "DEFECTIVE")
    wn_n  = sum(1 for s in slots_ctx if s["report"]["status"] == "OK_WITH_WARNING")

    sum_data = [
        ["Total Slots", "OK", "Defective", "Warning"],
        [str(len(slots_ctx)), str(ok_n), str(df_n), str(wn_n)],
    ]
    sum_tbl = Table(sum_data, colWidths=[42*mm]*4)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 8*mm))

    # ── Per-slot detail ──
    for s in slots_ctx:
        report = s["report"]
        status = report["status"]
        sc     = STATUS_COLOR.get(status, colors.grey)

        story.append(Paragraph(f"Slot {s['slot_index']} — {status}", h2_s))

        # annotated image (use annotated_b64 if available, else plain b64)
        img_b64 = s.get("annotated_b64") or s.get("b64", "")
        if img_b64:
            img_buf = io.BytesIO(base64.b64decode(img_b64))
            rl_img  = RLImage(img_buf, width=50*mm, height=65*mm)
            story.append(rl_img)

        # detail rows
        rows = [["Field", "Value"]]
        rows.append(["Status",            status])
        rows.append(["Stage stopped at",  report.get("stage", "—")])
        rows.append(["Crop label",        s["slot_label"]])
        rows.append(["Crop confidence",   f'{s["confidence"]:.2f}'])
        rows.append(["YOLO-1 detections", str(len(report.get("yolo_initial", [])))])
        rows.append(["YOLO-2 detections", str(len(report.get("yolo_recheck", [])))])

        for det in report.get("yolo_initial", []):
            bbox = [int(v) for v in det["bbox_xyxy"]]
            rows.append([f"  YOLO-1 · {det['label']}",
                         f"conf {det['confidence']:.2f}  bbox {bbox}"])

        for path, val in report.get("cnn", {}).items():
            region = os.path.basename(path).replace(".jpg", "")
            rows.append([f"  CNN · {region}",
                         f"{val['class']}  conf {val['confidence']:.2f}"])

        for det in report.get("yolo_recheck", []):
            bbox = [int(v) for v in det["bbox_xyxy"]]
            rows.append([f"  YOLO-2 · {det['label']}",
                         f"conf {det['confidence']:.2f}  bbox {bbox}"])

        tbl = Table(rows, colWidths=[75*mm, 95*mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 5*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 4*mm))

    doc.build(story)
    buf.seek(0)
    return buf
