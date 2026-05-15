import cv2

LABEL_COLORS_BGR = {
    "correct":    (34,  197,  94),
    "glue":       (234, 179,   8),
    "missalign":  (249, 115,  22),
    "missing":    (239,  68,  68),
    "r_shift":    (168,  85, 247),
    "shifted":    (239,  68,  68),
    "upsidedown": (236,  72, 153),
}

def _color(label):
    return LABEL_COLORS_BGR.get(label, (239, 68, 68))

def draw_boxes(img_bgr, detections, stage_tag):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    fs = max(0.4, min(w, h) / 900)
    lw = max(1, int(min(w, h) / 280))
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = _color(det["label"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw + 1)
        txt = f"{det['label']} {det['confidence']:.2f} [{stage_tag}]"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, txt, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lw)
    return out

def annotate_slot(slot_img, report):
    """Draw YOLO-1 and YOLO-2 boxes on the slot image."""
    out = slot_img.copy()
    if report.get("yolo_initial"):
        out = draw_boxes(out, report["yolo_initial"], "YOLO-1")
    if report.get("yolo_recheck"):
        out = draw_boxes(out, report["yolo_recheck"], "YOLO-2")
    return out
