import cv2
import os
import tempfile


def crop_pcb_for_cnn(image_bgr):
    h, w, _ = image_bgr.shape
    vs, hs = w // 2, h // 2

    crops = {
        "top_left":     image_bgr[0:hs, 0:vs],
        "top_right":    image_bgr[0:hs, vs:w],
        "bottom_left":  image_bgr[hs:h, 0:vs],
        "bottom_right": image_bgr[hs:h, vs:w],
    }

    ch = hs // 4
    cy = int(0.8 * hs)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch, h)
    crops["center"] = image_bgr[y1:y2, 0:w]

    return crops


def save_crops_to_files(crops, prefix="data/pcb"):
    order = ["bottom_right", "bottom_left", "center", "top_left", "top_right"]
    paths = []
    for name in order:
        fname = f"{prefix}_{name}.jpg"
        cv2.imwrite(fname, crops[name])
        paths.append(fname)
    return paths


def save_slot_to_tempfile(slot_bgr, slot_index, tmp_dir=None):
    """Save a single slot crop to a temp file and return the path."""
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, f"slot_{slot_index:02d}.jpg")
    cv2.imwrite(path, slot_bgr)
    return path
