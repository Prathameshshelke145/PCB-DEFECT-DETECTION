import cv2
import tempfile

from src.models.inference import run_yolo_detection, run_cnn_on_crops_filelist
from src.models.crop_model import extract_slots
from src.data.preprocessing import crop_pcb_for_cnn, save_crops_to_files, save_slot_to_tempfile
from src.utils.helpers import yolo_has_defect, cnn_has_defect


def analyze_pcb(image_path, initial_conf=0.25, recheck_conf=0.15):
    """
    Run the full 3-stage inspection pipeline on a single PCB image.
    Returns a report dict.
    """
    report = {
        "status":       None,
        "stage":        None,
        "yolo_initial": [],
        "cnn":          {},
        "yolo_recheck": [],
    }

    # STEP 1: YOLO initial
    dets1 = run_yolo_detection(image_path, conf=initial_conf)
    report["yolo_initial"] = dets1

    if yolo_has_defect(dets1):
        report["status"] = "DEFECTIVE"
        report["stage"]  = "YOLO initial"
        return report

    # STEP 2: CNN on crops
    img  = cv2.imread(image_path)
    tmp  = tempfile.mkdtemp()
    crops      = crop_pcb_for_cnn(img)
    crop_files = save_crops_to_files(crops, prefix=f"{tmp}/pcb")

    cnn_results = run_cnn_on_crops_filelist(crop_files)
    report["cnn"] = cnn_results

    if not cnn_has_defect(cnn_results):
        report["status"] = "OK"
        report["stage"]  = "CNN"
        return report

    # STEP 3: YOLO recheck at lower threshold
    dets2 = run_yolo_detection(image_path, conf=recheck_conf)
    report["yolo_recheck"] = dets2

    if yolo_has_defect(dets2):
        report["status"] = "DEFECTIVE"
        report["stage"]  = "YOLO recheck"
        return report

    report["status"] = "OK_WITH_WARNING"
    report["stage"]  = "Final"
    return report


def analyze_panel(image_path, initial_conf=0.25, recheck_conf=0.15):
    """
    Entry point for a multi-PCB panel image.
    1. Uses crop model to extract individual PCB slots.
    2. Runs the full 3-stage pipeline on each slot.
    Returns a list of per-slot result dicts.
    """
    slots = extract_slots(image_path)

    if not slots:
        return []

    tmp = tempfile.mkdtemp()
    panel_results = []

    for slot in slots:
        slot_path = save_slot_to_tempfile(slot["crop"], slot["index"], tmp_dir=tmp)

        report = analyze_pcb(slot_path, initial_conf=initial_conf, recheck_conf=recheck_conf)

        panel_results.append({
            "slot_index":  slot["index"],
            "slot_label":  slot["label"],
            "confidence":  slot["confidence"],
            "image_path":  slot_path,
            "report":      report,
        })

    return panel_results
