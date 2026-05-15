from src.config.config import CNN_CLASS_NAMES, YOLO_MOUNTED_CLASS_NAMES

NON_DEFECT_CNN_LABELS = [
    "Corrected_BR","Corrected_IC2","Corrected_R7","Corrected_iC1"
]

DEFECT_LABELS_CNN = [x for x in CNN_CLASS_NAMES if x not in NON_DEFECT_CNN_LABELS]
DEFECT_LABELS_YOLO = [x for x in YOLO_MOUNTED_CLASS_NAMES if x != "correct"]


def yolo_has_defect(detections):
    return any(d["label"] in DEFECT_LABELS_YOLO for d in detections)


def cnn_has_defect(results):
    return any(v["class"] in DEFECT_LABELS_CNN for v in results.values())



