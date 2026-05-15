CNN_CLASS_NAMES = [
    'Corrected_BR','Corrected_IC2','Corrected_R7','Corrected_iC1',
    'defected_BR!','miss_align_ic1','miss_aligned_ic2',
    'missing_R7','shifted_R7','white_R7'
]

YOLO_MOUNTED_CLASS_NAMES = [
    'correct','glue','missalign','missing',
    'r_shift','shifted','upsidedown'
]

CNN_MODEL_PATH          = "models/cnn_model.keras"
YOLO_MOUNTED_MODEL_PATH = "models/yolo_model.pt"
CROP_MODEL_PATH         = "models/crop_model.pt"

# conf=0.5 matches original Colab code exactly — do not change
CROP_CONF_THRESHOLD     = 0.5
