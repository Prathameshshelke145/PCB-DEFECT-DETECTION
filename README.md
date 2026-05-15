# PCB Inspector

An automated PCB (Printed Circuit Board) defect detection system that analyzes a panel image containing 10 PCB slots, extracts each slot individually, and runs a 3-stage AI inspection pipeline on every one.

---

## How It Works

### Stage 1 — Slot Extraction (Crop Model)
A YOLO-based crop model scans the full panel image and detects each individual PCB slot. Each detected region is cropped and auto-rotated to portrait orientation if needed.

### Stage 2 — 3-Stage Inspection Pipeline (per slot)

Each extracted slot goes through three sequential stages:

```
Panel Image
    │
    ▼
[Crop Model]  →  10 individual PCB images
    │
    ▼
[YOLO-1]  ──── defect found? ──→  DEFECTIVE  (stops here)
    │ no defect
    ▼
[CNN]  ──── all OK? ──→  OK  (stops here)
    │ CNN flags something
    ▼
[YOLO-2 recheck]  ──── defect confirmed? ──→  DEFECTIVE
    │ not confirmed
    ▼
OK_WITH_WARNING
```

| Stage | Model | Purpose |
|---|---|---|
| YOLO-1 | `yolo_model.pt` | Fast initial defect scan at conf=0.25 |
| CNN | `cnn_model.keras` | Deep crop-level classification |
| YOLO-2 | `yolo_model.pt` | Low-threshold recheck at conf=0.15 |

### Possible Outcomes

| Status | Meaning |
|---|---|
| `OK` | No defects found by YOLO-1 or CNN |
| `DEFECTIVE` | Defect confirmed by YOLO-1 or YOLO-2 |
| `OK_WITH_WARNING` | CNN flagged something but YOLO-2 couldn't confirm |

---

## Defect Classes

### CNN Classes (10)
| Label | Type |
|---|---|
| `Corrected_BR`, `Corrected_IC2`, `Corrected_R7`, `Corrected_iC1` | Non-defective |
| `defected_BR!` | Defective bridge |
| `miss_align_ic1`, `miss_aligned_ic2` | IC misalignment |
| `missing_R7` | Missing resistor |
| `shifted_R7`, `white_R7` | Shifted / discoloured resistor |

### YOLO Classes (7)
| Label | Type |
|---|---|
| `correct` | Non-defective |
| `glue` | Glue contamination |
| `missalign` | Component misalignment |
| `missing` | Missing component |
| `r_shift`, `shifted` | Component shift |
| `upsidedown` | Component flipped |

---

## Project Structure

```
├── data/                        # Sample PCB crop images
│   ├── pcb_bottom_left.jpg
│   ├── pcb_bottom_right.jpg
│   ├── pcb_center.jpg
│   ├── pcb_top_left.jpg
│   └── pcb_top_right.jpg
│
├── models/                      # Trained model files (not tracked in git)
│   ├── cnn_model.keras          # Keras CNN classifier
│   ├── crop_model.pt            # YOLO slot extraction model
│   └── yolo_model.pt            # YOLO defect detection model
│
├── src/
│   ├── config/
│   │   └── config.py            # Model paths, class names, thresholds
│   │
│   ├── data/
│   │   └── preprocessing.py     # Image cropping and slot saving utilities
│   │
│   ├── models/
│   │   ├── cnn_model.py         # CNN model loader + augmentation pipeline
│   │   ├── crop_model.py        # Crop YOLO model loader + slot extractor
│   │   ├── inference.py         # CNN and YOLO inference functions
│   │   └── yolo_model.py        # Mounted YOLO model loader
│   │
│   ├── pipeline/
│   │   └── analyzer.py          # analyze_pcb() and analyze_panel() pipeline
│   │
│   ├── utils/
│   │   └── helpers.py           # Defect label helpers
│   │
│   ├── web/
│   │   ├── annotate.py          # Bounding box drawing on slot images
│   │   ├── app.py               # Flask web app + REST API
│   │   └── pdf_report.py        # PDF report generation (reportlab)
│   │
│   ├── main.py                  # CLI entry point
│   └── requirements.txt         # Python dependencies
│
└── README.md
```

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r src/requirements.txt
```

### 3. Add model files

Place the following files in the `models/` directory:

```
models/cnn_model.keras
models/crop_model.pt
models/yolo_model.pt
```

> These are not tracked in git. Obtain them from your team or training pipeline.

---

## Running

### Web App

```bash
python -m src.web.app
```

Then open [http://localhost:5000](http://localhost:5000)

### CLI

```bash
python -m src.main
```

Runs the full panel pipeline on `data/test.jpg` and prints JSON results.

---

## Web Interface

### Upload & Analyze
- Upload a panel image containing 10 PCB slots
- Adjust YOLO-1 and YOLO-2 confidence thresholds via sliders (defaults: 0.25 / 0.15)
- Click **Analyze Panel**

### Results Grid
- Each slot shown as a card with a color-coded status badge
  - 🟢 Green — OK
  - 🔴 Red — DEFECTIVE
  - 🟡 Amber — OK_WITH_WARNING
- Summary bar shows total / OK / defective / warning counts

### Slot Detail Modal (click any card)
- Toggle between **Original** and **Annotated** image (bounding boxes drawn per detection)
- Defect table showing which stage caught what, label, confidence, and bounding box
- CNN region thumbnails with per-region classification
- Raw JSON report
- **Download PDF** button for that slot

### PDF Reports
- Per-slot PDF: click "Download PDF Report" inside any slot modal
- Full panel PDF: click "⬇ Full PDF Report" in the summary bar
- Each PDF includes annotated image, defect table, and all stage results

---

## REST API

### `POST /api/inspect`
Analyze a panel image programmatically.

**Request:** `multipart/form-data` with field `image`

**Response:** JSON array of per-slot results

```json
[
  {
    "slot_index": 1,
    "slot_label": "pcb",
    "confidence": 0.91,
    "report": {
      "status": "DEFECTIVE",
      "stage": "YOLO initial",
      "yolo_initial": [
        {
          "class_id": 3,
          "label": "missing",
          "confidence": 0.87,
          "bbox_xyxy": [120, 45, 310, 198]
        }
      ],
      "cnn": {},
      "yolo_recheck": []
    }
  }
]
```

### `GET /api/health`
Returns `{"status": "ok"}` — use for uptime checks.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLO inference (crop + defect models) |
| `tensorflow` / `keras` | CNN model loading and inference |
| `keras-cv` | Augmentation layers used in CNN pipeline |
| `opencv-python` | Image reading, cropping, annotation |
| `pillow` | Image conversion for base64 encoding |
| `flask` | Web server and REST API |
| `reportlab` | PDF report generation |

---

## Notes

- Model files are excluded from version control — never commit `.pt` or `.keras` files
- The `venv/` directory is excluded — recreate it with the setup steps above
- Crop model uses `conf=0.5` (matches original training/inference setup — do not change)
- YOLO-1 default `conf=0.25`, YOLO-2 recheck default `conf=0.15` — adjustable via UI sliders
