import os
import io
import base64
import tempfile
import json

import cv2
from flask import Flask, request, jsonify, render_template_string, send_file
from PIL import Image

from src.pipeline.analyzer import analyze_pcb, analyze_panel
from src.data.preprocessing import crop_pcb_for_cnn
from src.utils.helpers import cnn_has_defect
from src.web.annotate import annotate_slot
from src.web.pdf_report import build_pdf

app = Flask(__name__)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def bgr_to_base64(img):
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    buff = io.BytesIO()
    pil.save(buff, format="JPEG", quality=85)
    return base64.b64encode(buff.getvalue()).decode()


def path_to_base64(path):
    img = cv2.imread(path)
    if img is None:
        return ""
    return bgr_to_base64(img)


def status_color(status):
    return {
        "OK":              "#22c55e",
        "DEFECTIVE":       "#ef4444",
        "OK_WITH_WARNING": "#f59e0b",
    }.get(status, "#6b7280")


# ─────────────────────────────────────────────
#  HTML TEMPLATE
# ─────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCB Inspector</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
  }

  header {
    background: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header h1 { font-size: 1.25rem; font-weight: 600; color: #f1f5f9; }
  header span { font-size: 0.8rem; color: #64748b; }

  .container { max-width: 1400px; margin: 0 auto; padding: 32px; }

  /* Upload card */
  .upload-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 32px;
  }
  .upload-card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 20px; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }

  .form-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; }

  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group label { font-size: 0.8rem; color: #94a3b8; }
  .form-group input[type="file"] {
    background: #0f172a;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 8px 12px;
    color: #e2e8f0;
    font-size: 0.875rem;
    cursor: pointer;
    min-width: 260px;
  }

  .btn {
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s;
  }
  .btn:hover { background: #2563eb; }
  .btn:disabled { background: #475569; cursor: not-allowed; }

  /* Summary bar */
  .summary-bar {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }
  .summary-pill {
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .pill-total    { background: #1e293b; border: 1px solid #475569; color: #e2e8f0; }
  .pill-ok       { background: #14532d; color: #86efac; }
  .pill-defect   { background: #450a0a; color: #fca5a5; }
  .pill-warning  { background: #451a03; color: #fcd34d; }

  /* Panel grid */
  .panel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
  }

  .slot-card {
    background: #1e293b;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border-color .2s, transform .15s;
  }
  .slot-card:hover { transform: translateY(-2px); }
  .slot-card.ok      { border-color: #22c55e; }
  .slot-card.defect  { border-color: #ef4444; }
  .slot-card.warning { border-color: #f59e0b; }

  .slot-img { width: 100%; aspect-ratio: 3/4; object-fit: cover; display: block; }

  .slot-footer {
    padding: 10px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .slot-num { font-size: 0.8rem; color: #94a3b8; }
  .badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: .04em;
  }
  .badge-ok      { background: #14532d; color: #86efac; }
  .badge-defect  { background: #450a0a; color: #fca5a5; }
  .badge-warning { background: #451a03; color: #fcd34d; }

  /* Modal */
  .modal-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,.75);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  .modal-overlay.open { display: flex; }

  .modal {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    width: min(900px, 95vw);
    max-height: 90vh;
    overflow-y: auto;
    padding: 28px;
  }
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  .modal-header h3 { font-size: 1.1rem; font-weight: 600; }
  .close-btn {
    background: none; border: none; color: #94a3b8;
    font-size: 1.4rem; cursor: pointer; line-height: 1;
  }
  .close-btn:hover { color: #f1f5f9; }

  .modal-body { display: flex; gap: 24px; flex-wrap: wrap; }
  .modal-img-col { flex: 0 0 200px; }
  .modal-img-col img { width: 100%; border-radius: 8px; cursor: zoom-in; }
  .img-tabs { display: flex; gap: 8px; margin-bottom: 8px; }
  .img-tab {
    font-size: 0.72rem; padding: 3px 10px; border-radius: 6px; cursor: pointer;
    background: #0f172a; border: 1px solid #475569; color: #94a3b8;
  }
  .img-tab.active { background: #3b82f6; border-color: #3b82f6; color: #fff; }

  .modal-detail { flex: 1; min-width: 260px; }
  .detail-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #334155; font-size: 0.85rem; }
  .detail-row:last-child { border-bottom: none; }
  .detail-label { color: #94a3b8; }

  .cnn-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
  .cnn-item { text-align: center; font-size: 0.72rem; color: #94a3b8; }
  .cnn-item img { width: 80px; height: 80px; object-fit: cover; border-radius: 6px; display: block; margin-bottom: 4px; }

  pre.raw-json {
    background: #0f172a;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.75rem;
    overflow-x: auto;
    margin-top: 16px;
    color: #94a3b8;
    max-height: 220px;
    overflow-y: auto;
  }

  .defect-table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 0.78rem; }
  .defect-table th { background: #0f172a; color: #64748b; padding: 5px 8px; text-align: left; font-weight: 600; }
  .defect-table td { padding: 5px 8px; border-bottom: 1px solid #1e293b; }
  .defect-table tr:last-child td { border-bottom: none; }
  .stage-tag {
    font-size: 0.65rem; font-weight: 700; padding: 2px 6px;
    border-radius: 4px; text-transform: uppercase; letter-spacing: .04em;
  }
  .stage-yolo1  { background: #1e3a5f; color: #93c5fd; }
  .stage-cnn    { background: #1a2e1a; color: #86efac; }
  .stage-yolo2  { background: #3b1f1f; color: #fca5a5; }
  .btn-pdf {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0f172a; border: 1px solid #475569; color: #94a3b8;
    border-radius: 8px; padding: 7px 16px; font-size: 0.82rem;
    cursor: pointer; text-decoration: none; margin-top: 14px;
    transition: border-color .2s, color .2s;
  }
  .btn-pdf:hover { border-color: #3b82f6; color: #e2e8f0; }

  .empty-state { text-align: center; padding: 60px 0; color: #475569; }
  .empty-state svg { margin-bottom: 12px; }

  /* Loading overlay */
  #loading {
    display: none;
    position: fixed; inset: 0;
    background: rgba(15,23,42,.85);
    z-index: 200;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
  }
  #loading.show { display: flex; }
  .spinner {
    width: 48px; height: 48px;
    border: 4px solid #334155;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loading p { color: #94a3b8; font-size: 0.9rem; }

  .slider-group { display: flex; flex-direction: column; gap: 6px; min-width: 180px; }
  .slider-group label { font-size: 0.8rem; color: #94a3b8; display: flex; justify-content: space-between; }
  .slider-group label span { color: #e2e8f0; font-weight: 600; }
  input[type="range"] {
    -webkit-appearance: none; width: 100%; height: 4px;
    background: #334155; border-radius: 2px; outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: #3b82f6; cursor: pointer;
  }
</style>
</head>
<body>

<div id="loading">
  <div class="spinner"></div>
  <p>Analyzing PCB panel — this may take a moment…</p>
</div>

<header>
  <svg width="24" height="24" fill="none" stroke="#3b82f6" stroke-width="2" viewBox="0 0 24 24">
    <rect x="2" y="2" width="20" height="20" rx="2"/><path d="M7 7h.01M12 7h.01M17 7h.01M7 12h.01M12 12h.01M17 12h.01M7 17h.01M12 17h.01M17 17h.01"/>
  </svg>
  <h1>PCB Inspector</h1>
  <span>10-slot panel analysis</span>
</header>

<div class="container">

  <div class="upload-card">
    <h2>Upload Panel Image</h2>
    <form id="uploadForm" method="post" action="/inspect" enctype="multipart/form-data">
      <div class="form-row">
        <div class="form-group">
          <label>Panel image (10 PCB slots)</label>
          <input type="file" name="image" accept="image/*" required>
        </div>
        <div class="slider-group">
          <label>YOLO-1 confidence <span id="yi_val">0.25</span></label>
          <input type="range" name="yolo_init" min="0.05" max="0.95" step="0.01" value="0.25"
                 oninput="document.getElementById('yi_val').textContent=parseFloat(this.value).toFixed(2)">
        </div>
        <div class="slider-group">
          <label>YOLO-2 recheck conf <span id="yr_val">0.15</span></label>
          <input type="range" name="yolo_re" min="0.05" max="0.95" step="0.01" value="0.15"
                 oninput="document.getElementById('yr_val').textContent=parseFloat(this.value).toFixed(2)">
        </div>
        <button class="btn" type="submit" id="submitBtn">Analyze Panel</button>
      </div>
    </form>
  </div>

  {% if slots %}

  <!-- Summary -->
  <div class="summary-bar">
    <div class="summary-pill pill-total">{{ slots|length }} slots detected</div>
    <div class="summary-pill pill-ok">✓ {{ ok_count }} OK</div>
    {% if defect_count %}<div class="summary-pill pill-defect">✗ {{ defect_count }} Defective</div>{% endif %}
    {% if warn_count %}<div class="summary-pill pill-warning">⚠ {{ warn_count }} Warning</div>{% endif %}
    <a class="btn-pdf" href="/report/pdf/all" target="_blank" style="margin-left:auto;">⬇ Full PDF Report</a>
  </div>

  <!-- Grid -->
  <div class="panel-grid">
    {% for s in slots %}
    {% set st = s.report.status %}
    {% set css = "ok" if st == "OK" else ("defect" if st == "DEFECTIVE" else "warning") %}
    {% set badge = "badge-ok" if st == "OK" else ("badge-defect" if st == "DEFECTIVE" else "badge-warning") %}
    <div class="slot-card {{ css }}" onclick="openModal({{ loop.index0 }})">
      <img class="slot-img" src="data:image/jpeg;base64,{{ s.b64 }}" alt="Slot {{ s.slot_index }}">
      <div class="slot-footer">
        <span class="slot-num">Slot {{ s.slot_index }}</span>
        <span class="badge {{ badge }}">{{ st }}</span>
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- Modals -->
  {% for s in slots %}
  {% set st = s.report.status %}
  {% set badge = "badge-ok" if st == "OK" else ("badge-defect" if st == "DEFECTIVE" else "badge-warning") %}
  <div class="modal-overlay" id="modal-{{ loop.index0 }}" onclick="closeModal(event, {{ loop.index0 }})">
    <div class="modal">
      <div class="modal-header">
        <h3>Slot {{ s.slot_index }} &nbsp;<span class="badge {{ badge }}">{{ st }}</span></h3>
        <button class="close-btn" onclick="closeModalById({{ loop.index0 }})">&times;</button>
      </div>
      <div class="modal-body">
        <div class="modal-img-col">
          <div class="img-tabs">
            <span class="img-tab active" onclick="switchImg({{ loop.index0 }},'plain',this)">Original</span>
            <span class="img-tab" onclick="switchImg({{ loop.index0 }},'annot',this)">Annotated</span>
          </div>
          <img id="slotimg-{{ loop.index0 }}"
               src="data:image/jpeg;base64,{{ s.b64 }}"
               data-plain="{{ s.b64 }}"
               data-annot="{{ s.annotated_b64 }}"
               alt="Slot {{ s.slot_index }}">
        </div>
        <div class="modal-detail">
          <div class="detail-row"><span class="detail-label">Status</span><span>{{ st }}</span></div>
          <div class="detail-row"><span class="detail-label">Stage stopped at</span><span>{{ s.report.stage }}</span></div>
          <div class="detail-row"><span class="detail-label">Crop label</span><span>{{ s.slot_label }}</span></div>
          <div class="detail-row"><span class="detail-label">Crop confidence</span><span>{{ "%.2f"|format(s.confidence) }}</span></div>
          <div class="detail-row"><span class="detail-label">YOLO-1 detections</span><span>{{ s.report.yolo_initial|length }}</span></div>
          <div class="detail-row"><span class="detail-label">YOLO-2 detections</span><span>{{ s.report.yolo_recheck|length }}</span></div>

          {% if s.defect_rows %}
          <table class="defect-table">
            <thead><tr><th>Stage</th><th>Label</th><th>Confidence</th><th>BBox</th></tr></thead>
            <tbody>
            {% for row in s.defect_rows %}
            <tr>
              <td><span class="stage-tag stage-{{ row.stage_css }}">{{ row.stage }}</span></td>
              <td>{{ row.label }}</td>
              <td>{{ row.conf }}</td>
              <td style="font-size:0.7rem;color:#64748b">{{ row.bbox }}</td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
          {% endif %}

          {% if s.cnn_parts %}
          <p style="margin-top:14px; font-size:0.8rem; color:#94a3b8;">CNN regions</p>
          <div class="cnn-grid">
            {% for part in s.cnn_parts %}
            <div class="cnn-item">
              <img src="data:image/jpeg;base64,{{ part.img }}" alt="{{ part.name }}">
              {{ part.name }}<br>
              <span style="color:{% if part.defect %}#fca5a5{% else %}#86efac{% endif %}">{{ part.label }}</span>
            </div>
            {% endfor %}
          </div>
          {% endif %}
        </div>
      </div>
      <a class="btn-pdf" href="/report/pdf/{{ loop.index0 }}" target="_blank">
        ⬇ Download PDF Report
      </a>
      <pre class="raw-json">{{ s.raw_json }}</pre>
    </div>
  </div>
  {% endfor %}

  {% else %}
  <div class="empty-state">
    <svg width="48" height="48" fill="none" stroke="#475569" stroke-width="1.5" viewBox="0 0 24 24">
      <rect x="2" y="2" width="20" height="20" rx="2"/><path d="M9 9h6M9 12h6M9 15h4"/>
    </svg>
    <p>Upload a panel image to begin inspection</p>
  </div>
  {% endif %}

</div>

<script>
  document.getElementById('uploadForm').addEventListener('submit', function() {
    document.getElementById('loading').classList.add('show');
    document.getElementById('submitBtn').disabled = true;
  });

  function openModal(idx) {
    document.getElementById('modal-' + idx).classList.add('open');
  }
  function closeModalById(idx) {
    document.getElementById('modal-' + idx).classList.remove('open');
  }
  function closeModal(event, idx) {
    if (event.target === event.currentTarget) closeModalById(idx);
  }
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.open').forEach(function(m) {
        m.classList.remove('open');
      });
    }
  });

  function switchImg(idx, mode, tab) {
    var img = document.getElementById('slotimg-' + idx);
    img.src = 'data:image/jpeg;base64,' + img.dataset[mode];
    tab.closest('.img-tabs').querySelectorAll('.img-tab').forEach(function(t){ t.classList.remove('active'); });
    tab.classList.add('active');
  }
</script>

</body>
</html>
"""


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/inspect", methods=["POST"])
def inspect():
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400

    tmp_dir  = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, file.filename)
    file.save(img_path)

    initial_conf = float(request.form.get("yolo_init", 0.25))
    recheck_conf = float(request.form.get("yolo_re",   0.15))

    panel_results = analyze_panel(img_path, initial_conf=initial_conf, recheck_conf=recheck_conf)

    ok_count     = sum(1 for r in panel_results if r["report"]["status"] == "OK")
    defect_count = sum(1 for r in panel_results if r["report"]["status"] == "DEFECTIVE")
    warn_count   = sum(1 for r in panel_results if r["report"]["status"] == "OK_WITH_WARNING")

    # Build template context per slot
    slots_ctx = []
    for r in panel_results:
        slot_img = cv2.imread(r["image_path"])
        b64      = bgr_to_base64(slot_img) if slot_img is not None else ""

        # CNN crop thumbnails for modal
        cnn_parts = []
        if slot_img is not None:
            crops = crop_pcb_for_cnn(slot_img)
            cnn_report = r["report"].get("cnn", {})
            for name, cimg in crops.items():
                # find matching cnn result by filename suffix
                matched_label = ""
                matched_defect = False
                for path, val in cnn_report.items():
                    if name in path:
                        matched_label  = val["class"]
                        matched_defect = cnn_has_defect({path: val})
                        break
                cnn_parts.append({
                    "name":   name,
                    "img":    bgr_to_base64(cimg),
                    "label":  matched_label,
                    "defect": matched_defect,
                })

        # annotated image with bounding boxes
        annotated_b64 = ""
        if slot_img is not None:
            ann = annotate_slot(slot_img, r["report"])
            annotated_b64 = bgr_to_base64(ann)

        # defect rows table (YOLO-1 / CNN / YOLO-2)
        defect_rows = []
        for det in r["report"].get("yolo_initial", []):
            if det["label"] != "correct":
                defect_rows.append({
                    "stage": "YOLO-1", "stage_css": "yolo1",
                    "label": det["label"],
                    "conf":  f'{det["confidence"]:.2f}',
                    "bbox":  str([int(v) for v in det["bbox_xyxy"]]),
                })
        for fpath, val in r["report"].get("cnn", {}).items():
            region = os.path.basename(fpath).replace(".jpg", "")
            if cnn_has_defect({fpath: val}):
                defect_rows.append({
                    "stage": "CNN", "stage_css": "cnn",
                    "label": val["class"],
                    "conf":  f'{val["confidence"]:.2f}',
                    "bbox":  region,
                })
        for det in r["report"].get("yolo_recheck", []):
            if det["label"] != "correct":
                defect_rows.append({
                    "stage": "YOLO-2", "stage_css": "yolo2",
                    "label": det["label"],
                    "conf":  f'{det["confidence"]:.2f}',
                    "bbox":  str([int(v) for v in det["bbox_xyxy"]]),
                })

        slots_ctx.append({
            "slot_index":    r["slot_index"],
            "slot_label":    r["slot_label"],
            "confidence":    r["confidence"],
            "report":        r["report"],
            "b64":           b64,
            "annotated_b64": annotated_b64,
            "defect_rows":   defect_rows,
            "cnn_parts":     cnn_parts,
            "raw_json":      json.dumps(r["report"], indent=2),
        })

    app.config["LAST_SLOTS_CTX"] = slots_ctx

    return render_template_string(
        HTML,
        slots=slots_ctx,
        ok_count=ok_count,
        defect_count=defect_count,
        warn_count=warn_count,
    )


# ─────────────────────────────────────────────
#  PDF DOWNLOAD
# ─────────────────────────────────────────────

@app.route("/report/pdf/<int:slot_idx>")
def download_pdf_slot(slot_idx):
    """Download PDF report for a single slot."""
    slots = app.config.get("LAST_SLOTS_CTX", [])
    if not slots:
        return "No inspection data. Run an inspection first.", 404
    subset = [slots[slot_idx]] if slot_idx < len(slots) else slots
    buf = build_pdf(subset)
    fname = f"pcb_slot_{slot_idx + 1}_report.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=fname)


@app.route("/report/pdf/all")
def download_pdf_all():
    """Download PDF report for all slots."""
    slots = app.config.get("LAST_SLOTS_CTX", [])
    if not slots:
        return "No inspection data. Run an inspection first.", 404
    buf = build_pdf(slots)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name="pcb_full_report.pdf")


# ─────────────────────────────────────────────
#  REST API  (JSON endpoints)
# ─────────────────────────────────────────────

@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """
    POST /api/inspect
    multipart/form-data  field: image
    Returns JSON list of per-slot results.
    """
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    tmp_dir  = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, file.filename)
    file.save(img_path)

    panel_results = analyze_panel(img_path)

    # strip numpy arrays before serialising
    output = []
    for r in panel_results:
        output.append({
            "slot_index": r["slot_index"],
            "slot_label": r["slot_label"],
            "confidence": r["confidence"],
            "report":     r["report"],
        })

    return jsonify(output)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
