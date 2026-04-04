import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from PIL import Image
import keras
import tensorflow as tf
import base64
import io
import logging
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

CLASS_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
RISK_TIERS  = ["MONITOR", "MONITOR", "ENGAGE", "ACT NOW", "ACT NOW"]
RISK_COLORS = ["#10b981", "#10b981", "#f59e0b", "#ef4444", "#ef4444"]
BAR_COLORS  = ["#10b981", "#06b6d4", "#f59e0b", "#f97316", "#ef4444"]

CLINICAL_DESC = [
    "No diabetic retinopathy detected. Retinal vasculature appears healthy.",
    "Early microaneurysms present. No vision-threatening features at this stage.",
    "Moderate non-proliferative changes detected. Ophthalmology referral within 30 days.",
    "Severe non-proliferative retinopathy. Urgent referral required within 1 week.",
    "Proliferative DR confirmed. Immediate specialist intervention required."
]

AI_SUMMARIES = [
    {
        "summary": "Retinal imaging shows no evidence of diabetic retinopathy. Vascular architecture appears intact with no microaneurysms, exudates, or neovascularization detected.",
        "action": "Continue standard diabetes management. Schedule next retinal screening in 12 months.",
        "hcp": "Automated digital reminder via patient portal. No urgent HCP outreach required.",
        "channel": "Digital"
    },
    {
        "summary": "Early-stage microaneurysms identified in the peripheral retinal field. Consistent with mild non-proliferative diabetic retinopathy. No vision-threatening features present.",
        "action": "Optimise glycaemic and blood pressure control. Repeat retinal imaging in 6-12 months.",
        "hcp": "Flag for primary care physician review. Consider patient education on glycaemic targets.",
        "channel": "Digital + PCP Outreach"
    },
    {
        "summary": "Moderate non-proliferative diabetic retinopathy identified. Microaneurysms, retinal haemorrhages, and hard exudates detected in the central field. Progression risk elevated without intervention.",
        "action": "Ophthalmology referral within 30 days. Evaluate candidacy for anti-VEGF therapy.",
        "hcp": "Flag in CRM for HCP outreach via primary care provider. Consider therapy education for prescriber.",
        "channel": "CRM Flag + HCP Engagement"
    },
    {
        "summary": "Severe non-proliferative diabetic retinopathy detected. Extensive haemorrhages, venous beading, and intraretinal microvascular abnormalities across multiple quadrants. High risk of progression to PDR.",
        "action": "Urgent ophthalmology referral within 1 week. Consider pan-retinal photocoagulation or intravitreal injection.",
        "hcp": "Immediate field rep visit recommended. Prioritise specialist referral pathway and therapy initiation.",
        "channel": "Field Rep + Specialist Referral"
    },
    {
        "summary": "Proliferative diabetic retinopathy confirmed. Active neovascularization and vitreous haemorrhage signs detected. Highest-risk stage with imminent threat to vision.",
        "action": "Same-week specialist intervention required. Anti-VEGF therapy or surgical evaluation indicated.",
        "hcp": "Urgent field rep visit. Flag for immediate specialist coordination. Critical therapy initiation opportunity.",
        "channel": "Urgent Field Rep + Specialist"
    }
]

model = keras.models.load_model('models/diabetic_retinopathy_model.keras')

# Log model structure at startup to help diagnose Grad-CAM issues
logging.info("Top-level model layers:")
for i, l in enumerate(model.layers):
    logging.info("  [%d] %s (%s)", i, l.name, type(l).__name__)
    if hasattr(l, 'layers'):
        conv_names = [sl.name for sl in l.layers if 'conv' in sl.name.lower()]
        logging.info("      last 3 conv layers: %s", conv_names[-3:])


def _find_last_conv(layers):
    """Return the last Conv layer from a list, checking by type then name."""
    for l in reversed(layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l
    for l in reversed(layers):
        if 'conv' in l.name.lower():
            return l
    return None


def make_gradcam(img_array, model):
    try:
        # Find sub-model (e.g. EfficientNet) and last conv layer
        sub_model = next((l for l in model.layers if hasattr(l, 'layers')), None)

        if sub_model is not None:
            last_conv = _find_last_conv(sub_model.layers)
            grad_model = tf.keras.Model(
                inputs=sub_model.inputs,
                outputs=[last_conv.output, sub_model.output]
            )
        else:
            # Flat model — search top-level layers
            last_conv = _find_last_conv(model.layers)
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[last_conv.output, model.output]
            )

        if last_conv is None:
            logging.warning("Grad-CAM: no Conv2D layer found in model")
            return None
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(img_array, training=False)
            pred_class = tf.argmax(predictions[0])
            class_score = predictions[:, pred_class]
        grads = tape.gradient(class_score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1).numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR),
            dtype=np.float32
        ) / 255.0  # shape (224, 224), values in [0, 1]

        # Vectorized HSV->RGB heatmap (hue: 0.67=blue at low activation, 0=red at high)
        hue = (1.0 - cam_resized) * 0.67
        hi = (hue * 6).astype(int) % 6
        f  = hue * 6 - np.floor(hue * 6)
        ones  = np.ones_like(f)
        zeros = np.zeros_like(f)
        r = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5], [ones,   1-f,   zeros, zeros, f,     ones ])
        g = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5], [f,      ones,  ones,  1-f,   zeros, zeros])
        b = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5], [zeros,  zeros, f,     ones,  ones,  1-f  ])
        heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        orig = (img_array[0] * 255).astype(np.uint8)
        blended = (0.55 * orig + 0.45 * heatmap).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logging.warning("Grad-CAM failed: %s", e)
        return None


def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Retinal AI | DR Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #080b12;
  --surface: #0f1320;
  --surface2: #161c2e;
  --surface3: #1d2440;
  --border: #242d4a;
  --text: #eef0f8;
  --muted: #7b82a0;
  --accent: #6366f1;
  --accent2: #818cf8;
}
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; }
/* HEADER */
.header {
  background: #0d0a2e;
  border-bottom: 1px solid #3730a344;
  padding: 0 40px; height: 68px;
  display: flex; align-items: center; gap: 14px;
}
.logo {
  width: 38px; height: 38px; border-radius: 9px;
  background: linear-gradient(135deg, #6366f1, #4338ca);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; flex-shrink: 0;
}
.htitle { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.02em; }
.hsub { font-size: 0.72rem; color: #a5b4fc; margin-top: 2px; }
.hpills { margin-left: auto; display: flex; gap: 6px; }
.hpill {
  background: #3730a322; border: 1px solid #6366f133;
  color: #a5b4fc; font-size: 0.65rem; font-weight: 600;
  padding: 3px 10px; border-radius: 20px;
}
/* LAYOUT */
.main { max-width: 980px; margin: 0 auto; padding: 32px 24px 60px; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 22px;
}
.card-label {
  font-size: 0.67rem; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px;
}
/* UPLOAD STATE */
#uploadState { margin-bottom: 0; }
.upload-zone {
  border: 2px dashed var(--border); border-radius: 14px;
  padding: 52px 24px; text-align: center; cursor: pointer;
  transition: all 0.2s; position: relative; background: var(--surface2);
}
.upload-zone:hover { border-color: var(--accent); background: #6366f10a; }
.upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
.upload-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.upload-sub { font-size: 0.8rem; color: var(--muted); line-height: 1.6; }
.upload-sub span { color: var(--accent2); font-weight: 600; }
.upload-hint { font-size: 0.72rem; color: var(--muted); margin-top: 8px; opacity: 0.7; }
.btn-primary {
  width: 100%; margin-top: 14px;
  background: linear-gradient(135deg, #6366f1, #4338ca);
  border: none; border-radius: 11px; color: white;
  font-size: 0.88rem; font-weight: 700; padding: 14px;
  cursor: pointer; transition: all 0.2s;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  box-shadow: 0 4px 18px #6366f128;
  font-family: 'Inter', sans-serif;
}
.btn-primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 22px #6366f140; }
.btn-primary:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; transform: none; }
.btn-secondary {
  width: 100%; margin-top: 10px;
  background: transparent; border: 1px solid var(--border);
  border-radius: 10px; color: var(--muted);
  font-size: 0.78rem; font-weight: 500; padding: 10px;
  cursor: pointer; transition: all 0.2s;
  font-family: 'Inter', sans-serif;
}
.btn-secondary:hover { border-color: var(--accent); color: var(--accent2); }
/* RESULTS STATE */
#resultsState { display: none; }
/* SEVERITY SCALE */
.severity-scale {
  display: flex; margin-bottom: 18px; border-radius: 11px; overflow: hidden;
  border: 1px solid var(--border);
}
.sev-item {
  flex: 1; text-align: center; padding: 10px 6px;
  font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; background: var(--surface); color: var(--muted);
  border-right: 1px solid var(--border); transition: all 0.3s;
}
.sev-item:last-child { border-right: none; }
.sev-dot { width: 5px; height: 5px; border-radius: 50%; margin: 0 auto 5px; background: currentColor; opacity: 0.5; }
.sev-item.sev-active { z-index: 1; }
/* TOP GRID */
.top-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
@media(max-width: 640px) { .top-grid { grid-template-columns: 1fr; } }
/* IMAGE CARD */
.scan-img-wrap { border-radius: 11px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 10px; }
.scan-img-wrap img { width: 100%; display: block; max-height: 210px; object-fit: cover; }
/* RESULT CARD */
.risk-pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 14px; border-radius: 8px;
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em;
  margin-bottom: 10px;
}
.risk-dot { width: 7px; height: 7px; border-radius: 50%; }
.result-diagnosis { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 7px; }
.result-conf-row { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
.result-conf-text { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
.conf-track { flex: 1; background: var(--surface2); border-radius: 100px; height: 5px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 100px; transition: width 0.9s cubic-bezier(0.4,0,0.2,1); }
.result-desc {
  font-size: 0.78rem; color: var(--muted); line-height: 1.65;
  padding: 10px 13px; background: var(--surface2);
  border-radius: 9px; border-left: 3px solid var(--accent); margin-bottom: 14px;
}
.divider { height: 1px; background: var(--border); margin: 13px 0; }
.scores-title { font-size: 0.65rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
.bar-row { margin-bottom: 8px; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 4px; }
.bar-label span:last-child { color: var(--muted); }
.bar-track { background: var(--surface2); border-radius: 5px; height: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1); width: 0; }
/* GRADCAM */
.gradcam-card { margin-bottom: 18px; }
.gradcam-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
@media(max-width: 560px) { .gradcam-grid { grid-template-columns: 1fr; } }
.gcam-wrap { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
.gcam-wrap img { width: 100%; display: block; max-height: 180px; object-fit: cover; }
.gcam-label { font-size: 0.68rem; color: var(--muted); text-align: center; margin-top: 8px; font-weight: 500; }
.gradcam-note {
  margin-top: 14px; padding: 11px 14px;
  background: #1a1b3a; border: 1px solid #4338ca33;
  border-radius: 10px; font-size: 0.76rem; color: #a5b4fc; line-height: 1.65;
}
/* MODEL PERFORMANCE */
.perf-card { margin-bottom: 18px; }
.perf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
@media(max-width: 560px) { .perf-grid { grid-template-columns: repeat(2, 1fr); } }
.perf-metric { background: var(--surface2); border-radius: 10px; padding: 14px; text-align: center; }
.perf-val { font-size: 1.25rem; font-weight: 800; color: var(--accent2); margin-bottom: 3px; }
.perf-lbl { font-size: 0.63rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
/* AI CARD */
.ai-card { margin-bottom: 0; }
.ai-header { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.ai-icon-box {
  width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
  background: #4338ca22; border: 1px solid #6366f133;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 800; color: #a5b4fc;
}
.ai-title { font-size: 0.88rem; font-weight: 700; color: #c7d2fe; }
.ai-sub { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }
.ai-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
@media(max-width: 640px) { .ai-grid { grid-template-columns: 1fr; } }
.ai-panel { background: var(--surface2); border-radius: 12px; padding: 16px; border: 1px solid var(--border); }
.ai-panel-title { font-size: 0.65rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 8px; }
.ai-panel-text { font-size: 0.78rem; color: var(--text); line-height: 1.65; }
.channel-tag {
  display: inline-block; margin-top: 10px;
  background: #312e8122; border: 1px solid #6366f133;
  color: #a5b4fc; font-size: 0.68rem; font-weight: 600;
  padding: 4px 10px; border-radius: 20px;
}
/* FADE IN */
.fade-in { animation: fadeIn 0.4s ease forwards; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.spinner { width: 15px; height: 15px; border: 2px solid rgba(255,255,255,0.25); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.footer { text-align: center; color: var(--muted); font-size: 0.68rem; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); line-height: 1.8; padding-bottom: 20px; }
.footer a { color: var(--accent2); text-decoration: none; }
.footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
<header class="header">
  <div class="logo">&#128065;</div>
  <div>
    <div class="htitle">Retinal AI Classifier</div>
    <div class="hsub">Diabetic Retinopathy Severity Detection</div>
  </div>
  <div class="hpills">
    <div class="hpill">EfficientNetB3</div>
    <div class="hpill">APTOS 2019</div>
    <div class="hpill">Grad-CAM</div>
  </div>
</header>
<main class="main">
  <!-- UPLOAD STATE -->
  <div id="uploadState">
    <div class="card" style="margin-bottom: 18px;">
      <div class="card-label">Upload Retinal Image</div>
      <div class="upload-zone" id="uploadZone">
        <input type="file" id="imageInput" accept="image/*">
        <div class="upload-title">Upload a retinal fundus image</div>
        <div class="upload-sub"><span>Click to browse</span> or drag and drop</div>
        <div class="upload-hint">PNG, JPG, JPEG supported</div>
      </div>
      <button class="btn-primary" id="analyzeBtn" onclick="analyze()" disabled>
        <span id="btnText">Select an image to begin</span>
        <div class="spinner" id="spinner" style="display:none"></div>
      </button>
    </div>
  </div>
  <!-- RESULTS STATE -->
  <div id="resultsState">
    <!-- Severity Scale -->
    <div class="severity-scale" id="severityScale">
      <div class="sev-item" id="sev0"><div class="sev-dot"></div>No DR</div>
      <div class="sev-item" id="sev1"><div class="sev-dot"></div>Mild</div>
      <div class="sev-item" id="sev2"><div class="sev-dot"></div>Moderate</div>
      <div class="sev-item" id="sev3"><div class="sev-dot"></div>Severe</div>
      <div class="sev-item" id="sev4"><div class="sev-dot"></div>Proliferative</div>
    </div>
    <!-- Top Grid: Scan + Result -->
    <div class="top-grid">
      <div class="card">
        <div class="card-label">Retinal Scan</div>
        <div class="scan-img-wrap">
          <img id="scanPreview" alt="Retinal scan">
        </div>
        <button class="btn-secondary" onclick="resetToUpload()">Upload another image</button>
      </div>
      <div class="card" id="resultCard">
        <div class="card-label">Classification Result</div>
        <div id="resultBody"></div>
      </div>
    </div>
    <!-- Grad-CAM -->
    <div class="card gradcam-card" id="gradcamCard" style="display:none;">
      <div class="card-label">Model Attention &mdash; Grad-CAM Visualization</div>
      <div class="gradcam-grid">
        <div>
          <div class="gcam-wrap"><img id="origImg" alt="Original scan"></div>
          <div class="gcam-label">Original Retinal Scan</div>
        </div>
        <div>
          <div class="gcam-wrap"><img id="camImg" alt="Attention heatmap"></div>
          <div class="gcam-label">AI Attention Heatmap &mdash; red zones indicate highest model focus</div>
        </div>
      </div>
      <div class="gradcam-note">
        The heatmap highlights exactly which regions of the retina influenced the AI classification. Red zones indicate areas of highest model attention &mdash; typically where retinal damage features are detected. This allows clinicians to verify the AI reasoning in seconds, addressing the trust barrier to AI adoption in clinical settings.
      </div>
    </div>
    <!-- Model Performance -->
    <div class="card perf-card">
      <div class="card-label">Model Performance</div>
      <div class="perf-grid">
        <div class="perf-metric"><div class="perf-val">74%</div><div class="perf-lbl">Val Accuracy</div></div>
        <div class="perf-metric"><div class="perf-val">0.59</div><div class="perf-lbl">Cohen's Kappa</div></div>
        <div class="perf-metric"><div class="perf-val">3,662</div><div class="perf-lbl">Images Trained</div></div>
        <div class="perf-metric"><div class="perf-val">5</div><div class="perf-lbl">Severity Classes</div></div>
      </div>
    </div>
    <!-- AI Clinical Intelligence -->
    <div class="card ai-card">
      <div class="ai-header">
        <div class="ai-icon-box">AI</div>
        <div>
          <div class="ai-title">AI-Generated Clinical Intelligence</div>
          <div class="ai-sub">Automated insight generation &middot; Pharma commercial decision support</div>
        </div>
      </div>
      <div class="ai-grid">
        <div class="ai-panel">
          <div class="ai-panel-title">Clinical Summary</div>
          <div class="ai-panel-text" id="aiSummary"></div>
        </div>
        <div class="ai-panel">
          <div class="ai-panel-title">Recommended Action</div>
          <div class="ai-panel-text" id="aiAction"></div>
        </div>
        <div class="ai-panel">
          <div class="ai-panel-title">HCP Engagement</div>
          <div class="ai-panel-text" id="aiHcp"></div>
          <div class="channel-tag" id="aiChannel"></div>
        </div>
      </div>
    </div>
  </div>
</main>
<footer class="footer">
  <a href="https://github.com/priyankaraghunathan15/diabetic-retinopathy-detection" target="_blank">GitHub Repository</a>
  &nbsp;&middot;&nbsp;
  <a href="https://www.kaggle.com/competitions/aptos2019-blindness-detection" target="_blank">APTOS 2019 Dataset</a>
  <br>
  Built on EfficientNetB3 &middot; Trained on APTOS 2019 Blindness Detection Dataset<br>
  For demonstration and educational purposes only &middot; Not for clinical use
</footer>
<script>
const LABELS      = ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"];
const TIERS       = ["MONITOR","MONITOR","ENGAGE","ACT NOW","ACT NOW"];
const TIER_COLORS = ["#10b981","#10b981","#f59e0b","#ef4444","#ef4444"];
const BAR_COLORS  = ["#10b981","#06b6d4","#f59e0b","#f97316","#ef4444"];
document.getElementById('imageInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('scanPreview').src = ev.target.result;
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('btnText').textContent = 'Analyze Image';
    document.getElementById('uploadZone').style.borderColor = '#6366f1';
  };
  reader.readAsDataURL(file);
});
async function analyze() {
  const input = document.getElementById('imageInput');
  if (!input.files[0]) return;
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  document.getElementById('btnText').textContent = 'Analyzing...';
  document.getElementById('spinner').style.display = 'block';
  const formData = new FormData();
  formData.append('file', input.files[0]);
  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    renderResults(data);
  } catch(e) {
    alert('Error analyzing image. Please try again.');
  }
  btn.disabled = false;
  document.getElementById('btnText').textContent = 'Analyze Image';
  document.getElementById('spinner').style.display = 'none';
}
function renderResults(data) {
  const i = data.predicted_class;
  const color = TIER_COLORS[i];
  document.getElementById('uploadState').style.display = 'none';
  document.getElementById('resultsState').style.display = 'block';
  document.getElementById('resultsState').classList.add('fade-in');
  const sevColors = ["#10b981","#10b981","#f59e0b","#ef4444","#ef4444"];
  const sevBg     = ["#052e16","#052e16","#1c1408","#1c0a0a","#1c0a0a"];
  for (let j = 0; j < 5; j++) {
    const el = document.getElementById('sev' + j);
    if (j === i) {
      el.style.background   = sevBg[i];
      el.style.color        = sevColors[i];
      el.style.borderBottom = '2px solid ' + sevColors[i];
      el.querySelector('.sev-dot').style.opacity = '1';
    } else {
      el.style.background   = 'var(--surface)';
      el.style.color        = 'var(--muted)';
      el.style.borderBottom = 'none';
    }
  }
  document.getElementById('resultBody').innerHTML = `
    <div class="risk-pill" style="background:${color}18;border:1px solid ${color}44">
      <div class="risk-dot" style="background:${color}"></div>
      <span style="color:${color}">${TIERS[i]}</span>
    </div>
    <div class="result-diagnosis" style="color:${color}">${LABELS[i]}</div>
    <div class="result-conf-row">
      <span class="result-conf-text">${(data.confidence*100).toFixed(1)}% confidence</span>
      <div class="conf-track">
        <div class="conf-fill" id="confFill" style="background:${color};width:0%"></div>
      </div>
    </div>
    <div class="result-desc">${data.description}</div>
    <div class="divider"></div>
    <div class="scores-title">Confidence Scores</div>
    ${data.probabilities.map((p,j) => `
      <div class="bar-row">
        <div class="bar-label">
          <span>${LABELS[j]}</span>
          <span>${(p*100).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" id="bar${j}" style="background:${BAR_COLORS[j]}"></div>
        </div>
      </div>`).join('')}`;
  setTimeout(() => {
    document.getElementById('confFill').style.width = (data.confidence*100).toFixed(1) + '%';
    data.probabilities.forEach((p, j) => {
      const el = document.getElementById('bar' + j);
      if (el) el.style.width = (p*100).toFixed(1) + '%';
    });
  }, 80);
  if (data.gradcam) {
    document.getElementById('origImg').src = 'data:image/png;base64,' + data.original_image;
    document.getElementById('camImg').src  = 'data:image/png;base64,' + data.gradcam;
    document.getElementById('gradcamCard').style.display = 'block';
  }
  document.getElementById('aiSummary').textContent = data.ai_summary;
  document.getElementById('aiAction').textContent  = data.ai_action;
  document.getElementById('aiHcp').textContent     = data.ai_hcp;
  document.getElementById('aiChannel').textContent = data.ai_channel;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function resetToUpload() {
  document.getElementById('uploadState').style.display   = 'block';
  document.getElementById('resultsState').style.display  = 'none';
  document.getElementById('gradcamCard').style.display   = 'none';
  document.getElementById('imageInput').value            = '';
  document.getElementById('analyzeBtn').disabled         = true;
  document.getElementById('btnText').textContent         = 'Select an image to begin';
  document.getElementById('uploadZone').style.borderColor = '';
}
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream).resize((224, 224)).convert('RGB')
    img_array = np.array(image).astype('float32') / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_input, verbose=0)[0]
    predicted_class = int(np.argmax(probs))
    summary = AI_SUMMARIES[predicted_class]

    orig_b64    = img_to_b64(image)
    gradcam_b64 = make_gradcam(img_input, model)

    return jsonify({
        'predicted_class': predicted_class,
        'label':           CLASS_LABELS[predicted_class],
        'confidence':      float(probs[predicted_class]),
        'probabilities':   [float(p) for p in probs],
        'description':     CLINICAL_DESC[predicted_class],
        'ai_summary':      summary['summary'],
        'ai_action':       summary['action'],
        'ai_hcp':          summary['hcp'],
        'ai_channel':      summary['channel'],
        'original_image':  orig_b64,
        'gradcam':         gradcam_b64
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
