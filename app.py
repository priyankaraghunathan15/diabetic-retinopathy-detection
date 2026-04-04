import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from PIL import Image
import keras
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

CLASS_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
RISK_TIERS = ["MONITOR", "MONITOR", "ENGAGE", "ACT NOW", "ACT NOW"]
RISK_COLORS = ["#10b981", "#10b981", "#f59e0b", "#ef4444", "#ef4444"]
BAR_COLORS = ["#10b981", "#06b6d4", "#f59e0b", "#f97316", "#ef4444"]
CLINICAL_DESC = [
    "No signs of diabetic retinopathy detected. Routine annual screening recommended.",
    "Early microaneurysms present. Increased monitoring frequency advised.",
    "Moderate non-proliferative changes detected. Ophthalmology referral recommended within 30 days.",
    "Severe non-proliferative retinopathy. Urgent ophthalmology referral required within 1 week.",
    "Proliferative diabetic retinopathy. Immediate intervention required — same-week specialist visit."
]

# Pre-generated Gen AI clinical summaries (one per severity class)
AI_SUMMARIES = [
    {
        "summary": "Retinal imaging shows no evidence of diabetic retinopathy at this time. Vascular architecture appears intact with no microaneurysms, exudates, or neovascularization detected.",
        "action": "Continue standard diabetes management. Schedule next retinal screening in 12 months.",
        "hcp": "Automated digital reminder via patient portal. No urgent HCP outreach required.",
        "channel": "Digital"
    },
    {
        "summary": "Early-stage microaneurysms identified in the peripheral retinal field. Changes are consistent with mild non-proliferative diabetic retinopathy. No vision-threatening features present.",
        "action": "Optimise glycaemic and blood pressure control. Repeat retinal imaging in 6–12 months.",
        "hcp": "Flag for primary care physician review. Consider patient education on glycaemic targets.",
        "channel": "Digital + PCP Outreach"
    },
    {
        "summary": "Moderate non-proliferative diabetic retinopathy identified. Microaneurysms, retinal haemorrhages, and hard exudates detected in the central field. Progression risk is elevated without intervention.",
        "action": "Ophthalmology referral within 30 days. Evaluate candidacy for anti-VEGF therapy.",
        "hcp": "Flag in CRM for HCP outreach via primary care provider. Consider therapy education for prescriber.",
        "channel": "CRM Flag + HCP Engagement"
    },
    {
        "summary": "Severe non-proliferative diabetic retinopathy detected. Extensive haemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA) present across multiple quadrants. High risk of progression to PDR.",
        "action": "Urgent ophthalmology referral within 1 week. Patient should be considered for pan-retinal photocoagulation or intravitreal injection evaluation.",
        "hcp": "Immediate field rep visit recommended. Prioritise specialist referral pathway and therapy initiation discussion.",
        "channel": "Field Rep + Specialist Referral"
    },
    {
        "summary": "Proliferative diabetic retinopathy confirmed. Active neovascularization and vitreous haemorrhage signs detected. This represents the highest-risk stage with imminent threat to vision.",
        "action": "Same-week specialist intervention required. Anti-VEGF therapy or surgical evaluation indicated.",
        "hcp": "Urgent field rep visit. Flag for immediate specialist coordination. This patient represents a critical therapy initiation opportunity.",
        "channel": "Urgent Field Rep + Specialist Coordination"
    }
]

model = keras.models.load_model('models/diabetic_retinopathy_model.keras')

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DR Detection | Clinical AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #222536;
    --border: #2e3148;
    --text: #f1f3f9;
    --muted: #8b8fa8;
    --accent: #6366f1;
    --accent-light: #818cf8;
  }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; }

  .header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    border-bottom: 1px solid #4338ca44;
    padding: 20px 32px;
    display: flex; align-items: center; gap: 16px;
  }
  .header-icon { font-size: 2rem; }
  .header-title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
  .header-sub { font-size: 0.8rem; color: #a5b4fc; margin-top: 2px; }
  .header-badge {
    margin-left: auto;
    background: #4338ca33; border: 1px solid #6366f144;
    color: #a5b4fc; font-size: 0.7rem; font-weight: 600;
    padding: 4px 10px; border-radius: 20px; letter-spacing: 0.05em;
  }

  .main { max-width: 960px; margin: 0 auto; padding: 40px 24px; }

  .top-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  @media (max-width: 640px) { .top-grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 24px;
  }
  .card-title {
    font-size: 0.72rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;
  }

  .upload-zone {
    border: 2px dashed var(--border); border-radius: 12px;
    padding: 36px 24px; text-align: center; cursor: pointer;
    transition: all 0.2s; position: relative;
  }
  .upload-zone:hover { border-color: var(--accent); background: #6366f108; }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .upload-icon { font-size: 2.2rem; margin-bottom: 10px; }
  .upload-text { font-size: 0.85rem; color: var(--muted); line-height: 1.5; }
  .upload-text strong { color: var(--accent-light); }

  .preview-wrap { margin-top: 16px; display: none; }
  .preview-wrap img { width: 100%; border-radius: 10px; max-height: 220px; object-fit: cover; border: 1px solid var(--border); }

  .btn {
    width: 100%; margin-top: 16px;
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    border: none; border-radius: 10px;
    color: white; font-size: 0.9rem; font-weight: 600;
    padding: 13px; cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn:hover { opacity: 0.9; }
  .btn:active { transform: scale(0.99); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .risk-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    margin-bottom: 8px;
  }
  .risk-dot { width: 7px; height: 7px; border-radius: 50%; }
  .result-label { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
  .result-conf { font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; }
  .result-desc {
    font-size: 0.82rem; color: var(--muted); line-height: 1.6;
    padding: 10px 14px; background: var(--surface2);
    border-radius: 8px; border-left: 3px solid var(--accent);
  }

  .divider { height: 1px; background: var(--border); margin: 18px 0; }

  .bar-row { margin-bottom: 9px; }
  .bar-label { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px; }
  .bar-label span:first-child { font-weight: 500; }
  .bar-label span:last-child { color: var(--muted); }
  .bar-track { background: var(--surface2); border-radius: 6px; height: 7px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1); width: 0; }

  /* Gen AI Summary Card */
  .ai-card {
    background: linear-gradient(135deg, #1a1d27 0%, #1e1b3a 100%);
    border: 1px solid #4338ca33;
    border-radius: 16px; padding: 28px;
    display: none;
  }
  .ai-header { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
  .ai-icon { font-size: 1.2rem; }
  .ai-title { font-size: 0.85rem; font-weight: 700; color: #a5b4fc; }
  .ai-powered { font-size: 0.68rem; color: var(--muted); margin-top: 1px; }

  .ai-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  @media (max-width: 640px) { .ai-grid { grid-template-columns: 1fr; } }

  .ai-section-title {
    font-size: 0.68rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
  }
  .ai-text { font-size: 0.82rem; color: var(--text); line-height: 1.6; }
  .ai-action {
    background: var(--surface2); border-radius: 10px; padding: 14px;
    border: 1px solid var(--border);
  }

  .channel-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #4338ca22; border: 1px solid #6366f133;
    color: #a5b4fc; font-size: 0.72rem; font-weight: 600;
    padding: 5px 12px; border-radius: 20px; margin-top: 12px;
  }

  .placeholder-card {
    display: flex; align-items: center; justify-content: center;
    min-height: 180px; color: var(--muted); text-align: center;
  }

  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .footer { text-align: center; color: var(--muted); font-size: 0.72rem; margin-top: 48px; padding-bottom: 32px; line-height: 1.6; }
</style>
</head>
<body>

<header class="header">
  <div class="header-icon">👁️</div>
  <div>
    <div class="header-title">Retinal AI Classifier</div>
    <div class="header-sub">Diabetic Retinopathy Severity Detection</div>
  </div>
  <div class="header-badge">EfficientNetB3 · APTOS 2019</div>
</header>

<main class="main">

  <div class="top-grid">
    <!-- Upload -->
    <div class="card">
      <div class="card-title">Upload Retinal Image</div>
      <div class="upload-zone">
        <input type="file" id="imageInput" accept="image/*">
        <div class="upload-icon">🔬</div>
        <div class="upload-text"><strong>Click to upload</strong> or drag & drop<br>PNG, JPG, JPEG</div>
      </div>
      <div class="preview-wrap" id="previewWrap">
        <img id="preview" alt="Retinal image preview">
      </div>
      <button class="btn" id="btn" onclick="analyze()">
        <span id="btnText">Analyze Image</span>
        <div class="spinner" id="spinner" style="display:none"></div>
      </button>
    </div>

    <!-- Results -->
    <div class="card" id="resultCard">
      <div class="card-title">Classification Result</div>
      <div id="resultBody">
        <div class="placeholder-card">
          <div>
            <div style="font-size:1.8rem;margin-bottom:8px">📊</div>
            <div style="font-size:0.82rem">Results will appear here</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Gen AI Summary -->
  <div class="ai-card" id="aiCard">
    <div class="ai-header">
      <div class="ai-icon">✦</div>
      <div>
        <div class="ai-title">AI-Generated Clinical Intelligence</div>
        <div class="ai-powered">Automated insight generation · For clinical decision support</div>
      </div>
    </div>
    <div class="ai-grid">
      <div class="ai-action">
        <div class="ai-section-title">Clinical Summary</div>
        <div class="ai-text" id="aiSummary"></div>
      </div>
      <div class="ai-action">
        <div class="ai-section-title">Recommended Action</div>
        <div class="ai-text" id="aiAction"></div>
      </div>
      <div class="ai-action">
        <div class="ai-section-title">HCP Engagement</div>
        <div class="ai-text" id="aiHcp"></div>
        <div class="channel-badge">⚡ <span id="aiChannel"></span></div>
      </div>
    </div>
  </div>

</main>

<div class="footer">
  Built on EfficientNetB3 · Trained on APTOS 2019 Blindness Detection Dataset<br>
  For demonstration and educational purposes only · Not for clinical use
</div>

<script>
const LABELS = ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"];
const TIERS = ["MONITOR","MONITOR","ENGAGE","ACT NOW","ACT NOW"];
const TIER_COLORS = ["#10b981","#10b981","#f59e0b","#ef4444","#ef4444"];
const BAR_COLORS = ["#10b981","#06b6d4","#f59e0b","#f97316","#ef4444"];

document.getElementById('imageInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('preview').src = ev.target.result;
    document.getElementById('previewWrap').style.display = 'block';
  };
  reader.readAsDataURL(file);
});

async function analyze() {
  const input = document.getElementById('imageInput');
  if (!input.files[0]) { alert('Please select an image first.'); return; }

  const btn = document.getElementById('btn');
  btn.disabled = true;
  document.getElementById('btnText').style.display = 'none';
  document.getElementById('spinner').style.display = 'block';

  const formData = new FormData();
  formData.append('file', input.files[0]);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    renderResult(data);
  } catch(e) {
    alert('Error analyzing image. Please try again.');
  }

  btn.disabled = false;
  document.getElementById('btnText').style.display = 'inline';
  document.getElementById('spinner').style.display = 'none';
}

function renderResult(data) {
  const i = data.predicted_class;
  const color = TIER_COLORS[i];

  document.getElementById('resultBody').innerHTML = `
    <div class="risk-badge" style="background:${color}18;border:1px solid ${color}44">
      <div class="risk-dot" style="background:${color}"></div>
      <span style="color:${color}">${TIERS[i]}</span>
    </div>
    <div class="result-label">${LABELS[i]}</div>
    <div class="result-conf">${(data.confidence*100).toFixed(1)}% confidence</div>
    <div class="result-desc">${data.description}</div>
    <div class="divider"></div>
    <div class="card-title">Confidence Scores</div>
    ${data.probabilities.map((p,j) => `
      <div class="bar-row">
        <div class="bar-label"><span>${LABELS[j]}</span><span>${(p*100).toFixed(1)}%</span></div>
        <div class="bar-track"><div class="bar-fill" id="bar${j}" style="background:${BAR_COLORS[j]}"></div></div>
      </div>`).join('')}`;

  // Animate bars after render
  setTimeout(() => {
    data.probabilities.forEach((p, j) => {
      const el = document.getElementById('bar'+j);
      if (el) el.style.width = (p*100).toFixed(1)+'%';
    });
  }, 50);

  // Gen AI section
  document.getElementById('aiSummary').textContent = data.ai_summary;
  document.getElementById('aiAction').textContent = data.ai_action;
  document.getElementById('aiHcp').textContent = data.ai_hcp;
  document.getElementById('aiChannel').textContent = data.ai_channel;
  document.getElementById('aiCard').style.display = 'block';

  window.scrollTo({ top: 0, behavior: 'smooth' });
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
    img_array = np.expand_dims(img_array, axis=0)
    probs = model.predict(img_array, verbose=0)[0]
    predicted_class = int(np.argmax(probs))
    summary = AI_SUMMARIES[predicted_class]
    return jsonify({
        'predicted_class': predicted_class,
        'label': CLASS_LABELS[predicted_class],
        'confidence': float(probs[predicted_class]),
        'probabilities': [float(p) for p in probs],
        'description': CLINICAL_DESC[predicted_class],
        'ai_summary': summary['summary'],
        'ai_action': summary['action'],
        'ai_hcp': summary['hcp'],
        'ai_channel': summary['channel']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
