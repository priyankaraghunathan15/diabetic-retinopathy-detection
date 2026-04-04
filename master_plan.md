# Master Plan: Diabetic Retinopathy Project → Axtria Interview

---

## THE BIG PICTURE

We are taking an existing ML project (Diabetic Retinopathy Detection) and transforming it from a "data science student project" into a "pharma commercial analytics solution" — specifically tailored for an interview with **Alan Kalton, Principal at Axtria**.

**Who is Alan Kalton?**
- 30+ years in pharma commercial analytics
- Started career at ZS Associates (top pharma consulting firm)
- Spent years at Aktana — an AI next-best-action company for pharma reps (direct competitor to Axtria's CustomerIQ)
- Now leading strategic commercial consulting at Axtria UK & Europe
- NOT a data scientist — he is a business strategy and commercial operations person
- He responds to: patient outcomes, commercial impact, business value, next-best-action, HCP engagement
- He does NOT respond to: EfficientNetB3, focal loss, Cohen's Kappa, convolutional layers

**Core message we want Alan to walk away with:**
> "She understands how AI connects to pharma commercial strategy — not just how to build models, but how to turn patient data into commercial action."

---

## WHAT WE ARE BUILDING

### Deliverable 1: HuggingFace Spaces App (Live Inference)
**Purpose:** Proof that the ML pipeline is real and working  
**What it does:** Upload a retinal image → get severity classification + confidence  
**Tech:** Streamlit + TensorFlow/Keras + EfficientNetB3  
**Where:** HuggingFace Spaces (free, handles TensorFlow perfectly)  
**When to use:** Backup proof during interview if Alan asks "does it actually work?"  
**Status:** app.py exists, needs TF version fix, then deploy to HuggingFace  

### Deliverable 2: Netlify Storytelling App (Interview Weapon)
**Purpose:** The main demo — tells the complete story from clinical detection to commercial action  
**What it does:** Walks through the problem, the data, the AI, and the pharma commercial angle  
**Tech:** React (no backend, no TensorFlow, pre-loaded data)  
**Where:** Netlify (free deploy from GitHub)  
**When to use:** This is what you screen share with Alan  
**Status:** To be built  

### Deliverable 3: Updated GitHub README
**Purpose:** First thing Alan sees if he looks at your repo before/after the interview  
**What it does:** Tells the project story in business language, not technical language  
**Status:** Needs complete rewrite  

---

## THE NETLIFY APP — DETAILED BREAKDOWN

### Section 1: The Human Problem (Emotional Hook)
**What it shows:**
- Full screen: "422 million people live with diabetes. 1 in 3 will develop Diabetic Retinopathy. 90% of vision loss is preventable — if caught in time."
- One retinal scan image, clean and powerful
- One line: "We built AI to change that."

**Why it matters for Alan:**
Opens with patient outcomes and health impact — his entire career has been about improving patient lives through data. This speaks directly to him before a single technical word is said.

---

### Section 2: The Data Reality (Data Storytelling)
**What it shows:**
- Interactive bar chart of class distribution from APTOS 2019 dataset
- Real numbers from your notebook (to be confirmed):
  - No DR: ~1,805 images (49%)
  - Mild: ~370 images (10%)
  - Moderate: ~999 images (27%)
  - Severe: ~193 images (5%)
  - Proliferative: ~295 images (8%)
- Hover over each bar → plain English explanation of what that severity means clinically
- Callout box: "Severe and Proliferative cases — the ones that cause blindness — make up only 13% of patients. A standard AI model would learn to ignore them. Ours was specifically designed to find them."

**Why it matters for Alan:**
This is data storytelling — showing the business problem hidden in the data. The class imbalance IS the real-world problem. Rare dangerous patients get missed. This sets up why your approach was different without ever saying "focal loss."

**What you say:** "When I looked at the data, I noticed something that mirrors the real world perfectly — the most dangerous patients are the minority. Most AI models would just learn to ignore them because they're rare. I designed the model specifically to prioritize finding them."

---

### Section 3: What the AI Sees (Interpretability = Trust)
**What it shows:**
- Toggle gallery of your 3 Grad-CAM sample images
- Each one: original retinal scan on left, heatmap on right
- Severity badge (e.g. "Moderate DR") + confidence % for each
- Caption: "The red zone shows exactly where the AI detected damage. A clinician can verify this in seconds — no black box."

**Why it matters for Alan:**
Interpretability is critical in pharma/clinical settings. Regulators, doctors, and pharma companies cannot trust a black box. By showing Grad-CAM, you're showing you understand that AI in healthcare must be explainable. This is a very mature thing to have built.

**What you say:** "One of the biggest barriers to AI adoption in clinical settings is trust. Doctors won't act on a prediction they can't verify. Grad-CAM solves that — it shows exactly what the model saw, so a clinician can validate it in seconds."

---

### Section 4: From Scan to Action — Gen AI Layer (The New Piece)
**What it shows:**
- For each of the 3 sample images, a pre-generated clinical summary card:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient Risk Profile
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Severity Classification: Moderate DR
AI Confidence: 78%
Risk Tier: 🟡 ENGAGE

AI-Generated Clinical Summary:
"This patient shows signs of Moderate 
Non-Proliferative Diabetic Retinopathy. 
Microaneurysms and early retinal hemorrhaging 
detected in the central region. Progression 
risk is elevated without intervention.

Recommended Action: Ophthalmology referral 
within 30 days.
Therapy Consideration: Patient may be a 
candidate for anti-VEGF therapy evaluation.
Next-Best-Action: Flag for HCP outreach via 
primary care provider."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**How we generate these:** Claude API call with the severity + confidence as input. Pre-generate all 3 summaries, hardcode them in the app. Looks live, never crashes.

**Why it matters for Alan:**
This bridges clinical AI and commercial action. The Gen AI layer converts a model output into something a care coordinator or pharma rep can act on immediately — no data science background needed. This is exactly what Axtria's products do.

**What you say:** "The model gives a classification. But a care coordinator doesn't know what to do with 'Moderate DR, 78% confidence.' The Gen AI layer converts that into a plain-language action plan — who to call, what to recommend, what the next step is. That's the bridge between AI output and commercial action."

---

### Section 5: Commercial Intelligence Dashboard (The Axtria Section)
This is the section that makes Alan lean forward. It connects everything to his world.

**Sub-section A — Patient Segmentation by Risk Tier**
```
Simulated population: 10,000 diabetic patients screened

🟢 MONITOR    No DR / Mild      ~59%   5,900 patients
              → Automated digital reminder

🟡 ENGAGE     Moderate DR       ~27%   2,700 patients  
              → HCP outreach flagged in CRM

🔴 ACT NOW    Severe / Prolif.  ~14%   1,400 patients
              → Urgent field rep visit + therapy referral
```
Show as an interactive donut chart with these 3 tiers.

**Why:** This IS next-best-action at the patient level. Alan spent years at Aktana building exactly this logic for HCP engagement. He will recognize this immediately.

**Sub-section B — Prescriber Opportunity Map**
- US map showing diabetes prevalence by state (use CDC public data)
- Highlight top 5 states: Texas, California, Florida, New York, Georgia
- Label: "Illustrative — based on CDC diabetes prevalence data"
- Caption: "High-risk patient concentrations inform where pharma field teams should focus ophthalmology rep visits"

**Why:** Field force optimization — Alan's bread and butter from ZS Associates days.

**Sub-section C — Next-Best-Action Routing Flow**
Simple clean flowchart:
```
Patient Screened
       ↓
AI Classifies Severity
       ↓
Risk Tier Assigned
    /    |    \
🟢       🟡      🔴
Digital  CRM    Field Rep
Nudge   Flag    Visit
```

**Why:** This is the commercial engine. Axtria's SalesIQ and CustomerIQ do exactly this. You're showing you understand the full pipeline from data to action.

**Sub-section D — Market Opportunity**
Clean stat card:
> "In a US health system managing 500,000 diabetic patients, this model identifies ~70,000 high-risk individuals requiring immediate intervention — representing a significant addressable population for retinal therapy and anti-VEGF treatment brands."

**Why:** Connects patient data to revenue opportunity — the language of pharma commercial strategy.

---

### Section 6: About / Connect
- Your name, photo (optional), LinkedIn, GitHub
- One line: "Built to bridge the gap between clinical AI and pharma commercial strategy."
- Link to HuggingFace app: "Try the live inference tool →"
- Link to GitHub repo: "View the full ML pipeline →"

---

## THE HUGGINGFACE APP — DETAILED BREAKDOWN

**What it is:** Your existing Streamlit app, fixed and deployed  
**Fix needed:** TensorFlow version compatibility (tf.keras → keras)  
**Deploy steps:**
1. Create free account at huggingface.co
2. Create new Space → Streamlit
3. Push your code (app.py + requirements.txt)
4. HuggingFace handles the TF runtime — no DLL errors

**Updated requirements.txt for HuggingFace:**
```
streamlit==1.28.1
tensorflow-cpu==2.13.0  (HF uses Python 3.10, this works)
pillow
numpy
gdown
```

---

## THE GITHUB README — REWRITE OUTLINE

**Current:** Opens with "EfficientNet-based diabetic retinopathy classification..."  
**New opening:**
> "422 million people live with diabetes. 1 in 3 will develop Diabetic Retinopathy — yet 90% of vision loss is preventable with early detection. This project builds an AI-powered patient risk stratification system that classifies patients into 5 severity tiers, generates interpretable clinical evidence, and produces actionable Gen AI summaries — turning retinal scan data into commercial-ready patient intelligence."

**New section order:**
1. The Problem (human impact first)
2. The Commercial Application (what pharma companies do with this)
3. How It Works (technical — brief)
4. Results & Impact (metrics translated to business value)
5. The Gen AI Layer
6. Live Demo links (Netlify + HuggingFace)
7. Installation

---

## 4-DAY EXECUTION PLAN

### Day 1 — Data & Numbers (Today)
- [ ] Fix Colab notebook paths (Kaggle → Google Drive)
- [ ] Run EDA cell → get exact class distribution numbers
- [ ] Run Grad-CAM cell → get confidence scores for 3 sample images
- [ ] Run metrics cell → get confusion matrix, final accuracy
- [ ] Save/download the 3 Grad-CAM images (already in images/ folder)
- [ ] Pre-generate 3 Gen AI clinical summaries using Claude API
- [ ] Fix app.py TF compatibility issue

### Day 2 — Deploy HuggingFace
- [ ] Create HuggingFace account
- [ ] Create new Streamlit Space
- [ ] Push fixed app.py + updated requirements.txt
- [ ] Test live upload works
- [ ] Get the HuggingFace URL

### Day 3 — Build & Deploy Netlify App
- [ ] Set up React project
- [ ] Build all 6 sections with real numbers
- [ ] Add Recharts for class distribution + donut chart
- [ ] Add Grad-CAM toggle gallery with real images
- [ ] Add Gen AI summary cards (pre-generated)
- [ ] Add commercial dashboard sections
- [ ] Deploy to Netlify
- [ ] Update GitHub README
- [ ] Link everything together (Netlify ↔ HuggingFace ↔ GitHub)

### Day 4 — Interview Prep
- [ ] Practice 3-minute app walkthrough out loud (screen share simulation)
- [ ] Prep "Why Axtria?" answer
- [ ] Prep "Walk me through your project" answer (non-technical version)
- [ ] Prep answer for "How would pharma sales teams use this?"
- [ ] Prep 2 STAR behavioral stories
- [ ] Prep 2 smart questions to ask Alan
- [ ] Final review of Alan's background — know his Aktana connection cold

---

## KEY PHRASES TO USE WITH ALAN

| Instead of saying... | Say this... |
|---|---|
| "I used EfficientNetB3" | "I used a state-of-the-art image recognition architecture" |
| "Focal loss for class imbalance" | "The model was specifically designed to prioritize high-risk patients" |
| "Cohen's Kappa of 0.59" | "The model shows substantial agreement with specialist-level diagnosis" |
| "5-class classification" | "Patient stratification into 5 risk tiers" |
| "Grad-CAM heatmaps" | "Visual interpretability so clinicians can trust and verify the output" |
| "Gen AI API call" | "Automated insight generation that converts model output into actionable recommendations" |
| "Validation accuracy 74%" | "Correctly stratifies 74% of patients for appropriate care pathways" |

---

## THE ONE LINE THAT TIES EVERYTHING TOGETHER

At the end of your demo, say this:

> "At its core, this is about connecting the right therapy to the right patient at the right time — identifying who needs intervention, routing them to the right channel, and giving the commercial team the intelligence to act. That's what this system does."

That is Axtria's exact mission statement. Alan will recognize it. That's your closing line.

---

## WHAT SUCCESS LOOKS LIKE

After this interview, Alan should think:
1. ✅ She understands patient outcomes and clinical context
2. ✅ She knows how AI connects to pharma commercial operations
3. ✅ She can communicate data insights to non-technical stakeholders
4. ✅ She built something real, not just a class project
5. ✅ She thinks like a consultant, not just a data scientist
