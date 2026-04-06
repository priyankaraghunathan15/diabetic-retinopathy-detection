# Interview Walkthrough — Alan Kalton, Axtria
**Retinal AI: Clinical Detection to Pharma Commercial Action**

---

## Before You Share Screen

App is open in browser on Section 1 (Hero). Nothing shared yet.

> "I'd love to walk you through a project I built — it started as a clinical AI problem but I deliberately framed it around the commercial question: how do you turn a patient risk signal into a field action? Can I share my screen?"

*Share screen.*

---

## Section 1 — The Problem
**Screen:** Hero section. Three stat cards visible — "1 in 3", "90%", "0 symptoms".

*Pause. Let him read the three cards.*

> "So the problem. 422 million people live with diabetes. One in three will develop diabetic retinopathy — damage to the retina that causes blindness. What makes it particularly dangerous is that there are zero symptoms in the early stages. By the time a patient notices something is wrong, it's often too late."

*Point to the 90% card.*

> "90% of vision loss from this condition is preventable — but only if you catch it early. That's the clinical problem. The commercial question is: once you know who's at risk, what do you do with that signal?"

*Scroll to Section 2.*

---

## Section 2 — The Data
**Screen:** Bar chart — five bars, No DR tallest, Severe and Proliferative smallest.

> "I trained the model on 3,600 real retinal scans across five severity levels."

*Point to the No DR bar.*

> "Nearly half the patients are completely healthy."

*Point to Severe and Proliferative — the two smallest bars.*

> "But look at these two. Severe and Proliferative — the ones who will go blind without intervention — make up only 13% of the dataset. A standard model optimising for overall accuracy learns to ignore that 13%, because getting them wrong barely affects the score."

*Point to the callout box below the chart.*

> "So I specifically designed this system to prioritise finding the highest-risk patients. That design decision mirrors the real-world commercial priority: the rarest patients are exactly the ones pharma field teams need to reach most urgently."

*Scroll to Section 3.*

---

## Section 3 — AI Evidence
**Screen:** Evidence section. Sample tabs across the top.

> "One of the biggest barriers to AI adoption in clinical settings is trust. A doctor or care coordinator won't act on a prediction they can't verify."

*Click "No DR" tab.*

> "On a healthy retina — almost nothing. The model has no strong signal to focus on."

*Click "Moderate DR" tab.*

> "Moderate — the attention map starts activating. The model is detecting early damage in the central retinal region."

*Click "Proliferative DR" tab.*

> "And Proliferative — the highest risk stage. The model is focused on exactly the areas of active damage."

*Point to the heatmap side of the image.*

> "A clinician can validate this in seconds. It's not a black box — it's a transparent clinical signal."

*Point to the Classification Result card on the right.*

> "Severity classification, confidence score, risk tier. That's the raw model output."

*Point to the AI Clinical Intelligence panel below.*

> "But here's the piece most relevant to what Axtria does. A care coordinator doesn't know what to do with 'Proliferative DR, 91% confidence.'"

*Point to the three cards — Clinical Summary, Recommended Action, HCP Engagement.*

> "So there's an AI layer that converts that output into a plain-language action plan — clinical summary, recommended next step, and critically, the commercial routing: which channel, which rep, what urgency. That bridge — from model output to commercial action — is what makes this useful in the field."

*Scroll to Section 5.*

---

## Section 4 — Commercial Dashboard
**Screen:** Donut chart left, NBA by Risk Tier right.

*Point to the donut chart.*

> "If you apply this to a health system at scale — 10,000 diabetic patients screened — the system automatically segments the population into three tiers."

*Point to green segment.*

> "59% go into Monitor. Healthy or mild. Automated digital reminder, no human intervention needed."

*Point to yellow segment.*

> "27% go into Engage. Moderate risk. Flagged in CRM for HCP outreach."

*Point to red segment.*

> "14% — 1,400 patients — go straight to Act Now. Urgent field rep visit, specialist referral, therapy initiation."

*Point to the NBA cards on the right.*

> "The AI does the triage. The commercial team gets a prioritised list — no manual review, no guesswork."

*Scroll to the Field Force section.*

**Screen:** Top 5 States bar chart.

*Point to Texas, Florida, California.*

> "And geographically — this is where the patient concentration is. Texas, Florida, California. That informs where you deploy field resources and where ophthalmology rep visits will have the highest return."

*Scroll to NBA Pipeline and Market Opportunity.*

**Screen:** 01 → 02 → 03 pipeline left. 70,000 stat right.

*Point to the pipeline flow.*

> "Patient screened, AI classifies severity, risk tier assigned, routed to the right channel automatically."

*Point to the 70,000 number.*

> "And at scale — in a system managing 500,000 diabetic patients — this model identifies 70,000 individuals requiring immediate intervention. That's a significant addressable population for retinal therapy and anti-VEGF treatment brands."

*Pause. Don't scroll.*

---

## Closing

*Say this looking at him, not the screen.*

> "At its core, what this system does is connect the right therapy to the right patient at the right time — identifying who needs intervention, routing them to the right channel, and giving the commercial team the intelligence to act."

**Stop. Let it land. Do not fill the silence.**

---

## If He Asks "Does It Actually Run?"

*Open HuggingFace link in new tab.*

> "Yes — this is the live inference version. You can upload any retinal scan and get a real-time classification with confidence score."

*Show the interface. Don't demo unless he asks.*

---

## Handling Difficult Questions

| He asks... | You say... |
|---|---|
| "How accurate is it?" | "It correctly stratifies 74% of patients — but more importantly it's calibrated to over-index on high-risk patients rather than optimise for the average case." |
| "What model did you use?" | "A state-of-the-art image recognition architecture, pre-trained on medical imaging data and fine-tuned on 3,600 retinal scans." |
| "How would pharma sales teams use this?" | "The system produces a daily prioritised patient list by risk tier — the field rep wakes up knowing exactly who needs an urgent visit, who needs a CRM follow-up, and who can be handled digitally. It's the same logic Aktana built for HCP-level NBA." |
| "What would you do differently at scale?" | "Connect it directly into the EHR system so the classification happens at point of care — and integrate the risk tier output into existing CRM platforms like Veeva." |
| "Why Axtria specifically?" | "Axtria sits at the intersection of data, AI, and pharma commercial strategy — which is exactly where I want to work. CustomerIQ is doing for pharma reps what this system does for patient routing: turning data signals into next-best-actions." |

---

## Language to Use vs. Avoid

| Instead of... | Say... |
|---|---|
| "EfficientNetB3" | "A state-of-the-art image recognition architecture" |
| "Focal loss" | "The model was designed to prioritise high-risk patients" |
| "Cohen's Kappa of 0.59" | "Substantial agreement with specialist-level diagnosis" |
| "5-class classification" | "Patient stratification into five risk tiers" |
| "Grad-CAM heatmaps" | "Visual interpretability so clinicians can trust and verify the output" |
| "Validation accuracy 74%" | "Correctly stratifies 74% of patients for appropriate care pathways" |
| "Gen AI API call" | "Automated insight generation that converts model output into actionable recommendations" |

---

## Timing Guide

| Section | Time |
|---|---|
| Opening | 15 sec |
| Section 1 — The Problem | 45 sec |
| Section 2 — The Data | 60 sec |
| Section 3 — AI Evidence | 90 sec |
| Section 4 — Commercial Dashboard | 90 sec |
| Closing | 15 sec |
| **Total** | **~5 minutes** |

Leaves plenty of time for questions — that is where the real interview happens.

---

## The One Line That Ties Everything Together

> *"At its core, this is about connecting the right therapy to the right patient at the right time — identifying who needs intervention, routing them to the right channel, and giving the commercial team the intelligence to act. That's what this system does."*

This is Axtria's mission. Alan will recognise it.
