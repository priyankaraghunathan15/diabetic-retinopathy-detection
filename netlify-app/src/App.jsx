import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  PieChart, Pie, ResponsiveContainer,
} from 'recharts'

/* ── DATA ──────────────────────────────────────────────────────────── */

const CLASS_DIST = [
  { name: 'No DR',         count: 1805, pct: 49.3, color: '#10b981', tier: 'MONITOR' },
  { name: 'Mild',          count: 370,  pct: 10.1, color: '#06b6d4', tier: 'MONITOR' },
  { name: 'Moderate',      count: 999,  pct: 27.3, color: '#f59e0b', tier: 'ENGAGE'  },
  { name: 'Severe',        count: 193,  pct: 5.3,  color: '#f97316', tier: 'ACT NOW' },
  { name: 'Proliferative', count: 295,  pct: 8.1,  color: '#ef4444', tier: 'ACT NOW' },
]

const RISK_DATA = [
  { name: 'MONITOR',  patients: 5900, pct: '59%', color: '#10b981', label: 'No DR / Mild',          action: 'Automated digital reminder' },
  { name: 'ENGAGE',   patients: 2700, pct: '27%', color: '#f59e0b', label: 'Moderate DR',            action: 'HCP outreach flagged in CRM' },
  { name: 'ACT NOW',  patients: 1400, pct: '14%', color: '#ef4444', label: 'Severe / Proliferative', action: 'Urgent field rep visit + therapy referral' },
]

const SAMPLES = [
  {
    label: 'No DR',
    img: '/images/gradcam_nodr.png',
    severity: 'No DR',
    confidence: 94.3,
    tier: 'MONITOR',
    color: '#10b981',
    summary: 'Retinal imaging shows no evidence of diabetic retinopathy. Vascular architecture appears intact with no microaneurysms, exudates, or neovascularization detected.',
    action: 'Continue standard diabetes management. Schedule next retinal screening in 12 months.',
    hcp: 'Automated digital reminder via patient portal. No urgent HCP outreach required.',
    channel: 'Digital',
  },
  {
    label: 'Mild DR',
    img: '/images/gradcam_mild.png',
    severity: 'Mild DR',
    confidence: 78.6,
    tier: 'MONITOR',
    color: '#06b6d4',
    summary: 'Early-stage diabetic retinopathy detected. Small number of microaneurysms present in the peripheral retina. No vision-threatening features at this stage.',
    action: 'Increase screening frequency to every 6 months. Reinforce glycaemic control with primary care team.',
    hcp: 'Automated digital reminder with lifestyle guidance. Flag for next routine HCP visit.',
    channel: 'Digital + Primary Care Flag',
  },
  {
    label: 'Moderate DR',
    img: '/images/gradcam_moderate.png',
    severity: 'Moderate DR',
    confidence: 81.2,
    tier: 'ENGAGE',
    color: '#f59e0b',
    summary: 'Moderate non-proliferative diabetic retinopathy confirmed. Microaneurysms and early retinal haemorrhaging detected in the central region. Progression risk is elevated without intervention.',
    action: 'Ophthalmology referral within 30 days. Patient may be a candidate for anti-VEGF therapy evaluation.',
    hcp: 'Flag for HCP outreach via primary care provider. CRM alert for ophthalmology rep engagement.',
    channel: 'CRM Flag + HCP Outreach',
  },
  {
    label: 'Severe DR',
    img: '/images/gradcam_severe.png',
    severity: 'Severe DR',
    confidence: 87.4,
    tier: 'ACT NOW',
    color: '#f97316',
    summary: 'Severe non-proliferative diabetic retinopathy detected. Extensive retinal haemorrhages and venous beading observed across multiple quadrants. High risk of conversion to proliferative stage.',
    action: 'Urgent ophthalmology referral within 1 week. Laser photocoagulation or anti-VEGF therapy initiation required.',
    hcp: 'Priority field rep visit. Immediate specialist coordination. Time-sensitive therapy window.',
    channel: 'Urgent Field Rep + Specialist',
  },
  {
    label: 'Proliferative DR',
    img: '/images/gradcam_proliferative.png',
    severity: 'Proliferative DR',
    confidence: 91.8,
    tier: 'ACT NOW',
    color: '#ef4444',
    summary: 'Proliferative diabetic retinopathy confirmed. Active neovascularization and vitreous haemorrhage signs detected. Highest-risk stage with imminent threat to vision.',
    action: 'Same-week specialist intervention required. Anti-VEGF therapy or surgical evaluation indicated.',
    hcp: 'Urgent field rep visit. Flag for immediate specialist coordination. Critical therapy initiation opportunity.',
    channel: 'Urgent Field Rep + Specialist',
  },
]

const TOP_STATES = [
  { state: 'California', patients: '3.0M', pct: 10.0, bar: 100 },
  { state: 'Texas',      patients: '2.8M', pct: 12.3, bar: 91  },
  { state: 'Florida',    patients: '2.1M', pct: 11.6, bar: 68  },
  { state: 'New York',   patients: '1.7M', pct: 9.3,  bar: 55  },
  { state: 'Georgia',    patients: '1.1M', pct: 11.4, bar: 36  },
]

/* ── TOKENS ────────────────────────────────────────────────────────── */

const T = {
  bg:       '#080b12',
  surface:  '#0f1320',
  surface2: '#161c2e',
  border:   '#242d4a',
  text:     '#eef0f8',
  muted:    '#7b82a0',
  accent:   '#6366f1',
  accent2:  '#818cf8',
}

/* ── SHARED COMPONENTS ─────────────────────────────────────────────── */

function Label({ children }) {
  return (
    <div style={{
      fontSize: '0.63rem', fontWeight: 700, color: T.accent2,
      textTransform: 'uppercase', letterSpacing: '0.18em', marginBottom: 14,
    }}>{children}</div>
  )
}

function Title({ children, style }) {
  return (
    <h2 style={{
      fontSize: 'clamp(1.7rem, 4vw, 2.6rem)',
      fontWeight: 800, letterSpacing: '-0.03em',
      lineHeight: 1.12, marginBottom: 18, ...style,
    }}>{children}</h2>
  )
}

function Card({ children, style }) {
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 16, padding: 24, ...style,
    }}>{children}</div>
  )
}

function CardLabel({ children }) {
  return (
    <div style={{
      fontSize: '0.63rem', fontWeight: 700, color: T.muted,
      textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16,
    }}>{children}</div>
  )
}

function BarTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={{
      background: T.surface2, border: `1px solid ${T.border}`,
      borderRadius: 10, padding: '10px 14px', fontSize: '0.78rem',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.name}</div>
      <div style={{ color: T.muted }}>{d.count.toLocaleString()} images — {d.pct}%</div>
      <div style={{
        marginTop: 6, display: 'inline-block',
        background: d.color + '22', border: `1px solid ${d.color}44`,
        color: d.color, fontSize: '0.63rem', fontWeight: 700,
        padding: '2px 8px', borderRadius: 20,
      }}>{d.tier}</div>
    </div>
  )
}

function PieTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={{
      background: T.surface2, border: `1px solid ${T.border}`,
      borderRadius: 10, padding: '10px 14px', fontSize: '0.78rem', maxWidth: 200,
    }}>
      <div style={{ fontWeight: 700, color: d.color, marginBottom: 4 }}>{d.name}</div>
      <div style={{ color: T.muted, marginBottom: 4 }}>{d.patients.toLocaleString()} patients ({d.pct})</div>
      <div style={{ color: T.text }}>{d.action}</div>
    </div>
  )
}

/* ── NAVBAR ────────────────────────────────────────────────────────── */

function Navbar() {
  const nav = [
    ['Problem',    '#problem'],
    ['Data',       '#data'],
    ['AI Evidence','#evidence'],
    ['Commercial', '#commercial'],
    ['About',      '#about'],
  ]
  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 200,
      background: 'rgba(8,11,18,0.88)', backdropFilter: 'blur(14px)',
      borderBottom: `1px solid ${T.border}`,
      height: 60, display: 'flex', alignItems: 'center',
      padding: '0 32px', gap: 6,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginRight: 'auto' }}>
        <div style={{
          width: 30, height: 30, borderRadius: 7,
          background: 'linear-gradient(135deg, #6366f1, #4338ca)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '0.8rem', flexShrink: 0, fontWeight: 800, color: '#a5b4fc',
        }}>RI</div>
        <span style={{ fontWeight: 800, fontSize: '0.9rem', letterSpacing: '-0.01em' }}>Retinal AI</span>
        <span style={{
          fontSize: '0.6rem', fontWeight: 600, color: '#a5b4fc',
          background: '#6366f122', border: '1px solid #6366f133',
          padding: '2px 8px', borderRadius: 20, marginLeft: 4,
        }}>Clinical Intelligence Platform</span>
      </div>
      <div className="nav-links" style={{ display: 'flex', gap: 2 }}>
        {nav.map(([label, href]) => (
          <a key={label} href={href} style={{
            color: T.muted, fontSize: '0.75rem', fontWeight: 500,
            textDecoration: 'none', padding: '5px 12px', borderRadius: 7,
            transition: 'all 0.15s',
          }}
          onMouseEnter={e => { e.target.style.color = T.text; e.target.style.background = T.surface2 }}
          onMouseLeave={e => { e.target.style.color = T.muted; e.target.style.background = 'transparent' }}
          >{label}</a>
        ))}
      </div>
    </nav>
  )
}

/* ── SECTION 1: HERO ───────────────────────────────────────────────── */

function Hero() {
  return (
    <section id="problem" style={{
      minHeight: '100vh',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      textAlign: 'center', padding: '100px 24px 80px',
      background: 'radial-gradient(ellipse 80% 60% at 50% 30%, #1e1b4b 0%, #080b12 70%)',
    }}>
      <div style={{
        fontSize: '0.6rem', fontWeight: 700, color: '#a5b4fc',
        textTransform: 'uppercase', letterSpacing: '0.2em',
        background: '#6366f118', border: '1px solid #6366f133',
        padding: '5px 18px', borderRadius: 20, marginBottom: 32,
      }}>
        Clinical AI · Patient Risk Stratification · Pharma Commercial Intelligence
      </div>

      <h1 style={{
        fontSize: 'clamp(2.4rem, 8vw, 5rem)',
        fontWeight: 900, letterSpacing: '-0.04em',
        lineHeight: 1.08, maxWidth: 860, marginBottom: 32,
      }}>
        589 million people<br />
        <span style={{ color: T.accent2 }}>live with diabetes.</span>
      </h1>

      <div className="hero-stats" style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 20, maxWidth: 720, width: '100%', marginBottom: 44,
      }}>
        {[
          { stat: '1 in 3', desc: 'will develop Diabetic Retinopathy' },
          { stat: '90%', desc: 'of vision loss is preventable with early detection' },
          { stat: '0', desc: 'symptoms in early stages — it\'s completely silent' },
        ].map(({ stat, desc }) => (
          <div key={stat} style={{
            background: 'rgba(15,19,32,0.7)', border: `1px solid ${T.border}`,
            borderRadius: 16, padding: '28px 20px', backdropFilter: 'blur(8px)',
          }}>
            <div style={{
              fontSize: 'clamp(2rem, 5vw, 3rem)', fontWeight: 900,
              color: T.accent2, marginBottom: 10, letterSpacing: '-0.03em',
            }}>{stat}</div>
            <div style={{ fontSize: '0.8rem', color: T.muted, lineHeight: 1.55 }}>{desc}</div>
          </div>
        ))}
      </div>

      <p style={{
        fontSize: 'clamp(0.95rem, 2.5vw, 1.2rem)',
        color: T.muted, maxWidth: 600, lineHeight: 1.75, marginBottom: 44,
      }}>
        I built an AI system that detects diabetic retinopathy from retinal scans —
        and turns clinical findings into{' '}
        <span style={{ color: T.text, fontWeight: 600 }}>commercial-ready patient intelligence.</span>
      </p>

      <a href="#data" style={{
        display: 'inline-flex', alignItems: 'center', gap: 10,
        background: 'linear-gradient(135deg, #6366f1, #4338ca)',
        color: 'white', fontWeight: 700, fontSize: '0.9rem',
        padding: '15px 32px', borderRadius: 12, textDecoration: 'none',
        boxShadow: '0 6px 24px #6366f130', transition: 'transform 0.2s',
        fontFamily: 'inherit',
      }}
      onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
      onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
      >See the full story ↓</a>
    </section>
  )
}

/* ── SECTION 2: DATA ───────────────────────────────────────────────── */

function DataSection() {
  return (
    <section id="data" style={{ padding: '100px 24px', maxWidth: 1000, margin: '0 auto' }}>
      <Label>The Data Reality</Label>
      <Title>The Hidden Problem<br />in the Data</Title>
      <p style={{
        fontSize: '0.92rem', color: T.muted,
        lineHeight: 1.75, maxWidth: 680, marginBottom: 52,
      }}>
        The APTOS 2019 dataset mirrors the real world perfectly, most patients are healthy.
        But the patients who need urgent intervention are the minority.
        A standard model would learn to ignore them.
      </p>

      <div style={{ height: 320, marginBottom: 36 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={CLASS_DIST} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" tick={{ fill: T.muted, fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip content={<BarTooltip />} cursor={{ fill: '#ffffff06' }} />
            <Bar dataKey="count" radius={[6, 6, 0, 0]}>
              {CLASS_DIST.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{
        background: '#1a1b3a', border: '1px solid #4338ca33',
        borderRadius: 14, padding: '20px 24px',
        display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 44,
      }}>
        <div style={{
          width: 22, height: 22, borderRadius: 4, flexShrink: 0, marginTop: 2,
          background: '#4338ca44', border: '1px solid #6366f155',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '0.72rem', fontWeight: 900, color: '#a5b4fc',
        }}>!</div>
        <div>
          <div style={{ fontWeight: 700, color: '#c7d2fe', marginBottom: 7, fontSize: '0.9rem' }}>
            The Real-World Problem Hidden in This Chart
          </div>
          <div style={{ fontSize: '0.84rem', color: '#a5b4fc', lineHeight: 1.72 }}>
            Severe and Proliferative cases — the ones that cause blindness — make up only{' '}
            <strong>13.4% of patients.</strong> A standard model optimising for overall accuracy
            learns to ignore them. This system was specifically designed to{' '}
            <strong>prioritise finding the highest-risk patients.</strong>
          </div>
        </div>
      </div>

      <div className="four-col" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
        {[
          { val: '73.4%', lbl: 'Validation Accuracy' },
          { val: '0.59',  lbl: "Cohen's Kappa" },
          { val: '3,662', lbl: 'Training Images' },
          { val: '5',     lbl: 'Severity Classes' },
        ].map(({ val, lbl }) => (
          <Card key={lbl} style={{ textAlign: 'center', padding: '20px 14px' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: T.accent2, marginBottom: 5 }}>{val}</div>
            <div style={{ fontSize: '0.62rem', color: T.muted, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>{lbl}</div>
          </Card>
        ))}
      </div>
    </section>
  )
}

/* ── SECTION 3+4: AI EVIDENCE ──────────────────────────────────────── */

function EvidenceSection() {
  const [active, setActive] = useState(4) // Default to Proliferative DR (index 4, most dramatic)
  const s = SAMPLES[active]

  return (
    <section id="evidence" style={{
      padding: '100px 24px',
      background: 'linear-gradient(180deg, #080b12 0%, #0c0a24 50%, #080b12 100%)',
    }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        <Label>AI Interpretability + Clinical Intelligence</Label>
        <Title>From Pixel<br />to Prescription</Title>
        <p style={{
          fontSize: '0.92rem', color: T.muted,
          lineHeight: 1.75, maxWidth: 700, marginBottom: 44,
        }}>
          The Grad-CAM heatmap shows exactly which part of the retina the AI focused on.
          The clinical intelligence layer converts that finding into an action plan —
          no data science background needed.
        </p>

        {/* Sample tabs */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 28, flexWrap: 'wrap' }}>
          {SAMPLES.map((sample, i) => (
            <button key={i} onClick={() => setActive(i)} style={{
              padding: '9px 20px', borderRadius: 9, cursor: 'pointer',
              fontFamily: 'inherit', fontSize: '0.78rem', fontWeight: 600,
              transition: 'all 0.2s', border: 'none',
              background: active === i ? sample.color + '20' : T.surface,
              outline: `1px solid ${active === i ? sample.color + '66' : T.border}`,
              color: active === i ? sample.color : T.muted,
            }}>{sample.label} — {sample.severity}</button>
          ))}
        </div>

        {/* Grad-CAM + Result */}
        <div className="two-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{
              padding: '13px 18px', borderBottom: `1px solid ${T.border}`,
              fontSize: '0.63rem', fontWeight: 700, color: T.muted,
              textTransform: 'uppercase', letterSpacing: '0.1em',
            }}>Grad-CAM Attention Heatmap</div>
            <img src={s.img} alt="Grad-CAM heatmap" style={{
              width: '100%', display: 'block',
              maxHeight: 270, objectFit: 'cover',
            }} />
            <div style={{
              padding: '11px 18px', fontSize: '0.72rem',
              color: T.muted, lineHeight: 1.6,
              borderTop: `1px solid ${T.border}`,
            }}>
              Red zones = highest model attention. Clinicians can verify AI reasoning in seconds.
            </div>
          </Card>

          <Card>
            <CardLabel>Classification Result</CardLabel>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 7,
              padding: '6px 14px', borderRadius: 8, marginBottom: 12,
              background: s.color + '18', border: `1px solid ${s.color}44`,
            }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: s.color }} />
              <span style={{ color: s.color, fontSize: '0.7rem', fontWeight: 800, letterSpacing: '0.08em' }}>{s.tier}</span>
            </div>
            <div style={{
              fontSize: '2rem', fontWeight: 800, color: s.color,
              marginBottom: 8, letterSpacing: '-0.02em',
            }}>{s.severity}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
              <span style={{ fontSize: '0.78rem', color: T.muted, whiteSpace: 'nowrap' }}>
                {s.confidence}% confidence
              </span>
              <div style={{ flex: 1, background: T.surface2, borderRadius: 100, height: 5, overflow: 'hidden' }}>
                <div style={{
                  width: `${s.confidence}%`, height: '100%',
                  background: s.color, borderRadius: 100,
                  transition: 'width 0.6s ease',
                }} />
              </div>
            </div>
            <div style={{
              fontSize: '0.8rem', color: T.muted, lineHeight: 1.68,
              padding: '11px 14px', background: T.surface2,
              borderRadius: 10, borderLeft: `3px solid ${s.color}`,
            }}>{s.summary}</div>
          </Card>
        </div>

        {/* AI Clinical Intelligence */}
        <div style={{
          background: 'linear-gradient(135deg, #0f1320, #1a1b3a)',
          border: '1px solid #4338ca33', borderRadius: 16, padding: '24px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
            <div style={{
              width: 34, height: 34, borderRadius: 8, flexShrink: 0,
              background: '#4338ca22', border: '1px solid #6366f133',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.7rem', fontWeight: 800, color: '#a5b4fc',
            }}>AI</div>
            <div>
              <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#c7d2fe' }}>
                AI-Generated Clinical Intelligence
              </div>
              <div style={{ fontSize: '0.68rem', color: T.muted, marginTop: 2 }}>
                Automated insight generation · Pharma commercial decision support
              </div>
            </div>
          </div>
          <div className="three-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14 }}>
            {[
              { title: 'Clinical Summary', text: s.summary },
              { title: 'Recommended Action', text: s.action },
              { title: 'HCP Engagement',   text: s.hcp, channel: s.channel },
            ].map(({ title, text, channel }) => (
              <div key={title} style={{
                background: T.surface2, borderRadius: 12,
                padding: '16px', border: `1px solid ${T.border}`,
              }}>
                <div style={{
                  fontSize: '0.62rem', fontWeight: 700, color: T.muted,
                  textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 9,
                }}>{title}</div>
                <div style={{ fontSize: '0.78rem', color: T.text, lineHeight: 1.68 }}>{text}</div>
                {channel && (
                  <div style={{
                    display: 'inline-block', marginTop: 10,
                    background: '#312e8122', border: '1px solid #6366f133',
                    color: '#a5b4fc', fontSize: '0.66rem', fontWeight: 600,
                    padding: '4px 10px', borderRadius: 20,
                  }}>{channel}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

/* ── SECTION 5: COMMERCIAL ─────────────────────────────────────────── */

function CommercialSection() {
  return (
    <section id="commercial" style={{ padding: '100px 24px', maxWidth: 1000, margin: '0 auto' }}>
      <Label>Commercial Intelligence</Label>
      <Title>Connecting AI<br />to Commercial Action</Title>
      <p style={{
        fontSize: '0.92rem', color: T.muted,
        lineHeight: 1.75, maxWidth: 700, marginBottom: 56,
      }}>
        In a simulated population of 10,000 diabetic patients screened, the system automatically
        segments each patient into an actionable risk tier — routing them to the right commercial channel.
      </p>

      {/* Risk segmentation + NBA */}
      <div className="two-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        <Card>
          <CardLabel>Patient Risk Segmentation — 10,000 Patients</CardLabel>
          <div style={{ height: 210 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={RISK_DATA} dataKey="patients" nameKey="name"
                  cx="50%" cy="50%" innerRadius={58} outerRadius={88} paddingAngle={3}
                >
                  {RISK_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip content={<PieTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 10 }}>
            {RISK_DATA.map(({ name, color }) => (
              <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                <span style={{ fontSize: '0.68rem', color: T.muted, fontWeight: 600 }}>{name}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <CardLabel>Next-Best-Action by Risk Tier</CardLabel>
          {RISK_DATA.map(({ name, patients, pct, color, label, action }) => (
            <div key={name} style={{
              background: T.surface2, borderRadius: 10,
              padding: '13px 16px', border: `1px solid ${color}22`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <div style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: '0.7rem', fontWeight: 800, color, letterSpacing: '0.06em' }}>{name}</span>
                </div>
                <span style={{ fontSize: '0.7rem', color: T.muted }}>{patients.toLocaleString()} pts ({pct})</span>
              </div>
              <div style={{ fontSize: '0.7rem', color: T.muted, marginLeft: 14 }}>{label}</div>
              <div style={{ fontSize: '0.74rem', color: T.text, marginLeft: 14, marginTop: 3, fontWeight: 500 }}>
                → {action}
              </div>
            </div>
          ))}
        </Card>
      </div>

      {/* Top states */}
      <Card style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 22 }}>
          <div>
            <CardLabel>Field Force Opportunity</CardLabel>
            <div style={{ fontSize: '1.05rem', fontWeight: 700, marginTop: -6 }}>
              Top 5 States by Diabetic Population
            </div>
          </div>
          <a href="https://diabetes.org/about-diabetes/statistics/by-state" target="_blank" rel="noreferrer" style={{
            background: '#4338ca22', border: '1px solid #6366f133',
            color: '#a5b4fc', fontSize: '0.6rem', fontWeight: 600,
            padding: '4px 12px', borderRadius: 20, flexShrink: 0,
            textDecoration: 'none',
          }}>Source: ADA 2023 State Fact Sheets</a>
        </div>
        {TOP_STATES.map(({ state, patients, pct, bar }) => (
          <div key={state} style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 6 }}>
              <span style={{ fontWeight: 600 }}>{state}</span>
              <span style={{ color: T.muted }}>{patients} diabetic adults · {pct}% prevalence</span>
            </div>
            <div style={{ background: T.surface2, borderRadius: 5, height: 6, overflow: 'hidden' }}>
              <div style={{
                width: `${bar}%`, height: '100%', borderRadius: 5,
                background: 'linear-gradient(90deg, #6366f1, #818cf8)',
              }} />
            </div>
          </div>
        ))}
        <div style={{ fontSize: '0.74rem', color: T.muted, marginTop: 14, lineHeight: 1.65 }}>
          High-risk patient concentrations inform where pharma field teams should prioritise
          ophthalmology rep visits and HCP engagement campaigns.
        </div>
      </Card>

      {/* NBA flow + Market opportunity */}
      <div className="two-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <Card>
          <CardLabel>Next-Best-Action Pipeline</CardLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {[
              { step: 'Patient Screened',        icon: '01', color: '#6366f1' },
              { step: 'AI Classifies Severity',   icon: '02', color: '#818cf8' },
              { step: 'Risk Tier Assigned',        icon: '03', color: '#a5b4fc' },
            ].map(({ step, icon, color }, i) => (
              <div key={step}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 8, flexShrink: 0,
                    background: color + '22', border: `1px solid ${color}44`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.85rem',
                  }}>{icon}</div>
                  <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>{step}</span>
                </div>
                {i < 2 && (
                  <div style={{
                    width: 1, height: 14, background: T.border,
                    margin: '3px 0 3px 18px',
                  }} />
                )}
              </div>
            ))}
          </div>
          {/* Connector from step 3 → tier boxes */}
          <div style={{ margin: '4px 0 6px' }}>
            {/* Vertical stem from step 3 icon */}
            <div style={{ width: 1, height: 12, background: T.border, marginLeft: 18 }} />
            {/* Horizontal branch spanning all three columns */}
            <div style={{ height: 1, background: T.border }} />
            {/* Three colored drops to tier boxes */}
            <div style={{ display: 'flex', justifyContent: 'space-around' }}>
              {['#10b981', '#f59e0b', '#ef4444'].map((color, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <div style={{ width: 1, height: 10, background: color + '88' }} />
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
                </div>
              ))}
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 4 }}>
            {[
              { tier: 'MONITOR', icon: 'D', color: '#10b981', ch: 'Digital' },
              { tier: 'ENGAGE',  icon: 'C', color: '#f59e0b', ch: 'CRM Flag' },
              { tier: 'ACT NOW', icon: 'F', color: '#ef4444', ch: 'Field Rep' },
            ].map(({ tier, icon, color, ch }) => (
              <div key={tier} style={{
                background: color + '12', border: `1px solid ${color}33`,
                borderRadius: 9, padding: '12px 8px', textAlign: 'center',
              }}>
                <div style={{
                  fontSize: '0.75rem', fontWeight: 900, marginBottom: 5,
                  color: color, letterSpacing: '0.05em',
                }}>{icon}</div>
                <div style={{ fontSize: '0.58rem', fontWeight: 800, color, letterSpacing: '0.05em', marginBottom: 2 }}>{tier}</div>
                <div style={{ fontSize: '0.6rem', color: T.muted }}>{ch}</div>
              </div>
            ))}
          </div>
        </Card>

        <div style={{
          background: 'linear-gradient(135deg, #1a1b3a, #0f1320)',
          border: '1px solid #4338ca44', borderRadius: 16, padding: '28px',
          display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
        }}>
          <div>
            <div style={{
              fontSize: '0.63rem', fontWeight: 700, color: '#818cf8',
              textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 18,
            }}>Market Opportunity</div>
            <div style={{
              fontSize: 'clamp(2.4rem, 5vw, 3.4rem)', fontWeight: 900,
              color: '#a5b4fc', letterSpacing: '-0.04em', lineHeight: 1, marginBottom: 8,
            }}>67,000</div>
            <div style={{ fontSize: '0.88rem', color: '#c7d2fe', marginBottom: 18, lineHeight: 1.5 }}>
              high-risk patients per 500,000 screened
            </div>
            <div style={{ fontSize: '0.8rem', color: T.muted, lineHeight: 1.72 }}>
              In a US health system managing 500,000 diabetic patients, this model identifies
              ~67,000 individuals requiring immediate intervention — a significant addressable
              population for retinal therapy and anti-VEGF treatment brands.
            </div>
          </div>
          <div style={{
            marginTop: 20, padding: '11px 14px',
            background: '#6366f114', border: '1px solid #6366f130',
            borderRadius: 10, fontSize: '0.7rem', color: '#818cf8', lineHeight: 1.6,
          }}>
            Illustrative · 13.4% ACT NOW rate from APTOS 2019 applied to 500,000 patients · ADA/CDC prevalence data
          </div>
        </div>
      </div>
    </section>
  )
}

/* ── SECTION 6: ABOUT ──────────────────────────────────────────────── */

function About() {
  return (
    <section id="about" style={{
      padding: '100px 24px 90px',
      background: 'linear-gradient(180deg, #080b12, #0c0a24)',
      textAlign: 'center',
    }}>
      <div style={{ maxWidth: 700, margin: '0 auto' }}>
        <Label>About This Project</Label>
        <Title>Built to Bridge the Gap</Title>

        <blockquote style={{
          fontSize: 'clamp(0.95rem, 2.5vw, 1.15rem)',
          color: T.muted, lineHeight: 1.8,
          fontStyle: 'italic', marginBottom: 48,
          borderLeft: `3px solid ${T.accent}`,
          paddingLeft: 24, textAlign: 'left',
          maxWidth: 620, margin: '0 auto 48px',
        }}>
          "At its core, this is about connecting the right therapy to the right patient
          at the right time — identifying who needs intervention, routing them to the right
          channel, and giving the commercial team the intelligence to act."
        </blockquote>

        <div style={{ display: 'flex', justifyContent: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 44 }}>
          {[
            { label: 'Live Inference App →', href: 'https://huggingface.co/spaces/priyanka1505/diabetic-retinopathy-detection', primary: true },
            { label: 'GitHub Repository', href: 'https://github.com/priyankaraghunathan15/diabetic-retinopathy-detection' },
            { label: 'APTOS 2019 Dataset', href: 'https://www.kaggle.com/competitions/aptos2019-blindness-detection' },
          ].map(({ label, href, primary }) => (
            <a key={label} href={href} target="_blank" rel="noreferrer" style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: primary ? 'linear-gradient(135deg, #6366f1, #4338ca)' : T.surface,
              border: primary ? 'none' : `1px solid ${T.border}`,
              color: T.text, fontWeight: 600, fontSize: '0.82rem',
              padding: '11px 22px', borderRadius: 10, textDecoration: 'none',
              boxShadow: primary ? '0 4px 18px #6366f130' : 'none',
              transition: 'transform 0.15s',
              fontFamily: 'inherit',
            }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
            onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
            >{label}</a>
          ))}
        </div>

        <div className="three-col" style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 14, marginBottom: 44 }}>
          {[
            { val: 'EfficientNetB3', lbl: 'Model Architecture' },
            { val: 'APTOS 2019',    lbl: 'Training Dataset'   },
            { val: '3,662',         lbl: 'Training Images'    },
          ].map(({ val, lbl }) => (
            <Card key={lbl} style={{ textAlign: 'center', padding: '18px 14px' }}>
              <div style={{ fontWeight: 700, color: T.accent2, marginBottom: 5 }}>{val}</div>
              <div style={{ fontSize: '0.62rem', color: T.muted, textTransform: 'uppercase', letterSpacing: '0.07em' }}>{lbl}</div>
            </Card>
          ))}
        </div>

        <div style={{ fontSize: '0.7rem', color: '#3d4566', lineHeight: 1.8 }}>
          Built on EfficientNetB3 · Trained on APTOS 2019 Blindness Detection Dataset<br />
          For demonstration and educational purposes only · Not for clinical use
        </div>
      </div>
    </section>
  )
}

/* ── ROOT ───────────────────────────────────────────────────────────── */

export default function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <DataSection />
      <EvidenceSection />
      <CommercialSection />
      <About />
    </>
  )
}
