import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import json
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

.stApp { background: #04080f; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 780px !important;
    padding: 1rem 1.5rem 2rem !important;
}

/* ── NAV ── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: #070d1a;
    border: 1px solid #0d1f3c;
    border-radius: 14px;
    margin-bottom: 40px;
}
.nav-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
}
.nav-pill {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(56,189,248,0.08);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
    letter-spacing: 0.04em;
}
.nav-right {
    font-size: 12px;
    color: #334155;
    font-weight: 500;
}

/* ── HERO ── */
.hero { text-align: center; margin-bottom: 36px; }
.hero-eyebrow {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 999px;
    padding: 5px 14px;
    margin-bottom: 18px;
    background: rgba(56,189,248,0.04);
}
.hero-title {
    font-size: 44px !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    letter-spacing: -1.2px !important;
    color: #f8fafc !important;
    margin: 0 0 14px !important;
    background: none !important;
    -webkit-text-fill-color: #f8fafc !important;
}
.hero-title b {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 15px;
    color: #475569;
    line-height: 1.7;
    max-width: 480px;
    margin: 0 auto;
}

/* ── STATS ── */
.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 36px;
}
.stat {
    background: #070d1a;
    border: 1px solid #0d1f3c;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-n {
    font-size: 20px;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    white-space: nowrap;
}
.stat-n em { color: #38bdf8; font-style: normal; }
.stat-l {
    font-size: 10px;
    color: #334155;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── UPLOAD ── */
.upload-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 10px;
}
section[data-testid="stFileUploader"] {
    background: #070d1a !important;
    border: 1.5px dashed #0d1f3c !important;
    border-radius: 16px !important;
    padding: 24px !important;
    transition: border-color 0.25s;
}
section[data-testid="stFileUploader"]:hover {
    border-color: #1e4976 !important;
}

/* ── TYPE CARDS ── */
.types { margin-top: 36px; }
.types-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 10px;
}
.tc {
    background: #070d1a;
    border: 1px solid #0d1f3c;
    border-radius: 12px;
    padding: 16px 14px;
}
.tc-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-bottom: 10px;
}
.tc-name {
    font-size: 12px;
    font-weight: 700;
    color: #cbd5e1;
    margin-bottom: 5px;
}
.tc-desc {
    font-size: 10px;
    color: #334155;
    line-height: 1.6;
}

/* ── RESULTS ── */
.res-header {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #334155;
    margin: 28px 0 12px;
}
.res-main {
    background: #070d1a;
    border: 1px solid #0d1f3c;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 12px;
}
.res-status {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 14px;
}
.res-name {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.res-model {
    font-size: 11px;
    color: #334155;
    font-weight: 500;
    margin-bottom: 20px;
}
.conf-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 6px;
}
.conf-num {
    font-size: 40px;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 10px;
}
.conf-pct { font-size: 18px; font-weight: 500; color: #475569; }
.bar-bg {
    height: 5px;
    background: #0d1f3c;
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
}

/* about */
.about-card {
    background: #070d1a;
    border: 1px solid #0d1f3c;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 12px;
}
.about-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 12px;
}
.about-text {
    font-size: 13px;
    color: #64748b;
    line-height: 1.8;
    margin-bottom: 14px;
}
.symp-head {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1e3a5f;
    margin-bottom: 6px;
}
.symp-body { font-size: 12px; color: #334155; }

/* chart */
.chart-card {
    background: #070d1a;
    border: 1px solid #0d1f3c;
    border-radius: 16px;
    padding: 20px 24px 8px;
    margin-bottom: 12px;
}

/* disclaimer */
.disc {
    background: rgba(56,189,248,0.03);
    border: 1px solid #0d1f3c;
    border-left: 3px solid #0d3a5c;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 11px;
    color: #334155;
    line-height: 1.7;
    margin-top: 4px;
}
.disc b { color: #1e4976; }

/* footer */
.foot {
    border-top: 1px solid #0a1628;
    margin-top: 48px;
    padding-top: 20px;
    text-align: center;
    font-size: 11px;
    color: #1e3a5f;
    line-height: 1.8;
}

/* hide streamlit progress */
.stProgress { display: none !important; }
div[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid #0d1f3c !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
with open("classes.json") as f:
    classes = json.load(f)

display_names = {
    "glioma":     "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary":  "Pituitary Tumor",
    "notumor":    "No Tumor Detected"
}

tumor_info = {
    "glioma": {
        "desc": "Gliomas arise from glial cells in the brain or spine. They are the most common primary brain tumor and range from low-grade to aggressive high-grade variants requiring prompt clinical evaluation.",
        "symp": "Persistent headaches, seizures, memory loss, personality changes, blurred vision.",
        "color": "#f87171",
        "status_bg": "rgba(248,113,113,0.08)",
        "status_border": "rgba(248,113,113,0.25)",
        "dot": "#f87171", "bar": "#f87171",
        "status": "⚠ Tumor Detected"
    },
    "meningioma": {
        "desc": "Meningiomas form in the meninges surrounding the brain and spinal cord. Most are benign and slow-growing, though location can cause significant neurological symptoms.",
        "symp": "Headaches, vision or hearing changes, memory problems, limb weakness.",
        "color": "#fb923c",
        "status_bg": "rgba(251,146,60,0.08)",
        "status_border": "rgba(251,146,60,0.25)",
        "dot": "#fb923c", "bar": "#fb923c",
        "status": "⚠ Tumor Detected"
    },
    "pituitary": {
        "desc": "Pituitary tumors develop at the base of the brain affecting hormone regulation. The vast majority are benign adenomas with excellent outcomes when detected early.",
        "symp": "Hormonal imbalances, vision changes, chronic headaches, fatigue, weight changes.",
        "color": "#818cf8",
        "status_bg": "rgba(129,140,248,0.08)",
        "status_border": "rgba(129,140,248,0.25)",
        "dot": "#818cf8", "bar": "#818cf8",
        "status": "⚠ Tumor Detected"
    },
    "notumor": {
        "desc": "No tumor detected. The MRI scan appears within normal parameters. Regular neurological screenings are recommended as part of routine health monitoring.",
        "symp": "",
        "color": "#34d399",
        "status_bg": "rgba(52,211,153,0.08)",
        "status_border": "rgba(52,211,153,0.25)",
        "dot": "#34d399", "bar": "#34d399",
        "status": "✓ Clear Scan"
    }
}

@st.cache_resource
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Linear(256, len(classes))
    )
    m.load_state_dict(torch.load("best_model.pth", map_location=device))
    m.to(device)
    m.eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── NAV ──────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-left">
    🧠 &nbsp;NeuroScan <span style="color:#38bdf8;">AI</span>
    &nbsp;<span class="nav-pill">EfficientNet-B0</span>
  </div>
  <div class="nav-right">University of Hail &nbsp;·&nbsp; AI & Data Science</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Neuroimaging</div>
  <div class="hero-title">Brain Tumor <b>Detection</b></div>
  <p class="hero-sub">
    Upload a brain MRI scan and receive an instant classification —
    Glioma, Meningioma, Pituitary, or No Tumor.
  </p>
</div>
""", unsafe_allow_html=True)

# ── STATS ────────────────────────────────────────────────────────
st.markdown("""
<div class="stats">
  <div class="stat">
    <div class="stat-n">7<em>,023</em></div>
    <div class="stat-l">Training Images</div>
  </div>
  <div class="stat">
    <div class="stat-n"><em>4</em> Classes</div>
    <div class="stat-l">Tumor Types</div>
  </div>
  <div class="stat">
    <div class="stat-n" style="font-size:15px;">EfficientNet&#8209;B0</div>
    <div class="stat-l">Architecture</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ───────────────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# ── TYPE CARDS (only when no upload) ────────────────────────────
if not uploaded:
    st.markdown("""
    <div class="types">
      <div class="upload-label">Detectable Classifications</div>
      <div class="types-grid">
        <div class="tc">
          <div class="tc-dot" style="background:#f87171;"></div>
          <div class="tc-name">Glioma</div>
          <div class="tc-desc">Most common primary brain tumor from glial cells.</div>
        </div>
        <div class="tc">
          <div class="tc-dot" style="background:#fb923c;"></div>
          <div class="tc-name">Meningioma</div>
          <div class="tc-desc">Forms in the meninges. Usually benign.</div>
        </div>
        <div class="tc">
          <div class="tc-dot" style="background:#818cf8;"></div>
          <div class="tc-name">Pituitary</div>
          <div class="tc-desc">Affects pituitary gland and hormones.</div>
        </div>
        <div class="tc">
          <div class="tc-dot" style="background:#34d399;"></div>
          <div class="tc-name">No Tumor</div>
          <div class="tc-desc">Normal brain tissue detected.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── RESULTS ──────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    color_diff = np.abs(r-g).mean() + np.abs(r-b).mean() + np.abs(g-b).mean()

    if color_diff > 60:
        st.markdown("""
        <div style="margin-top:20px; background:#070d1a; border:1px solid #2d1515;
                    border-radius:14px; padding:18px 20px; color:#f87171; font-size:13px; font-weight:500;">
          ⚠️ &nbsp;Please upload a valid grayscale brain MRI image.
        </div>""", unsafe_allow_html=True)
    else:
        x = transform(image).unsqueeze(0).to(device)
        with st.spinner("Analyzing..."):
            with torch.no_grad():
                out   = model(x)
                probs = torch.softmax(out, dim=1)[0]

        pred_idx  = probs.argmax().item()
        raw_class = classes[str(pred_idx)]
        conf      = probs[pred_idx].item() * 100
        predicted = display_names.get(raw_class, raw_class)
        info      = tumor_info[raw_class]

        if conf < 85:
            st.markdown("""
            <div style="margin-top:20px; background:#070d1a; border:1px solid #0d1f3c;
                        border-radius:14px; padding:18px 20px; color:#64748b; font-size:13px; font-weight:500;">
              ⚠️ &nbsp;Low confidence. Please upload a clearer MRI scan.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="res-header">Analysis Results</div>', unsafe_allow_html=True)

            # Image + result side by side
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.image(image, use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class="res-main" style="height:100%;">
                  <div class="res-status"
                       style="background:{info['status_bg']};
                              color:{info['color']};
                              border:1px solid {info['status_border']};">
                    {info['status']}
                  </div>
                  <div class="res-name" style="color:{info['color']};">{predicted}</div>
                  <div class="res-model">EfficientNet-B0 · Deep Learning</div>
                  <div class="conf-label">Confidence Score</div>
                  <div class="conf-num" style="color:{info['color']};">
                    {conf:.1f}<span class="conf-pct">%</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill"
                         style="width:{conf}%; background:{info['bar']};">
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="about-title">Class Probability Distribution</div>', unsafe_allow_html=True)
            labels = [display_names[classes[str(i)]] for i in range(len(classes))]
            values = [round(probs[i].item() * 100, 2) for i in range(len(classes))]
            bar_colors = [
                info['bar'] if classes[str(i)] == raw_class else "#0d1f3c"
                for i in range(len(classes))
            ]
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(color="#334155", size=11),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", family="Inter, sans-serif"),
                xaxis=dict(range=[0,118], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#475569")),
                margin=dict(l=0, r=55, t=0, b=0),
                height=165, bargap=0.45,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # About
            symp_html = f"""
            <div style="margin-top:12px;">
              <div class="symp-head">Common Symptoms</div>
              <div class="symp-body">{info['symp']}</div>
            </div>""" if info['symp'] else ""

            st.markdown(f"""
            <div class="about-card">
              <div class="about-title">About This Classification</div>
              <div class="about-text">{info['desc']}</div>
              {symp_html}
            </div>
            <div class="disc">
              <b>⚕ Medical Disclaimer</b><br>
              This tool is intended for educational and research purposes only.
              Results do not replace professional medical diagnosis.
              Always consult a licensed neurologist or radiologist for clinical decisions.
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  NeuroScan AI &nbsp;·&nbsp;
  Department of Artificial Intelligence &amp; Data Science &nbsp;·&nbsp;
  University of Hail &nbsp;·&nbsp; 2025
</div>
""", unsafe_allow_html=True)
