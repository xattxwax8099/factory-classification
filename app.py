# app.py — Minimal, clean UI + tidy code
# (Blue background blobs, interactive title, spaced layout, toggle table)
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import densenet121
from PIL import Image
import streamlit as st
import pandas as pd

# ---------- Basic Config ----------
st.set_page_config(page_title="Industry Machine Image Classification", page_icon="🏭", layout="centered")

# ---------- Constants ----------
MODEL_PATH = "densenet121_checkpoint_fold0.pt"
CLASS_NAMES = [
    "fan abnormal", "fan normal",
    "pump abnormal", "pump normal",
    "slider abnormal", "slider normal",
    "valve abnormal", "valve normal",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------- Theme & CSS ----------
st.markdown("""
<style>
:root{
  --bg:#f0f4ff; --text:#232629; --muted:#6b7280; --card:#ffffff; --border:#eef2f7;

  /* gradients (หัวข้อ + predicted ใช้ชุดเดียวกัน) */
  --grad-title-1:#ffecd2;  /* พีชอ่อน */
  --grad-title-2:#fcb69f;  /* ส้มอ่อน */

  --pill:#ffffff;
}

/* ---------- GLOBAL BACKGROUND (blue + blobs) ---------- */
[data-testid="stAppViewContainer"]{
  background: var(--bg);
  background-image:
    radial-gradient(circle at 20% 30%, rgba(255,200,150,0.25) 0, transparent 60%),
    radial-gradient(circle at 80% 70%, rgba(150,200,255,0.25) 0, transparent 60%),
    linear-gradient(180deg, #f0f4ff 0%, #ffffff 100%);
  background-attachment: fixed;
  color: var(--text);
}

/* ---------- Layout spacing ---------- */
.block-container{
  max-width: 960px;
  padding-top: 3rem;
  padding-bottom: 4rem;
  padding-left: 2rem;
  padding-right: 2rem;
}

/* ---------- Card ---------- */
.card{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 18px; padding: 28px 24px;
  box-shadow: 0 8px 24px rgba(16,24,40,0.05);
  animation: float-in .6s ease-out both;
}

/* ---------- INTERACTIVE TITLE ---------- */
.title-wrap{
  text-align:center; border-radius:18px; border: 1px solid var(--border);
  box-shadow: 0 8px 24px rgba(16,24,40,0.05);
  margin-top: 1rem; margin-bottom: 2rem;
  position: relative; overflow: hidden;
  background-image: linear-gradient(135deg, var(--grad-title-1), var(--grad-title-2));
  background-size: 200% 200%;
  animation: slow-shift 10s ease-in-out infinite alternate;
  cursor: pointer;
  transition: transform .25s ease, box-shadow .25s ease;
}
.title-wrap:focus { outline: none; box-shadow: 0 0 0 3px rgba(0,0,0,.08), 0 8px 24px rgba(16,24,40,0.10); }
.title-wrap:hover{
  transform: translateY(-4px) scale(1.01);
  box-shadow: 0 14px 32px rgba(16,24,40,0.12);
}
/* shine */
.title-wrap::after{
  content:""; position:absolute; top:-60%; left:-140%;
  width: 50%; height: 220%;
  background: linear-gradient(120deg, rgba(255,255,255,0) 0%, rgba(255,255,255,.35) 45%, rgba(255,255,255,0) 100%);
  transform: skewX(-20deg);
  pointer-events: none;
}
.title-wrap:hover::after{ animation: shine 1.1s ease; }
.title-wrap:active{ transform: translateY(-1px) scale(0.995); }

h1.hero-title{
  margin:0; font-weight:800; letter-spacing:.2px; color:#1f2937;
  font-size: clamp(26px, 4.6vw, 40px);
}
.title-sub{
  margin:.6rem 0 0; color:#2b2b2b; font-size: .98rem; font-weight: 600; opacity: .9;
}

/* ---------- Predicted hero (ใช้สี header แต่จางลง) ---------- */
.hero{
  margin-top: 2.5rem; margin-bottom: 2rem;
  border-radius: 20px; padding: 30px 26px; text-align: center;
  box-shadow: 0 10px 26px rgba(16,24,40,0.12);
  background-image:
    linear-gradient(0deg, rgba(255,255,255,0.28), rgba(255,255,255,0.28)),
    linear-gradient(135deg, var(--grad-title-1), var(--grad-title-2));
  background-size: 200% 200%, 200% 200%;
  animation: pop-in .5s ease-out both, slow-shift 12s ease-in-out infinite alternate;
}
.hero .title{
  font-size: clamp(22px, 4.2vw, 34px);
  font-weight: 800; margin: 0; color:#1f2937;
}
.hero .pill{
  display: inline-block; margin-top: 14px; padding: .55rem 1.05rem; border-radius: 999px;
  font-weight: 800; color: #1f2937; background: var(--pill); border: 1px solid rgba(0,0,0,.08);
  box-shadow: 0 4px 18px rgba(0,0,0,.08);
  animation: breathe 2.8s ease-in-out infinite;
}

/* ---------- Subheader ของตาราง ---------- */
.subheader-custom {
  background-color: rgba(239,230,217,.75);
  padding: 10px 16px;
  border-radius: 12px;
  font-weight: 800;
  font-size: 1.1rem;
  margin-top: 2rem;
  margin-bottom: 1rem;
  color: #2b2b2b;
  display: inline-block;
  animation: float-in .6s .15s ease-out both;
}

/* ปุ่ม */
button[kind="primary"], .stButton>button{
  transition: transform .12s ease, box-shadow .12s ease;
  font-weight: 700;
}
button[kind="primary"]:hover, .stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(0,0,0,.08);
}

/* ---------- Animations ---------- */
@keyframes pop-in{ from{opacity:0; transform: translateY(8px) scale(.98);} to{opacity:1; transform: none;} }
@keyframes float-in{ from{opacity:0; transform: translateY(10px);} to{opacity:1; transform: translateY(0);} }
@keyframes breathe{ 0%,100%{transform: scale(1);} 50%{transform: scale(1.03);} }
@keyframes slow-shift{ 0%{ background-position: 0% 50%; } 100%{ background-position: 100% 50%; } }
@keyframes shine{ to{ left: 200%; } }

/* Reduce motion */
@media (prefers-reduced-motion: reduce) {
  .title-wrap, .hero{ animation: none; }
  .title-wrap:hover{ transform: none; box-shadow: 0 8px 24px rgba(16,24,40,0.05); }
  .hero .pill{ animation: none; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="title-wrap card" role="button" tabindex="0" aria-label="Interactive title">
  <h1 class="hero-title">Industry Machine Image Classification</h1>
  <p class="title-sub">Minimal interface • Highlighted headers • Easy to read</p>
</div>
""", unsafe_allow_html=True)

# ---------- Image Preprocess ----------
_img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return _img_transform(img).unsqueeze(0)

# ---------- Model Loader ----------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    device = torch.device("cpu")
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    ckpt = torch.load(path, map_location=device)
    state = None

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    elif isinstance(ckpt, nn.Module):
        model = ckpt
        state = None
    else:
        try:
            model = torch.jit.load(path, map_location=device)
            state = None
        except Exception:
            pass

    if state is not None:
        cleaned = {k.replace("model.","").replace("module.",""): v for k,v in state.items()}
        model.load_state_dict(cleaned, strict=False)

    model.to(device).eval()
    return model

@torch.inference_mode()
def predict(model, x: torch.Tensor) -> np.ndarray:
    y = model(x)
    return torch.softmax(y, dim=1)[0].cpu().numpy()

# ---------- Helpers ----------
def style_df(df: pd.DataFrame, predicted_label: str) -> pd.io.formats.style.Styler:
    def highlight_row(row):
        base = 'background-color: #fff;'
        if row["Class"] == predicted_label:
            return ['background-color: #fff7ed; border-left: 4px solid #fb923c'] * len(row)
        return [base] * len(row)

    return (
        df.style
        .format({"Confidence (%)": "{:.2f}"})
        .apply(highlight_row, axis=1)
        .bar(subset=["Confidence (%)"], color='#c7d2fe')
    )

# ---------- UI ----------
file = st.file_uploader("อัปโหลดรูปภาพ (PNG/JPG)", type=["png","jpg","jpeg"])
model = load_model(MODEL_PATH)

if file is None:
    st.info("อัปโหลดรูปเพื่อให้โมเดลทำนายคลาส")
else:
    file_bytes = file.getvalue()

    # รูปตรงกลาง
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image(file_bytes, caption="Preview", width=340)

    # Progress
    progress = st.progress(0, text="กำลังประมวลผล…")
    x = preprocess_image(file_bytes); progress.progress(35, text="กำลังประมวลผล…")
    probs = predict(model, x);        progress.progress(100, text="เสร็จสิ้น")
    progress.empty()

    # Result
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = float(probs[idx]) * 100.0

    st.markdown(f"""
    <div class="hero">
      <div class="title">Predicted: {label}</div>
      <div class="pill">Confidence: {conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Toggle table (เหมือนเดิม)
    show_table = st.toggle("ดูตารางการทำนาย", value=False)

    if show_table:
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Confidence (%)": (probs * 100).round(2)
        }).sort_values("Confidence (%)", ascending=False).reset_index(drop=True)

        st.markdown('<div class="subheader-custom">Confidence (all classes)</div>', unsafe_allow_html=True)
        st.dataframe(
            style_df(df, predicted_label=label),
            use_container_width=True,
            hide_index=True
        )
