# app.py
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import densenet121
from PIL import Image
import streamlit as st
import pandas as pd
import altair as alt

import torch, streamlit as st
import timm  # หรือ torchvision.models ถ้าใช้ตรง ๆ

MODEL_PATH = "./densenet121_checkpoint_fold0.pt"

@st.cache_resource
def load_model():
    model = timm.create_model("densenet121", pretrained=False, num_classes=8)
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

model = load_model()
st.success("✅ Model loaded on CPU")

# ---------- Config ----------
st.set_page_config(page_title="ML Demo", page_icon="🤖", layout="centered")


CLASS_NAMES = [
    "fan abnormal", "fan normal",
    "pump abnormal", "pump normal",
    "slider abnormal", "slider normal",
    "valve abnormal", "valve normal"
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = r"densenet121_checkpoint_fold0.pt"

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;700&family=Noto+Sans+Thai:wght@400;700&display=swap');
:root{
  --bg-light:#f7faff; --text:#3a3a3a; --muted:#7a7a7a;
  --primary:#ff90b3; --accent1:#a3d8f4; --accent2:#ffe066; --accent3:#b5ead7;
  --border:#e0e7ef;
}
[data-testid="stAppViewContainer"]{
  position: relative; z-index: 0; color: var(--text) !important;
  font-family: 'Noto Sans Thai','Fredoka',Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
  background: var(--bg-light);
}
[data-testid="stAppViewContainer"]::before{
  content:""; position: fixed; inset: 0; z-index: -1;
  background:
    radial-gradient(1200px 800px at 15% 20%, rgba(255,144,179,0.18), transparent 60%),
    radial-gradient(1000px 700px at 85% 15%, rgba(163,216,244,0.18), transparent 60%),
    radial-gradient(900px 600px at 30% 90%, rgba(255,224,102,0.18), transparent 60%),
    var(--bg-light);
  filter: saturate(110%) contrast(102%);
  animation: drift 24s ease-in-out infinite;
}
@keyframes drift{ 0%{transform:translate3d(0,0,0) scale(1)} 50%{transform:translate3d(-1.2%,-1.5%,0) scale(1.02)} 100%{transform:translate3d(0,0,0) scale(1)} }
@media (prefers-reduced-motion: reduce){ [data-testid="stAppViewContainer"]::before{ animation:none; } }
.block-container{ max-width: 980px; padding-top: 1.2rem; padding-bottom: 3rem; }

/* การ์ดหัวเรื่อง */
.card{
  background: rgba(255,255,255,0.88);
  border: 1px solid var(--border);
  border-radius: 22px; padding: 20px;
  box-shadow: 0 8px 18px rgba(163,216,244,0.15), inset 0 1px 0 rgba(255,255,255,0.04);
  backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);
  margin-left: auto; margin-right: auto;
}

/* HERO ผลลัพธ์ใหญ่ */
.predicted-hero{
  margin-top: 14px; background: #fff; border: 1px solid var(--border);
  border-radius: 24px; padding: 22px 24px;
  box-shadow: 0 10px 22px rgba(163,216,244,0.18);
  text-align: center;
}
.predicted-hero .title{
  font-size: clamp(28px, 5vw, 44px);
  font-weight: 800; letter-spacing: .2px; margin: 0 0 6px 0; color: var(--text);
}
.predicted-hero .conf{
  display: inline-block; margin-top: 6px; padding: .45rem .9rem; border-radius: 999px;
  font-weight: 700; color: var(--primary);
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--primary) 28%, transparent);
}

/* ตาราง/กราฟ */
.vega-embed, .stDataFrame { color: var(--text); }
</style>
""", unsafe_allow_html=True)

# ---------- Header (single, centered, gradient text) ----------
st.markdown("""
<div class="card">
  <h1 style="
    text-align:center;
    margin:0;
    font-weight:800;
    background: linear-gradient(90deg, #ff90b3, #a3d8f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;">
    Industry Machine Image Classification
  </h1>
</div>
""", unsafe_allow_html=True)

# ---------- Preprocess ----------
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
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    device = torch.device("cpu")
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        state = None
        for key in ["model_state_dict","state_dict","weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        if state is not None:
            cleaned = {k.replace("model.","").replace("module.",""):v for k,v in state.items()}
            model.load_state_dict(cleaned, strict=False)
        elif isinstance(ckpt, nn.Module):
            model = ckpt
    else:
        model = torch.jit.load(path, map_location=device)
    model.to(device).eval()
    return model

@torch.inference_mode()
def predict(model, x):
    y = model(x)
    probs = torch.softmax(y, dim=1)[0].cpu().numpy()
    return probs

# ---------- UI ----------
model = load_model(MODEL_PATH)
file = st.file_uploader("อัปโหลดรูป (PNG/JPG)", type=["png","jpg","jpeg"])

if file:
    file_bytes = file.getvalue()

    # รูปอยู่กลาง + ย่อให้เล็กลง
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(file_bytes, width=320)

    with st.spinner("⚙️ กำลังทำนาย..."):
        x = preprocess_image(file_bytes)
        probs = predict(model, x)

    # ผลลัพธ์ Top-1
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = float(probs[idx]) * 100.0

    st.markdown(f"""
    <div class="predicted-hero">
      <div class="title">Predicted: {label}</div>
      <div class="conf">Confidence: {conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # เปรียบเทียบทุกคลาส
    st.subheader("📊 Confidence comparison (all classes)")
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Confidence (%)": (probs * 100).round(2)
    }).sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    chart = (
        alt.Chart(df)
        .mark_bar(color="#ff90b3")
        .encode(
            x=alt.X("Confidence (%):Q", title="Confidence (%)", scale=alt.Scale(domain=[0,100])),
            y=alt.Y("Class:N", sort="-x", title=""),
            tooltip=["Class","Confidence (%)"]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

else:
    st.info("อัปโหลดรูปเพื่อให้โมเดลทำนายคลาส")
