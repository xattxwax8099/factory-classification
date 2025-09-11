# app.py — Minimal, clean UI + tidy code (Warm white theme + bubbles + EfficientNet fix)
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import densenet121, densenet201
import timm
from PIL import Image
import streamlit as st
import pandas as pd
import random
from pandas.io.formats.style import Styler

# ---------- Basic Config ----------
st.set_page_config(page_title="Industry Machine Image Classification", page_icon="🏭", layout="centered")

# ---------- Constants ----------
CLASS_NAMES = [
    "fan abnormal", "fan normal",
    "pump abnormal", "pump normal",
    "slider abnormal", "slider normal",
    "valve abnormal", "valve normal",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------- Bubble Generator ----------
NUM_BUBBLES = 12
bubble_divs = []
colors = ["rgba(144,202,235,0.35)", "rgba(254,212,110,0.35)",
          "rgba(220,150,255,0.35)", "rgba(255,182,193,0.35)",
          "rgba(163,49,102,0.25)"]

for i in range(NUM_BUBBLES):
    size = random.randint(60, 150)
    left = random.randint(5, 90)
    delay = random.randint(0, 15)
    duration = random.randint(15, 30)
    color = random.choice(colors)
    bubble_divs.append(
        f'<div class="bubble" style="width:{size}px; height:{size}px; '
        f'left:{left}%; background:{color}; '
        f'animation-duration:{duration}s; animation-delay:{delay}s;"></div>'
    )

# ---------- Theme & CSS ----------
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {{
  font-family: 'Poppins','Noto Sans Thai', sans-serif;
}}

:root {{
  --bg:#f9fafb; --text:#1f2937;
  --warm-1:#fdf6ec; --warm-2:#fae8c8;
  --card:#ffffff; --border:#e5e7eb; --pill:#f9fafb;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#111827; --text:#f9fafb;
    --warm-1:#2a2a2a; --warm-2:#3a3a3a;
    --card:#1f2937; --border:#374151; --pill:#1f2937;
  }}
}}

[data-testid="stAppViewContainer"]{{
  background: var(--bg);
  overflow: hidden;
  color: var(--text);
}}

.bubble {{
  position: fixed;
  bottom: -200px;
  border-radius: 50%;
  opacity: 0.5;
  animation: rise infinite ease-in;
  z-index: 1;
}}
@keyframes rise {{
  0%   {{ transform: translateY(0) scale(1); opacity: 0.6; }}
  50%  {{ transform: translateY(-50vh) scale(1.15); opacity: 0.8; }}
  100% {{ transform: translateY(-110vh) scale(1); opacity: 0; }}
}}

.block-container{{
  max-width: 900px;
  padding: 2rem;
  position: relative;
  z-index: 2;
}}

/* ---------- Title ---------- */
.title-wrap{{
  text-align:center; border-radius:16px;
  padding: 1.6rem 1.2rem;
  margin-bottom: 1.5rem;
  background-image: linear-gradient(135deg, var(--warm-1), var(--warm-2));
  border: 1px solid var(--border);
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
h1.hero-title{{
  margin:0; font-weight:700; color:var(--text);
  font-size: clamp(32px, 5vw, 48px);
}}
.title-sub{{
  margin-top:.5rem; font-size:1rem; font-weight:500;
  color:#444;
}}

/* ---------- Predicted hero ---------- */
.hero{{
  margin: 2rem 0;
  border-radius: 16px;
  padding: 26px 22px;
  text-align: center;
  background-image: linear-gradient(135deg, var(--warm-1), var(--warm-2));
  border: 1px solid var(--border);
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}}
.hero .title{{ font-size: clamp(22px, 4.5vw, 34px); font-weight: 700; margin: 0; color:var(--text); }}
.hero .pill{{
  display: inline-block; margin-top: 12px; padding: .45rem 1.1rem;
  border-radius: 999px; font-weight: 600;
  color: var(--text); background: #ffffff; border: 1px solid var(--border);
}}

/* ---------- Selectbox ---------- */
.stSelectbox label {{
  font-weight:600; color: var(--text);
}}
.stSelectbox div[data-baseweb="select"] > div {{
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--card);
  color: var(--text);
  font-size: 0.95rem;
}}
.stSelectbox div[data-baseweb="select"]:hover {{
  border-color: #f59e0b;
}}

/* ---------- DataFrame ---------- */
[data-testid="stDataFrame"] {{
  border-radius: 12px;
  border: 1px solid var(--border);
  overflow: hidden;
  background: var(--card);
}}
[data-testid="stDataFrame"] table {{
  font-size: 0.9rem; color: var(--text);
}}
</style>
{''.join(bubble_divs)}
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="title-wrap">
  <h1 class="hero-title">Industry Machine Image Classification</h1>
  <p class="title-sub">"Smart predictions • AI-powered classification • Easy to understand"</p>
</div>
""", unsafe_allow_html=True)

# ---------- Model Options ----------
MODEL_OPTIONS = {
    "DenseNet121": ("densenet121_checkpoint_fold0.pt", "DenseNet121"),
    "EfficientNet-B0": ("efficientnet_b0_checkpoint_fold3.pt", "timm_efficientnet_b0")
}
selected_model_name = st.selectbox("เลือกโมเดล", list(MODEL_OPTIONS.keys()))
MODEL_PATH, MODEL_TYPE = MODEL_OPTIONS[selected_model_name]

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
def load_model(path: str, model_name: str):
    device = torch.device("cpu")

    # เลือก backbone ให้ตรง
    if model_name == "DenseNet121":
        model = densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "DenseNet201":
        model = densenet201(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "timm_efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)

    else:
        raise ValueError(f"ไม่รู้จักโมเดล: {model_name}")

    # โหลด checkpoint
    ckpt = torch.load(path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    elif isinstance(ckpt, nn.Module):
        model = ckpt; state = None
    else:
        try: 
            model = torch.jit.load(path, map_location=device); state = None
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
def style_df(df: pd.DataFrame, predicted_label: str) -> Styler:
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
model = load_model(MODEL_PATH, MODEL_TYPE)

if file:
    file_bytes = file.getvalue()
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image(file_bytes, caption="Preview", width=340)

    progress = st.progress(0, text="กำลังประมวลผล…")
    x = preprocess_image(file_bytes); progress.progress(35)
    probs = predict(model, x); progress.progress(100); progress.empty()

    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = float(probs[idx]) * 100.0

    st.markdown(f"""
    <div class="hero">
      <div class="title">Predicted: {label}</div>
      <div class="pill">Confidence: {conf:.2f}%</div>
      <p style="margin-top:10px; font-size:0.9rem; font-weight:500; color:#444;">
        Model Used: {selected_model_name}
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.toggle("ดูตารางการทำนาย", value=False):
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Confidence (%)": (probs * 100).round(2)
        }).sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
        st.dataframe(style_df(df, predicted_label=label), use_container_width=True, hide_index=True)
else:
    st.info("อัปโหลดรูปเพื่อให้โมเดลทำนายคลาส")
