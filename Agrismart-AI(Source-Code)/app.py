import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from datetime import datetime

st.set_page_config(page_title="AgriSmart AI Lab", layout="wide", page_icon="🌿")

# --- LOAD MODEL ---
@st.cache_resource
def load_brain():
    if os.path.exists("models/best_model.pth"):
        checkpoint = torch.load(
    "models/best_model.pth",
    map_location='cpu',
    weights_only=False   
)

        class_names = checkpoint['class_names']

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, class_names
    return None, None

brain, CLASS_NAMES = load_brain()

if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Researcher Profile")
    st.write("**Balbir Singh** | Chitkara University")
    if brain:
        st.success(f"✅ AI Brain Connected ({len(CLASS_NAMES)} Categories)")
    else:
        st.error("Model not found. Train first.")

# --- MAIN UI ---
st.title("🌿 AgriSmart: Dynamic Diagnostic Lab")
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📸 Upload Specimen")
    uploaded_file = st.file_uploader("Choose leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Current Specimen", width=500)

with col2:
    st.subheader("🔬 AI Analysis")

    if uploaded_file and brain:
        if st.button("🚀 Run Deep Diagnostic Scan"):

            # SAME TRANSFORM AS TRAINING
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = brain(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                index = torch.argmax(probabilities).item()
                conf = probabilities[index].item() * 100
                prediction = CLASS_NAMES[index]

            # --- CLEAN DISPLAY ---
            if '___' in prediction:
                parts = prediction.split('___')
            elif '__' in prediction:
                parts = prediction.split('__')
            else:
                parts = prediction.split('_', 1)

            plant = parts[0].replace('_', ' ').title()
            status = parts[1].replace('_', ' ').title()

            st.metric(label="Specimen Identified", value=plant)
            st.subheader(f"Diagnosis: {status}")
            st.progress(conf / 100)
            st.write(f"**Confidence Score: {conf:.2f}%**")

            if "Healthy" in status:
                st.success("Result: Optimal Health Detected.")
            else:
                st.error(f"Alert: {status} confirmed.")

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "leaf": plant,
                "status": status
            })

# --- HISTORY ---
if st.session_state.history:
    st.markdown("---")
    st.table(st.session_state.history)