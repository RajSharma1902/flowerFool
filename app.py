import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
import numpy as np
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.qnn import TorchLayer
import torch.nn as nn
import os

# Fix file watcher issue on some systems
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# --- Quantum Layer Setup ---
@st.cache_resource
def load_quantum_layer(n_qubits=8, n_q_layers=4):
    dev = qml.device("default.qubit", wires=range(n_qubits))

    def quantum_circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_q_layers, n_qubits, 3)}
    qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")
    return TorchLayer(qnode, weight_shapes)

# --- Model Definition ---
class HybridCNNVQC(nn.Module):
    def __init__(self, n_classes, quantum_layer, n_qubits):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, n_qubits), nn.Tanh()
        )
        self.quantum = quantum_layer
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        feats = self.cnn(x)
        angles = (feats + 1) * (torch.pi / 2)
        q_out = self.quantum(angles)
        return self.classifier(q_out)

# --- Load Model ---
@st.cache_resource
def load_model(model_path, class_names):
    n_qubits = 8
    q_layer = load_quantum_layer(n_qubits=n_qubits)
    model = HybridCNNVQC(n_classes=len(class_names), quantum_layer=q_layer, n_qubits=n_qubits)
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

# --- UI Setup ---
st.set_page_config(page_title="Flower Classifier VQC", layout="centered")
st.title("🌸 Flower Recognition using Quantum-Classical Model")

# Load class names and model
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
model = load_model("flowerPred_vqc_model.pth", CLASS_NAMES)

# Image Upload Section
uploaded_file = st.file_uploader("Upload a flower image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
if uploaded_file:
    try:
        # Convert to RGB JPG internally
        img = Image.open(uploaded_file).convert("RGB")

        # Preprocess
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        img_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 64, 64]

        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
            pred_idx = np.argmax(probs)
            pred_name = CLASS_NAMES[pred_idx]

        # Display result
        st.image(img.resize((256, 256)), caption="Uploaded Image", use_container_width =True)
        st.success(f"**Prediction: {pred_name.title()}**")

        st.subheader("Class Probabilities:")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"- {name.title()}: **{probs[i]*100:.2f}%**")

    except UnidentifiedImageError:
        st.error("❌ Unsupported image format or corrupted file.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
