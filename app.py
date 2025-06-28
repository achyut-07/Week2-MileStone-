import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set page config
st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="centered")

# Load model
model = load_model('garbage_classifier.keras')
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title and Description
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2E86AB;'>🌍 Garbage Classification AI</h1>
        <p style='font-size:18px;'>Upload an image of garbage, and the AI will identify its category using a deep learning model.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📘 About the Project")
st.sidebar.info("""
This AI model classifies garbage into:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

🧠 Model: MobileNetV2 (Transfer Learning)

🎓 Built for AICTE Internship
""")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"🧠 Predicted Class: **{predicted_class.upper()}** ✅")

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; color: gray;'>
        Built with ❤️ by Pawan Achyutanand<br>
        Powered by TensorFlow & Streamlit<br>
        Special thanks to AICTE for the internship opportunity            
    </div>
""", unsafe_allow_html=True)
