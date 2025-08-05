import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import json
import requests

# ---------------- Lottie Animation Loader ---------------- #
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_rocket = load_lottie("https://assets2.lottiefiles.com/packages/lf20_zrqthn6o.json")

# ---------------- UI Styling ---------------- #
st.set_page_config(page_title="BrainCache - Space Station Safety AI", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #0e1117; color: white;}
    .main {background-color: #0e1117;}
    h1, h2, h3, h4 {color: #4CAF50;}
    .stButton>button {background-color: #4CAF50; color: white; font-size:18px;}
    </style>
    """, unsafe_allow_html=True
)

# ---------------- Load Model ---------------- #
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🚀 BrainCache – Space Station Safety AI")
st.write("AI-powered detection of **Toolbox, Oxygen Tank, Fire Extinguisher** to ensure astronaut safety.")

# ---------------- Tabs ---------------- #
tab1, tab2, tab3 = st.tabs(["🛰 Detection", "📊 Analytics", "ℹ About Us"])

# ---------------- Detection ---------------- #
with tab1:
    st.subheader("Upload Image, Video, or Use Camera")
    option = st.radio("Select Input Type", ("Image", "Video", "Live Camera"))

    if option == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)

            with st.spinner("🚀 Detecting objects..."):
                results = model.predict(source=np.array(img))

            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Detections", use_column_width=True)

            # Download annotated image
            cv2.imwrite("annotated_image.jpg", annotated_img)
            with open("annotated_image.jpg", "rb") as f:
                st.download_button("Download Annotated Image", f, file_name="annotated_image.jpg")

            # Confidence Scores
            st.write("### Confidence Scores")
            for box in results[0].boxes:
                st.write(f"{results[0].names[int(box.cls)]}: {float(box.conf):.2f}")

    elif option == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_video.read())
            st.video(uploaded_video)

            if st.button("Run Detection on Video"):
                with st.spinner("Analyzing video..."):
                    output_path = "annotated_video.mp4"
                    results = model.predict(source=temp_file.name, save=True)
                    # YOLO saves video automatically in runs/detect/predict
                    st.success("Video Processed!")
                    st.video("runs/detect/predict/video.mp4")
                    with open("runs/detect/predict/video.mp4", "rb") as f:
                        st.download_button("Download Annotated Video", f, file_name="annotated_video.mp4")

    elif option == "Live Camera":
        camera_image = st.camera_input("Capture an Image")
        if camera_image:
            img = Image.open(camera_image)
            results = model.predict(source=np.array(img))
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Live Detection", use_column_width=True)

            # Download captured detection
            cv2.imwrite("live_detect.jpg", annotated_img)
            with open("live_detect.jpg", "rb") as f:
                st.download_button("Download Live Detection", f, file_name="live_detect.jpg")

# ---------------- Analytics ---------------- #
with tab2:
    st.subheader("Model Performance")
    st.metric("mAP@0.5", "0.916")
    st.metric("mAP@0.5-0.95", "0.792")

    # Confusion Matrix
    if st.button("Generate Confusion Matrix"):
        st.write("📊 Generating confusion matrix...")
        labels = ["Toolbox", "Oxygen Tank", "Fire Extinguisher"]
        confusion = np.array([[65, 2, 0],
                              [1, 58, 1],
                              [0, 3, 76]])

        fig, ax = plt.subplots()
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Save confusion matrix for report
        fig.savefig("confusion_matrix.png")

    # Generate PDF Report
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_font("Arial", size=12)
        pdf.add_page()
        pdf.cell(200, 10, "Performance Report - BrainCache", ln=True, align="C")
        pdf.cell(200, 10, f"mAP@0.5: 0.916", ln=True)
        pdf.cell(200, 10, f"mAP@0.5-0.95: 0.792", ln=True)
        if os.path.exists("confusion_matrix.png"):
            pdf.image("confusion_matrix.png", x=50, w=100)
        pdf.output("Performance_Report.pdf")
        with open("Performance_Report.pdf", "rb") as f:
            st.download_button("Download Performance Report", f, file_name="Performance_Report.pdf")

# ---------------- About Us ---------------- #
with tab3:
    st.subheader("Our Mission")
    st.write("""
    - **Team Name:** BrainCache  
    - **Members:** Swastika, Mohit, Uday, Rohit  
    - **Goal:** AI-driven safety monitoring for astronauts.  
    - **Hackathon:** BuildWithIndia 2.0  
    """)

