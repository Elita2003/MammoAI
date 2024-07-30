import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import gdown
import os

# Download the models
@st.cache_resource
def download_models():
    gdown.download('https://drive.google.com/uc?id=12TX1C2FlErD44aLEuwQriXWavt0LpA7L', 'Breast-Or-Not-Model.pth', quiet=True)
    gdown.download('https://drive.google.com/uc?id=1f9TQHMSEqr7_5rsPFUH9z3tPLtWc_zpT', 'Breast-Model.pth', quiet=True)

class BreastPipeline:
    def __init__(self, b_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the Breast detection model
        self.b_model = torch.load(b_model_path, map_location=self.device)
        self.b_model = self.b_model.to(self.device)
        self.b_model.eval()

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor()
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def detect_cancer(self, image_tensor):
        with torch.no_grad():
            output = self.b_model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
        return prediction, confidence

    def run(self, image_path):
        image_tensor = self.preprocess(image_path)
        cancer_result, confidence = self.detect_cancer(image_tensor)
        result_text = "Benign" if cancer_result == 0 else "Malignant"
        return f"{result_text} with probability {confidence:.4f}", True

@st.cache_resource
def load_pipeline():
    download_models()
    return BreastPipeline(b_model_path='Breast-Model.pth')

def show_breast_cancer_info():
    st.header("About Breast Cancer")
    st.write("""
    Breast cancer is a type of cancer that forms in the cells of the breasts. It can occur in both women and men, but it's far more common in women.

    Key facts about breast cancer:
    - Breast cancer is the most common cancer in women worldwide.
    - Early detection significantly improves the chances of successful treatment.
    - Regular screenings, including mammograms, are crucial for early detection.
    - Symptoms may include a lump in the breast, changes in breast shape or size, and skin changes.

    Risk factors:
    - Age (risk increases with age)
    - Family history of breast cancer
    - Genetic mutations (BRCA1 and BRCA2)
    - Personal history of breast conditions or cancer
    - Radiation exposure
    - Obesity
    - Alcohol consumption

    Types of breast cancer:
    - Ductal carcinoma in situ (DCIS)
    - Invasive ductal carcinoma
    - Invasive lobular carcinoma
    - Other less common types

    Remember, this app is not a substitute for professional medical advice. If you have concerns about breast cancer, consult a healthcare professional.
    """)

def main():
    st.set_page_config(page_title="Breast Image Analysis", page_icon=":microscope:", layout="wide", initial_sidebar_state="expanded")

    st.title("Breast Image Analysis and Cancer Information")

    # Add a navigation menu
    menu = ["Home", "Analyze Image", "Breast Cancer Information"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Breast Image Analysis app. This tool can help analyze breast images for potential abnormalities.")
        st.write("Please note that this tool is for educational purposes only and should not be used as a substitute for professional medical advice.")
        st.write("Use the menu on the left to navigate to the image analysis tool or to learn more about breast cancer.")

    elif choice == "Analyze Image":
        pipeline = load_pipeline()

        st.header("Breast Image Analysis")
        uploaded_file = st.file_uploader("Choose a breast image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Image"):
                # Save the uploaded file temporarily
                temp_file = "temp_image.jpg"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run the pipeline
                result = pipeline.run(temp_file)
                st.write(result)

        st.warning("Remember: This tool is for educational purposes only. Always consult with a healthcare professional for medical advice.")

    elif choice == "Breast Cancer Information":
        show_breast_cancer_info()

if __name__ == "__main__":
    main()
