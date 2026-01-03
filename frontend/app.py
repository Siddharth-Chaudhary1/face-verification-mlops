import streamlit as st
import requests
import base64

API_URL = "http://localhost:8000/compare"

def encode_image(file):
    return base64.b64encode(file.read()).decode("utf-8")

st.set_page_config(page_title="Face Verification", layout="centered")

st.title("🔍 Face Verification System")
st.write("Upload two face images to verify if they belong to the same person.")

img1 = st.file_uploader("Upload Image 1", type=["jpg", "png"])
img2 = st.file_uploader("Upload Image 2", type=["jpg", "png"])

threshold = st.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01
)

if img1 and img2:
    if st.button("Compare Faces"):
        payload = {
            "image1": encode_image(img1),
            "image2": encode_image(img2),
            "threshold": threshold
        }

        with st.spinner("Comparing faces..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Comparison complete!")
            st.json(result)

            if result["is_match"]:
                st.markdown("### ✅ Same Person")
            else:
                st.markdown("### ❌ Different Persons")
        else:
            st.error("API error. Check backend logs.")
