import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="AI Face Comparison", layout="centered")

st.title("🧠 AI Face Comparison Tool")
st.markdown(
    "Check if two faces match using AI (FaceNet).\n\n"
    "**100% private – no server upload.**"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image 1")
    img1 = st.file_uploader("Choose Image 1", type=["jpg", "jpeg", "png"], key="img1")
    if img1:
        image1 = Image.open(img1)
        st.image(image1, caption="Uploaded Image 1", use_container_width=True)  # Updated parameter

with col2:
    st.subheader("Image 2")
    img2 = st.file_uploader("Choose Image 2", type=["jpg", "jpeg", "png"], key="img2")
    if img2:
        image2 = Image.open(img2)
        st.image(image2, caption="Uploaded Image 2", use_container_width=True)  # Updated parameter

if img1 and img2:
    if st.button("🔍 Compare Faces"):
        with st.spinner("Comparing..."):
            try:
                # Prepare files with proper metadata
                files = {
                    "user_img": (img1.name, img1.getvalue(), "image/jpeg"),
                    "celeb_img": (img2.name, img2.getvalue(), "image/jpeg")
                }

                response = requests.post(
                    "http://localhost:8000/compare",
                    files=files,
                    timeout=15
                )

                response.raise_for_status()
                result = response.json()

                if "similarity" in result:
                    st.success(f"Face Similarity: **{result['similarity']}%**")
                    st.metric("Match Score", f"{result['similarity']}%")
                else:
                    st.error(result.get("detail", "Unknown response format"))

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
                if hasattr(e, 'response') and e.response:
                    st.text(f"Server response: {e.response.text}")
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                st.text(f"Full response: {response.text if 'response' in locals() else 'No response'}")