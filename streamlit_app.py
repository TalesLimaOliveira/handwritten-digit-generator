import streamlit as st
st.set_page_config(layout="centered", page_title="Handwritten Digit Generator")

from backend import generate_digit_images

# --- Streamlit UI ---
st.title("Handwritten Digit Generator")
st.write("Generate 5 diverse images of a handwritten digit (0-9) similar to the MNIST dataset.")

col1, col2 = st.columns([2,1])
with col1:
    selected_digit = st.selectbox("Select a digit to generate:", options=list(range(10)), key="digit_select")
with col2:
    generate_btn = st.button("Generate Images", key="generate_btn", use_container_width=True)
    st.markdown("""
        <style>
        div[data-testid="stButton"] button {
            background-color: #21ba45 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

if generate_btn:
    st.write(f"Generating 5 images for digit: **{selected_digit}**...")
    with st.spinner('Generating images...'):
        images = generate_digit_images(selected_digit)
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Digit {selected_digit} - Image {i+1}", use_container_width=True)
    st.success("Images generated!")