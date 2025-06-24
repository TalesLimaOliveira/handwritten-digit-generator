import streamlit as st
st.set_page_config(layout="centered", page_title="Handwritten Digit Generator")

from digit_generator import generate_digit_images

# --- Streamlit UI ---
st.title("Handwritten Digit Generator")
st.markdown('<div style="font-size:16px; margin-bottom:10px; margin-top:-20px;">by <a href="https://github.com/TalesLimaOliveira" target="_blank">Tales Lima Oliveira</a></div>', unsafe_allow_html=True)
st.write("Generate 5 diverse images of a handwritten digit (0-9) similar to the MNIST dataset.")

col1, col2 = st.columns([2,1])
with col1:
    selected_digit = st.selectbox("Select a digit to generate:", options=list(range(10)), key="digit_select")

with col2:
    st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)  # Espaço extra acima do botão
    generate_btn = st.button("Generate Images", key="generate_btn", use_container_width=True)
    st.markdown("""
        <style>
        div[data-testid="stButton"] button {
            background-color: #21ba45 !important;
            color: white !important;
            font-size: 1.2em !important;
            padding: 1 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

if generate_btn:
    with st.spinner('Generating images...'):
        images = generate_digit_images(selected_digit)
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Digit {selected_digit} - Image {i+1}", use_container_width=True)