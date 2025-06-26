import sys
import os
import streamlit as st

# Garante que o diretório src está no sys.path para importação do backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from digit_generator import generate_digit_images



# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Handwritten Digit Generator")
left_col, center_col, right_col = st.columns([1, 3, 1])

# --- Top menu navigation ---
with left_col:
    st.markdown('''
    <div style="text-align:left; margin-bottom:2em; margin-top:0.5em;">
        <span style="color:#888; font-size:1.08em; font-weight:bold;">See other projects:</span><br>
        <a href="https://ocr-taleslimaoliveira.streamlit.app/" style="color:#1a73e8; text-decoration:none; font-size:1.1em; display:block; margin-bottom:0.3em; margin-left:0.7em;">• Character Recognition</a>
        <span style="color:#444; background:#e5e5e5; border-radius:6px; font-size:1.1em; display:inline-block; margin-bottom:0.3em; margin-left:0.7em; padding:2px 8px; width:auto;">• Digit Generator</span>
        <a href="https://github.com/TalesLimaOliveira" target="_blank" style="color:#1a73e8; text-decoration:none; font-size:1.1em; display:block; margin-left:0.7em;">• My Github</a>
    </div>
    ''', unsafe_allow_html=True)

# <a href="https://gen-ocr-taleslimaoliveira.streamlit.app/" style="color:#1a73e8; text-decoration:none; font-size:1.1em; display:block; margin-left:0.7em;">• Gen and OCR</a>

# --- Title and description ---
with center_col:
    st.title("Handwritten Digit Generator")
    st.markdown('<div style="font-size:16px; margin-bottom:0px; margin-top:-16px; text-align:left;">by <a href="https://github.com/TalesLimaOliveira" target="_blank">Tales Lima Oliveira</a></div>', unsafe_allow_html=True)
    st.write("Generate 5 diverse images of a handwritten digit (0-9) similar to the MNIST dataset.")
    st.markdown('<div style="margin-bottom:0.5em; margin-top:1.5em;"></div>', unsafe_allow_html=True)

    data_col = st.columns([1,2,1,1])
    with data_col[1]:
        selected_digit = st.selectbox("Select a digit to generate:", options=list(range(10)), key="digit_select")

    with data_col[2]:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)  # Espaço extra acima do botão
        generate_btn = st.button("Generate Images", key="generate_btn", use_container_width=True)
        st.markdown("""
            <style>
            div[data-testid=\"stButton\"] button {
                background-color: #21ba45 !important;
                color: white !important;
                font-size: 1.2em !important;
                padding: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)

    # Estado para manter as imagens geradas até novo clique
    if 'last_digit' not in st.session_state:
        st.session_state['last_digit'] = None
    if 'last_images' not in st.session_state:
        st.session_state['last_images'] = None

    if generate_btn:
        with st.spinner(f'Generating 5 images for digit: **{selected_digit}**...'):
            images = generate_digit_images(selected_digit)
        st.session_state['last_digit'] = selected_digit
        st.session_state['last_images'] = images

    # Exibe as imagens salvas no estado, se houver
    if st.session_state['last_images'] is not None:
        cols = st.columns(5)
        for i, img in enumerate(st.session_state['last_images']):
            with cols[i]:
                st.image(img, caption=f"Digit {st.session_state['last_digit']} - Image {i+1}", use_container_width=True)