
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Device configuration (must match training device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture (must be identical to the trained model's Generator)
latent_dim = 100
num_classes = 10
image_size = 28

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise.view(noise.size(0), -1)), -1)
        gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
        return self.main(gen_input)

# Load the pre-trained generator model
@st.cache_resource # Cache the model loading for efficiency
def load_generator_model(model_path="generator_final.pth"):
    generator = Generator().to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval() # Set to evaluation mode for inference
        st.success(f"Generator model loaded successfully from {model_path}")
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model is trained and saved.")
        st.stop() # Stop execution if model is not found
    return generator

generator_model = load_generator_model()

# Function to generate images for a given digit
def generate_digit_images(digit, num_images=5):
    with torch.no_grad(): # Disable gradient calculations for inference
        # Generate random noise vectors for diversity
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        # Create labels for the specified digit (all images will be of this digit)
        labels = torch.full((num_images,), digit, dtype=torch.long, device=device)

        # Generate images using the model
        generated_images = generator_model(noise, labels).cpu()

        # Denormalize images from [-1, 1] to [0, 1] for display
        generated_images = (generated_images + 1) / 2

        # Convert to PIL images for display in Streamlit
        pil_images = []
        for i in range(num_images):
            img_tensor = generated_images[i].squeeze(0) # Remove channel dimension (1, 28, 28) -> (28, 28)
            img_np = img_tensor.numpy() # Convert to NumPy array
            # Scale to 0-255 and convert to uint8 for PIL
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'L') # 'L' mode for grayscale
            pil_images.append(img_pil)
    return pil_images

# --- Streamlit UI ---
st.set_page_config(layout="centered", page_title="Handwritten Digit Generator")

st.title("Handwritten Digit Generator")
st.write("Generate 5 diverse images of a handwritten digit (0-9) similar to the MNIST dataset.")

# User input for digit selection
selected_digit = st.selectbox("Select a digit to generate:", options=list(range(10)))

# Button to trigger image generation
if st.button("Generate Images"):
    st.write(f"Generating 5 images for digit: **{selected_digit}**...")
    with st.spinner('Generating images...'): # Show a spinner while images are being generated
        images = generate_digit_images(selected_digit)

    # Display images in a grid format
    cols = st.columns(5) # Create 5 columns for the images
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Digit {selected_digit} - Image {i+1}", use_column_width=True)
    st.success("Images generated!")

