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
def load_generator_model(model_path="generator_final.pth"):
    generator = Generator().to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval() # Set to evaluation mode for inference
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
    return generator

generator_model = load_generator_model()

def generate_digit_images(digit, num_images=5):
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
        generated_images = generator_model(noise, labels).cpu()
        generated_images = (generated_images + 1) / 2
        pil_images = []
        for i in range(num_images):
            img_tensor = generated_images[i].squeeze(0)
            img_np = img_tensor.numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'L')
            pil_images.append(img_pil)
    return pil_images
