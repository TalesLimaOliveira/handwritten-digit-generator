import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Constantes globais
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
NUM_CLASSES = 10
MODEL_FILENAME = "models/generator_final.pth"

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM + NUM_CLASSES, 256, 7, 1, 0, bias=False),
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
    

def load_generator_model():
    model_path = MODEL_FILENAME
    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
    generator.eval()
    return generator

_generator_model = None  # Singleton para evitar recarregamento

def get_generator_model():
    global _generator_model
    if _generator_model is None:
        _generator_model = load_generator_model()
    return _generator_model

def generate_digit_images(digit, num_images=5):
    """Gera imagens sintéticas para um dígito específico."""
    model = get_generator_model()
    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
        labels = torch.full((num_images,), digit, dtype=torch.long, device=DEVICE)
        generated_images = model(noise, labels).cpu()
        generated_images = (generated_images + 1) / 2
        return [Image.fromarray((img.squeeze(0).numpy() * 255).astype(np.uint8), 'L') for img in generated_images]
