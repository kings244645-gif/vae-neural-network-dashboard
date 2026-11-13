import torch
from vae_model import VAE

# Create the VAE
model = VAE(latent_dim=2)

# Simulate one 28 x 28 handwritten digit
x = torch.rand(1, 1, 28, 28)

# Process the image
with torch.no_grad():
    result = model(x)

print("Input shape:", x.shape)
print("Encoder features:", result["features"].shape)
print("Mean (mu):", result["mu"])
print("Standard deviation:", result["std"])
print("Latent point (z):", result["z"])
print("Reconstructed image:", result["reconstruction"].shape)

print("\nVAE model test successful!")