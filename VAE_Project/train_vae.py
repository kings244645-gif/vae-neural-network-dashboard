import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from vae_model import VAE


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
LATENT_DIM = 8

DEVICE = torch.device("cpu")

print(f"Training device: {DEVICE}")


# --------------------------------------------------
# 2. LOAD MNIST DATASET
# --------------------------------------------------
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Training images: {len(train_dataset)}")


# --------------------------------------------------
# 3. CREATE MODEL AND OPTIMIZER
# --------------------------------------------------
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 4. VAE LOSS FUNCTION
# --------------------------------------------------
def vae_loss(reconstruction, original, mu, logvar):
    original = original.view(original.size(0), -1)

    reconstruction = reconstruction.view(
        reconstruction.size(0), -1
    )

    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstruction,
        original,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss


# --------------------------------------------------
# 5. STORE TRAINING HISTORY
# --------------------------------------------------
history = {
    "epoch": [],
    "total_loss": [],
    "reconstruction_loss": [],
    "kl_loss": []
}

# --------------------------------------------------
# 6. TRAIN THE VAE
# --------------------------------------------------
for epoch in range(EPOCHS):

    model.train()

    total_loss_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0

    for batch_index, (images, _) in enumerate(train_loader):

        images = images.to(DEVICE)

        optimizer.zero_grad()

        result = model(images)

        loss, reconstruction_loss, kl_loss = vae_loss(
            result["reconstruction"],
            images,
            result["mu"],
            result["logvar"]
        )

        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        reconstruction_sum += reconstruction_loss.item()
        kl_sum += kl_loss.item()

    # Calculate average losses per image.
    dataset_size = len(train_dataset)

    avg_total = total_loss_sum / dataset_size
    avg_reconstruction = reconstruction_sum / dataset_size
    avg_kl = kl_sum / dataset_size

    # Record the current epoch.
    history["epoch"].append(epoch + 1)
    history["total_loss"].append(avg_total)
    history["reconstruction_loss"].append(avg_reconstruction)
    history["kl_loss"].append(avg_kl)

    # Save the CSV after EVERY epoch.
    with open(
        "vae_loss_history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "Epoch",
            "Total Loss",
            "Reconstruction Loss",
            "KL Loss"
        ])

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["total_loss"][i],
                history["reconstruction_loss"][i],
                history["kl_loss"][i]
            ])

    # Display progress.
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Total Loss: {avg_total:.2f} "
        f"| Reconstruction Loss: {avg_reconstruction:.2f} "
        f"| KL Loss: {avg_kl:.2f}"
    )


# --------------------------------------------------
# 7. SAVE TRAINED MODEL AND HISTORY
# --------------------------------------------------
model.eval()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "latent_dim": LATENT_DIM,
        "epochs": EPOCHS,
        "history": history
    },
    "vae_mnist.pth"
)

print("\nTraining completed successfully!")
print("Saved trained model to vae_mnist.pth")
print("Training history saved to vae_loss_history.csv")