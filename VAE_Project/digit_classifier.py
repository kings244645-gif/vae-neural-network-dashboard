import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

DEVICE = torch.device("cpu")
MODEL_PATH = "digit_classifier.pth"


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


def train_classifier(epochs=5):
    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    model = DigitClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = 100 * total_correct / total_samples

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Loss: {total_loss / len(loader):.4f} "
            f"| Training Accuracy: {accuracy:.2f}%"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nClassifier saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("A saved classifier already exists.")
        print("Delete it only if you want to train a new classifier.")
    else:
        train_classifier(epochs=5)