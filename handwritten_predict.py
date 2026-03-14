import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=33):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path: str, device: str):

    model = SimpleCNN(num_classes=1)  # russian alphabet size
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    return model


def predict_image(image_path: str, model, inv_label_map, device: str):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("L")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_letter.py path/to/image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = load_model(
        model_path="russian_letters_cnn.pth",
        device=device,
    )

    pred, confidence = predict_image(...)
    print("Predicted class:", pred)


if __name__ == "__main__":
    main()