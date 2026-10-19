import torch
import torch.nn as nn

class FFTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = FFTClassifier()
    dummy_images = torch.randn(8, 1, 256, 256)
    outputs = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad)
    )
