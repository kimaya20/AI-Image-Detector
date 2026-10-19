import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.fft_dataset import FFTDataset
from src.fft_model import FFTClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to the evaluation Parquet file")
    parser.add_argument("--checkpoint", default="models/best_fft.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )

    model = FFTClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = FFTDataset(
        args.data, image_size=checkpoint["image_size"]
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = (torch.sigmoid(logits).squeeze(1) >= 0.5)

            all_labels.extend(labels.to(torch.int64).tolist())
            all_predictions.extend(
                predictions.to(torch.int64).cpu().tolist()
            )

    print("Accuracy:", accuracy_score(all_labels, all_predictions))
    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["Real", "AI-generated"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
