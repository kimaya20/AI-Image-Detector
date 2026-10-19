
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.fft_dataset import FFTDataset
from src.fft_model import FFTClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to a training Parquet file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output", default="models/best_fft.pth")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between 0 and 1")

    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    dataset = FFTDataset(args.data, image_size=args.image_size)

    if len(dataset) < 2:
        raise ValueError("Need at least two images to train and validate.")

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small for this validation split.")

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0
    )

    model = FFTClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output_path,
            )
            print("Saved best checkpoint:", output_path)


if __name__ == "__main__":
    main()
