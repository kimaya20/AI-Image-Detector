
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TinyGenImageDataset
from model import CNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = (
    r"D:\AI_Datasets\Tiny-GenImage\data"
    r"\train-00000-of-00014.parquet"
)

BATCH_SIZE = 8       # Start small because you are using CPU
EPOCHS = 5
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MODEL_PATH = "best_cnn.pth"

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. LOAD DATASET
# ==========================================

dataset = TinyGenImageDataset(DATA_PATH)

total_size = len(dataset)

train_size = int(TRAIN_RATIO * total_size)
val_size = int(VAL_RATIO * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

print("\nDataset split:")
print("Total images:", total_size)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))


# ==========================================
# 3. CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==========================================
# 4. INITIALIZE MODEL
# ==========================================

model = CNN().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("\nModel initialized successfully!")


# ==========================================
# 5. TRAINING AND VALIDATION
# ==========================================

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # ---------- TRAIN ----------
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)

        # Logit >= 0 means predicted class 1
        predictions = (outputs.detach() >= 0).float()

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.numel()

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total


    # ---------- VALIDATE ----------
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)

            predictions = (outputs >= 0).float()

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.numel()

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total


    # ---------- PRINT RESULTS ----------
    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f"\nTrain Loss: {train_loss:.4f}"
        f" | Train Accuracy: {train_accuracy:.2%}"
        f"\nValidation Loss: {val_loss:.4f}"
        f" | Validation Accuracy: {val_accuracy:.2%}"
    )


    # ---------- SAVE BEST MODEL ----------
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print("Best model saved!")


# ==========================================
# 6. LOAD BEST MODEL
# ==========================================

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("\nTraining complete!")
print("Best validation loss:", round(best_val_loss, 4))


# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

test_loss_sum = 0.0
test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.float().to(device).view(-1, 1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss_sum += loss.item() * images.size(0)

        predictions = (outputs >= 0).float()

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.numel()

        all_predictions.extend(
            predictions.cpu().view(-1).tolist()
        )

        all_labels.extend(
            labels.cpu().view(-1).tolist()
        )

test_loss = test_loss_sum / test_total
test_accuracy = test_correct / test_total

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

# Class 0 = Real
# Class 1 = AI-generated

tn = sum(
    p == 0 and y == 0
    for p, y in zip(all_predictions, all_labels)
)

fp = sum(
    p == 1 and y == 0
    for p, y in zip(all_predictions, all_labels)
)

fn = sum(
    p == 0 and y == 1
    for p, y in zip(all_predictions, all_labels)
)

tp = sum(
    p == 1 and y == 1
    for p, y in zip(all_predictions, all_labels)
)

print("\nConfusion Matrix:")
print("                 Predicted Real  Predicted AI")
print(f"Actual Real      {tn:13d}  {fp:12d}")
print(f"Actual AI        {fn:13d}  {tp:12d}")

print("\nClass mapping:")
print("0 = Real")
print("1 = AI-generated")

print("\nSaved model:", MODEL_PATH)
