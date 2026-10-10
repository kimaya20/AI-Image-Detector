import torch
from torch.utils.data import DataLoader

from dataset import TinyGenImageDataset


path = r"D:\AI_Datasets\Tiny-GenImage\data\train-00000-of-00014.parquet"

dataset = TinyGenImageDataset(path)


print(
    dataset.data.group_by(["label", "generator"])
    .len()
    .sort(["label", "generator"])
)

print("Label counts:")
print(dataset.data["label"].value_counts())

print("Dataset schema:")
print(dataset.data.schema)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(dataloader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)