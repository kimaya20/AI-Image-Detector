import io

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset


class TinyGenImageDataset(Dataset):

    def __init__(self, parquet_file, image_size=256):
        self.data = pl.read_parquet(parquet_file)
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.row(index, named=True)

        image_bytes = row["image"]["bytes"]
        label = row["label"]

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        image = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        )

        image = image.reshape(self.image_size, self.image_size, 3)
        image = image.permute(2, 0, 1)
        image = image / 255.0

        label = torch.tensor(label, dtype=torch.float32)

        return image, label

if __name__ == "__main__":

    path = r"D:\AI_Datasets\Tiny-GenImage\data\train-00000-of-00014.parquet"

    dataset = TinyGenImageDataset(path)

    image, label = dataset[0]

    print("Image shape:", image.shape)
    print("Label:", label)
    print("Pixel range:", image.min().item(), "to", image.max().item())