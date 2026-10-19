import io
import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.fft_preprocessing import image_to_fft

class FFTDataset(Dataset):

    def __init__(self, parquet_file, image_size=256):
        self.data = pl.read_parquet(parquet_file)
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.row(index, named=True)

        image_bytes = row["image"]["bytes"]
        label = row["label"]

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # image_to_fft converts the image to grayscale internally
        fft_map = image_to_fft(
            image,
            image_size=self.image_size
        )

        # Shape: [1, image_size, image_size]
        fft_tensor = torch.from_numpy(fft_map).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.float32)

        return fft_tensor, label


if __name__ == "__main__":
    print("FFTDataset class imported successfully.")
