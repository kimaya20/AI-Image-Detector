
import matplotlib.pyplot as plt
from PIL import Image

from src.fft_preprocessing import image_to_fft

# Load your saved image
image = Image.open("tests/sample.jpg").convert("RGB")

# Generate FFT magnitude map
fft_image = image_to_fft(image)

# Compare original and frequency representations
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(fft_image, cmap="gray")
axes[1].set_title("FFT Magnitude Map")
axes[1].axis("off")

plt.tight_layout()
plt.show()
