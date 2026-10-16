import numpy as np
from PIL import Image


def image_to_fft(image, image_size=256):
    """
    Convert an image into a normalized FFT magnitude map.

    Returns a 2D NumPy array representing spatial
    frequency information.
    """
    if isinstance(image, Image.Image):
        image = image.convert("L")
        image = image.resize((image_size, image_size))
        image = np.asarray(image, dtype=np.float32)
    else:
        image = np.asarray(image, dtype=np.float32)

    # Convert pixel intensities to the frequency domain.
    fft_result = np.fft.fft2(image)

    # Move low frequencies to the center.
    fft_shifted = np.fft.fftshift(fft_result)

    # Log magnitude helps represent the wide range of values.
    magnitude = np.log1p(np.abs(fft_shifted))

    # Normalize to [0, 1].
    magnitude_min = magnitude.min()
    magnitude_max = magnitude.max()

    magnitude = (
        (magnitude - magnitude_min)
        / (magnitude_max - magnitude_min + 1e-8)
    )

    return magnitude.astype(np.float32)
