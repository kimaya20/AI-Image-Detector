import numpy as np
from PIL import Image

from src.fft_preprocessing import image_to_fft


def test_fft_output():
    image = Image.fromarray(
        np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    )

    result = image_to_fft(image)

    assert result.shape == (256, 256)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1


if __name__ == "__main__":
    test_fft_output()
    print("FFT preprocessing test passed!")