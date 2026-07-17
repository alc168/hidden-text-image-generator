"""Shared utilities for the hidden text image generator."""

import torch
from PIL import ImageFont

# Font paths tried in order, first match wins. Each entry is (path, index).
DEFAULT_FONT_CANDIDATES = (
    ("/System/Library/Fonts/Helvetica.ttc", 1),  # macOS bold variant
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),  # Linux bold
)


def load_font(font_size, candidates=DEFAULT_FONT_CANDIDATES):
    """Load the first available font from ``candidates``.

    Falls back to Pillow's default font if none of the candidates can be loaded.
    """
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, font_size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def save_image(image, filename, label="Image"):
    """Save ``image`` to ``filename`` and print a confirmation."""
    image.save(filename)
    print(f"{label} saved as {filename}")


def get_device_and_dtype():
    """Return the best available torch device and matching dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    return device, torch_dtype
