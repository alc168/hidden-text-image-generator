"""Shared utilities for the hidden text image generator."""

import logging

import torch
from PIL import ImageFont

logger = logging.getLogger(__name__)

# Font paths tried in order, first match wins. Each entry is (path, index).
DEFAULT_FONT_CANDIDATES = (
    ("/System/Library/Fonts/Helvetica.ttc", 1),  # macOS bold variant
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),  # Debian/Ubuntu
    ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 0),  # Fedora/others
    ("C:\\Windows\\Fonts\\arialbd.ttf", 0),  # Windows
    ("DejaVuSans-Bold.ttf", 0),  # bundled with Pillow
)


def load_font(font_size, candidates=DEFAULT_FONT_CANDIDATES):
    """Load the first available scalable font from ``candidates``.

    Each candidate is tried in order; the specific font-loading error is logged
    so that a fallback is never silent. Raises if none can be loaded, because
    Pillow's default bitmap font ignores ``font_size`` and produces an unusably
    tiny mask.
    """
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, font_size, index=index)
        except OSError as exc:
            logger.warning("Could not load font '%s': %s", path, exc)

    raise RuntimeError(
        "Could not load a scalable TrueType font. Install DejaVu "
        "(e.g. `apt-get install fonts-dejavu`) or provide a font available on this system."
    )


def save_image(image, filename, label="Image"):
    """Save ``image`` to ``filename``, raising a clear error if the write fails."""
    try:
        image.save(filename)
    except OSError as exc:
        raise OSError(f"Failed to save {label.lower()} '{filename}': {exc}") from exc
    logger.info("%s saved as %s", label, filename)


def get_device_and_dtype():
    """Return the best available torch device and matching dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    return device, torch_dtype
