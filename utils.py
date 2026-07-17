"""Shared utilities for the hidden text image generator."""

import logging

import torch
from PIL import ImageFont

logger = logging.getLogger(__name__)

# Font paths tried in order, first match wins. Each entry is (path, index).
DEFAULT_FONT_CANDIDATES = (
    ("/System/Library/Fonts/Helvetica.ttc", 1),  # macOS bold variant
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),  # Linux bold
)


def load_font(font_size, candidates=DEFAULT_FONT_CANDIDATES):
    """Load the first available font from ``candidates``.

    Each candidate is tried in order; the specific font-loading error is logged
    so that a fallback is never silent. Falls back to Pillow's default bitmap
    font if none of the candidates can be loaded (with a warning, since it
    ignores ``font_size`` and produces a poor text mask).
    """
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, font_size, index=index)
        except OSError as exc:
            logger.warning("Could not load font '%s': %s", path, exc)

    logger.warning(
        "No TrueType font available; falling back to Pillow's default bitmap "
        "font. The hidden text will be low quality because font_size (%s) is "
        "ignored.",
        font_size,
    )
    return ImageFont.load_default()


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
