"""Unit tests for ``generate_hidden_text.create_text_mask``."""

import numpy as np
import pytest
from PIL import Image


def _count_black_pixels(img):
    """Return the number of (near-)black pixels in an RGB image."""
    arr = np.asarray(img)
    return int(np.count_nonzero(arr.sum(axis=2) < 30))


def test_returns_rgb_pil_image(generate_module):
    img = generate_module.create_text_mask("Jeremy")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_default_size_is_512(generate_module):
    img = generate_module.create_text_mask("Jeremy")
    assert img.size == (512, 512)


@pytest.mark.parametrize("size", [(128, 128), (256, 512), (640, 480)])
def test_custom_size_is_respected(generate_module, size):
    img = generate_module.create_text_mask("Hi", size=size)
    assert img.size == size


def test_background_is_white(generate_module):
    img = generate_module.create_text_mask("Jeremy")
    # Corners should never be covered by centered text.
    for corner in [(0, 0), (img.width - 1, 0), (0, img.height - 1)]:
        assert img.getpixel(corner) == (255, 255, 255)


def test_text_is_drawn_in_black(generate_module):
    img = generate_module.create_text_mask("Jeremy")
    assert _count_black_pixels(img) > 0


def test_empty_string_produces_blank_mask(generate_module):
    img = generate_module.create_text_mask("")
    assert isinstance(img, Image.Image)
    assert _count_black_pixels(img) == 0


def test_larger_font_size_yields_more_ink(generate_module):
    small = generate_module.create_text_mask("W", font_size=40)
    large = generate_module.create_text_mask("W", font_size=200)
    assert _count_black_pixels(large) > _count_black_pixels(small)


def test_text_is_uppercased(generate_module):
    # Lowercase and uppercase input should render identically because the
    # function uppercases its input before drawing.
    lower = generate_module.create_text_mask("jeremy")
    upper = generate_module.create_text_mask("JEREMY")
    assert list(np.asarray(lower).ravel()) == list(np.asarray(upper).ravel())


def test_font_fallback_when_truetype_unavailable(generate_module, monkeypatch):
    """If no TrueType font can be loaded, it falls back to the default font."""
    real_truetype = generate_module.ImageFont.truetype

    def _boom(font=None, *args, **kwargs):
        # Simulate the named font files being unavailable, but still allow
        # Pillow's own ``load_default`` (which loads a bundled font from a
        # file-like object) to succeed.
        if isinstance(font, str):
            raise OSError("no font")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(generate_module.ImageFont, "truetype", _boom)
    img = generate_module.create_text_mask("Jeremy")
    assert isinstance(img, Image.Image)
    assert img.size == (512, 512)
