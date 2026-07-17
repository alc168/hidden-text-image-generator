#!/usr/bin/env python3
"""
Generate an image with hidden text using Stable Diffusion and a brightness
(QR-code-monster) ControlNet.

The text is embedded as a low-frequency luminance pattern in a generated scene,
so it stays hidden when viewed at full size but becomes visible when the image
is shrunk or viewed from a distance / with squinted eyes.
"""

import argparse
import sys

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw, ImageFont


# Model ids. runwayml/stable-diffusion-v1-5 was removed from the Hub in 2024,
# so we use the community-maintained mirror.
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "monster-labs/control_v1p_sd15_qrcode_monster"

# Candidate bold fonts, tried in order across macOS / Linux / Windows.
FONT_CANDIDATES = [
    ("/System/Library/Fonts/Helvetica.ttc", 1),  # macOS bold variant
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),  # Debian/Ubuntu
    ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 0),  # Fedora/others
    ("C:\\Windows\\Fonts\\arialbd.ttf", 0),  # Windows
    ("DejaVuSans-Bold.ttf", 0),  # bundled with Pillow
]


def load_font(font_size):
    """Load a bold TrueType font at the requested size.

    Falls back through several well-known system paths and finally to the font
    bundled with Pillow. Raises if no scalable font can be loaded, because the
    default bitmap font ignores ``font_size`` and produces an unusable mask.
    """
    for path, index in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size, index=index)
        except OSError:
            continue
    raise RuntimeError(
        "Could not load a scalable TrueType font. Install DejaVu "
        "(e.g. `apt-get install fonts-dejavu`) or pass a font available on this system."
    )


def create_text_mask(text, size=(512, 512), font_size=180, stroke_width=6):
    """Create a filled white-text-on-black mask for the brightness ControlNet.

    Bright (white) regions steer the model toward brighter output and dark
    regions toward darker output, which is what forms the hidden pattern.
    """
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)

    text = text.upper()
    font = load_font(font_size)

    # Shrink the font until the (stroked) text fits within the image.
    while font_size > 8:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= size[0] and text_height <= size[1]:
            break
        font_size -= 8
        font = load_font(font_size)

    # Center accounting for the font's origin offset (bbox[0], bbox[1]).
    x = (size[0] - text_width) // 2 - bbox[0]
    y = (size[1] - text_height) // 2 - bbox[1]

    draw.text(
        (x, y),
        text,
        font=font,
        fill="white",
        stroke_width=stroke_width,
        stroke_fill="white",
    )

    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image with hidden text using Stable Diffusion + ControlNet."
    )
    parser.add_argument("--text", default="Jeremy", help="Text to hide in the image.")
    parser.add_argument("--size", type=int, default=512, help="Square image size in pixels.")
    parser.add_argument(
        "--prompt",
        default="beautiful mountain landscape, majestic peaks, scenic view, high quality, detailed, photorealistic",
        help="Scene description for the image.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted, ugly, bad anatomy",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=1.2,
        help="ControlNet conditioning scale (higher = more visible text).",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG guidance scale.")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output", default="hidden_landscape.png", help="Output image filename."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.size, args.size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # float16 only helps on GPU; CPU requires float32.
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading models...")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    # Attention slicing lowers peak memory; useful on CPU and low-VRAM GPUs.
    pipe.enable_attention_slicing()

    print("Creating text mask...")
    text_mask = create_text_mask(args.text, size=image_size)
    text_mask.save("text_mask.png")
    print("Text mask saved as text_mask.png")

    print("Generating image...")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=text_mask,
        controlnet_conditioning_scale=args.controlnet_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
        height=image_size[1],
        width=image_size[0],
    )

    output_image = output.images[0]
    output_image.save(args.output)
    print(f"Image saved as {args.output}")

    # Small version where the hidden text becomes visible.
    small_image = output_image.resize((128, 128), Image.Resampling.LANCZOS)
    stem = args.output.rsplit(".", 1)[0]
    small_filename = f"{stem}_small.png"
    small_image.save(small_filename)
    print(f"Small version saved as {small_filename}")

    print("\nDone! View the small version to see the hidden text.")


if __name__ == "__main__":
    sys.exit(main())
