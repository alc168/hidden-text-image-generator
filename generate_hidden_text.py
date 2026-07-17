#!/usr/bin/env python3
"""
Generate an image with hidden text using Stable Diffusion and a brightness
(QR-code-monster) ControlNet.

The text is embedded as a low-frequency luminance pattern in a generated scene,
so it stays hidden when viewed at full size but becomes visible when the image
is shrunk or viewed from a distance / with squinted eyes.
"""

import argparse
import logging
import sys

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw

from utils import get_device_and_dtype, load_font, save_image


logger = logging.getLogger(__name__)

# Model ids. runwayml/stable-diffusion-v1-5 was removed from the Hub in 2024,
# so we use the community-maintained mirror.
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "monster-labs/control_v1p_sd15_qrcode_monster"


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

    device, torch_dtype = get_device_and_dtype()
    logger.info("Using device: %s", device)

    # Load ControlNet + Stable Diffusion models. Downloading/loading these can
    # fail for many reasons (no network, out of disk, corrupt cache); surface a
    # clear, actionable error instead of a raw stack trace deep in diffusers.
    logger.info("Loading models...")
    try:
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch_dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the ControlNet/Stable Diffusion models. Check your "
            "network connection, available disk space, and the Hugging Face "
            "model cache."
        ) from exc

    pipe = pipe.to(device)

    # Attention slicing lowers peak memory; useful on CPU and low-VRAM GPUs.
    pipe.enable_attention_slicing()

    logger.info("Creating text mask...")
    text_mask = create_text_mask(args.text, size=image_size)
    save_image(text_mask, "text_mask.png", label="Text mask")

    logger.info("Generating image...")
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

    if not output.images:
        raise RuntimeError("Pipeline returned no images; generation failed.")

    output_image = output.images[0]
    save_image(output_image, args.output, label="Image")

    # Small version where the hidden text becomes visible.
    small_image = output_image.resize((128, 128), Image.Resampling.LANCZOS)
    stem = args.output.rsplit(".", 1)[0]
    small_filename = f"{stem}_small.png"
    save_image(small_image, small_filename, label="Small version")

    logger.info("Done! View the small version to see the hidden text.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(130)
    except Exception:
        # Log the full traceback and exit non-zero so failures propagate to the
        # caller / shell instead of being masked by a zero exit status.
        logger.exception("Image generation failed.")
        sys.exit(1)
