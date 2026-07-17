#!/usr/bin/env python3
"""
Generate an image with hidden text using Stable Diffusion and ControlNet.
The text "Jeremy" will be embedded in a mountain landscape, visible only when
viewed at small sizes or from a distance.
"""

import logging
import sys

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw
from controlnet_aux import CannyDetector

from utils import get_device_and_dtype, load_font, save_image


logger = logging.getLogger(__name__)


def create_text_mask(text, size=(512, 512), font_size=120):
    """Create a black text on white background mask."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Convert to uppercase for better visibility
    text = text.upper()

    font = load_font(font_size)

    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw text in black with stroke for thickness
    draw.text((x, y), text, font=font, fill='black', stroke_width=3, stroke_fill='black')

    return img


def main():
    # Configuration
    text = "Jeremy"
    image_size = (512, 512)
    prompt = "beautiful mountain landscape, majestic peaks, scenic view, high quality, detailed, photorealistic"
    negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

    # ControlNet strength (lower = more subtle)
    controlnet_conditioning_scale = 0.8
    guidance_scale = 7.5
    num_inference_steps = 40

    logger.info("Loading models...")

    # Move to GPU if available, using float16 on GPU and float32 on CPU
    device, torch_dtype = get_device_and_dtype()
    logger.info("Using device: %s", device)

    # Load ControlNet + Stable Diffusion models. Downloading/loading these can
    # fail for many reasons (no network, out of disk, corrupt cache); surface a
    # clear, actionable error instead of a raw stack trace deep in diffusers.
    try:
        # ControlNet using Canny edge detection for text embedding.
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch_dtype
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch_dtype
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the ControlNet/Stable Diffusion models. Check your "
            "network connection, available disk space, and the Hugging Face "
            "model cache."
        ) from exc

    pipe = pipe.to(device)

    # Enable memory optimizations for CPU
    if device == "cpu":
        pipe.enable_attention_slicing()

    logger.info("Creating text mask...")
    # Create text mask
    text_mask = create_text_mask(text, size=image_size)
    save_image(text_mask, "text_mask.png", label="Text mask")

    logger.info("Processing with ControlNet...")
    # Use Canny detector to get edges from text mask
    canny = CannyDetector()
    control_image = canny(text_mask, low_threshold=50, high_threshold=100)
    save_image(control_image, "control_image.png", label="Control image")

    logger.info("Generating image...")
    # Generate the image
    generator = torch.Generator(device).manual_seed(42)  # For reproducibility

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=image_size[1],
        width=image_size[0]
    )

    if not output.images:
        raise RuntimeError("Pipeline returned no images; generation failed.")

    # Save the result
    output_image = output.images[0]
    save_image(output_image, "jeremy_hidden_landscape.png", label="Image")

    # Also save a smaller version to test visibility
    small_size = (128, 128)
    small_image = output_image.resize(small_size, Image.Resampling.LANCZOS)
    save_image(small_image, "jeremy_hidden_landscape_small.png", label="Small version")

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
