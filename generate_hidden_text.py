#!/usr/bin/env python3
"""
Generate an image with hidden text using Stable Diffusion and ControlNet.
The text "Jeremy" will be embedded in a mountain landscape, visible only when
viewed at small sizes or from a distance.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from controlnet_aux import CannyDetector


def create_text_mask(text, size=(512, 512), font_size=120):
    """Create a black text on white background mask."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Convert to uppercase for better visibility
    text = text.upper()
    
    # Try to use a bold font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size, index=1)  # Bold variant
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
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
    
    print("Loading models...")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use float16 only on GPU, float32 on CPU
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load ControlNet model (using Canny edge detection for text embedding)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch_dtype
    )
    
    # Load Stable Diffusion model
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch_dtype
    )
    
    pipe = pipe.to(device)
    
    # Enable memory optimizations for CPU
    if device == "cpu":
        pipe.enable_attention_slicing()
    
    print("Creating text mask...")
    # Create text mask
    text_mask = create_text_mask(text, size=image_size)
    text_mask.save("text_mask.png")
    print("Text mask saved as text_mask.png")
    
    print("Processing with ControlNet...")
    # Use Canny detector to get edges from text mask
    canny = CannyDetector()
    control_image = canny(text_mask, low_threshold=50, high_threshold=100)
    control_image.save("control_image.png")
    print("Control image saved as control_image.png")
    
    print("Generating image...")
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
    
    # Save the result
    output_image = output.images[0]
    output_filename = "jeremy_hidden_landscape.png"
    output_image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
    # Also save a smaller version to test visibility
    small_size = (128, 128)
    small_image = output_image.resize(small_size, Image.Resampling.LANCZOS)
    small_filename = "jeremy_hidden_landscape_small.png"
    small_image.save(small_filename)
    print(f"Small version saved as {small_filename}")
    
    print("\nDone! View the small version to see the hidden text.")


if __name__ == "__main__":
    main()
