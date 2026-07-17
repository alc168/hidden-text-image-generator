# Hidden Text Image Generator

Generate images with hidden text using Stable Diffusion and ControlNet. The text is embedded as a low-frequency component in the image, making it visible only when viewed at small sizes or from a distance.

## Features

- Embed custom text in AI-generated images
- Control text visibility with adjustable parameters
- Support for various scene prompts
- Generates both full-size and small preview images

## Requirements

- Python 3.9+
- GPU recommended (CPU will be very slow)

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script to generate an image:
```bash
python generate_hidden_text.py
```

This will generate:
- `jeremy_hidden_landscape.png` - Full-size image with hidden text
- `jeremy_hidden_landscape_small.png` - Small version where text is visible
- `text_mask.png` - Text mask used for ControlNet
- `control_image.png` - Canny edge detection of text mask

## Customization

Edit the `main()` function in `generate_hidden_text.py` to customize:

- `text` - The text to hide (default: "Jeremy")
- `image_size` - Output image dimensions (default: 512x512)
- `prompt` - Scene description for the image
- `controlnet_conditioning_scale` - Text visibility (0.0-1.0, higher = more visible)
- `guidance_scale` - Image quality guidance (default: 7.5)
- `num_inference_steps` - Generation steps (default: 40)

## How It Works

The script uses:
- **Stable Diffusion v1.5** - Text-to-image model
- **ControlNet (Canny)** - Guides the image generation using text edges
- **Low-frequency embedding** - Text is hidden in low-frequency components

When viewing the full-size image, high-frequency details hide the text. When viewing the small version (or squinting), low-frequency components dominate and the text becomes visible.

## Notes

- First run will download models (~5GB)
- GPU is highly recommended for reasonable performance
- CPU generation takes 20+ minutes per image
