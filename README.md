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
- `hidden_landscape.png` - Full-size image with hidden text
- `hidden_landscape_small.png` - Small version where text is visible
- `text_mask.png` - Brightness mask used for ControlNet

## Customization

All options are exposed as command-line flags:

```bash
python generate_hidden_text.py \
  --text "Jeremy" \
  --size 512 \
  --prompt "beautiful mountain landscape, majestic peaks" \
  --controlnet-scale 1.2 \
  --guidance-scale 7.5 \
  --steps 40 \
  --seed 42 \
  --output hidden_landscape.png
```

Run `python generate_hidden_text.py --help` for the full list.

- `--text` - The text to hide (default: "Jeremy")
- `--size` - Square output size in pixels (default: 512)
- `--prompt` - Scene description for the image
- `--controlnet-scale` - Text visibility (higher = more visible; ~1.0-1.5)
- `--guidance-scale` - Image quality guidance (default: 7.5)
- `--steps` - Generation steps (default: 40)
- `--seed` - Random seed for reproducibility (default: 42)

## How It Works

The script uses:
- **Stable Diffusion v1.5** - Text-to-image model
- **ControlNet (QR-code-monster / brightness)** - Modulates image luminance from a filled text mask
- **Low-frequency embedding** - Text is hidden in low-frequency luminance components

When viewing the full-size image, high-frequency details hide the text. When viewing the small version (or squinting), low-frequency components dominate and the text becomes visible.

## Notes

- First run will download models (~5GB)
- GPU is highly recommended for reasonable performance
- CPU generation takes 20+ minutes per image
