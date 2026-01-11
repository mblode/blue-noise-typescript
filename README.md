# Blue Noise Dithering

[![npm version](https://img.shields.io/npm/v/blue-noise.svg)](https://www.npmjs.com/package/blue-noise)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A CLI tool to dither images with blue noise, and generate blue noise textures using the void-and-cluster algorithm.

![Dithered profile image](img/matthew-profile-dithered.png)

## What is Blue Noise?

Blue noise distributes pixels evenly, avoiding the clusters and voids of white noise and the repetitive patterns of Bayer dithering. The result is natural-looking dithered images that preserve detail without obvious artefacts.

![Blue noise texture](blue-noise.png)

_64×64 tileable blue noise texture_

## Installation

```bash
# Install globally
npm install -g blue-noise

# Or use directly with npx
npx blue-noise <input-image>
```

## Usage

```bash
# Basic usage
blue-noise <input-image>

# Custom colours
blue-noise <input-image> -f <foreground-hex> -b <background-hex>
```

### Examples

```bash
# Black and white dithering
blue-noise photo.jpg

# Custom colours
blue-noise photo.jpg -f "#ff0000" -b "#ffffff"

# Resize and adjust contrast
blue-noise photo.jpg -w 800 -c 1.2

# Use custom noise texture
blue-noise photo.jpg -n custom-noise.png

# Using npx (no installation required)
npx blue-noise photo.jpg -f "#0000ff" -b "#ffff00"
```

## CLI Options

### Dithering Command

- `<input>` - Path to input image (required)
- `-o, --output <path>` - Output directory (default: "output")
- `-n, --noise <path>` - Path to blue noise texture (default: "./blue-noise.png")
- `-f, --foreground <hex>` - Foreground colour in hex (default: "#000000")
- `-b, --background <hex>` - Background colour in hex (default: "#ffffff")
- `-w, --width <pixels>` - Resize image width
- `-h, --height <pixels>` - Resize image height
- `-c, --contrast <value>` - Adjust contrast (default: 1.0)

### Generate Command

Generate custom blue noise textures:

```bash
blue-noise generate --size 64 --sigma 1.9 --verbose
```

- `-s, --size <pixels>` - Texture size (8-512, default: 64)
- `--sigma <value>` - Gaussian sigma (1.0-3.0, default: 1.9)
- `--seed <number>` - Random seed for reproducibility
- `-v, --verbose` - Show generation progress

**Examples:**

```bash
# Generate 128×128 texture with default settings
blue-noise generate -s 128

# Generate with custom sigma and seed
blue-noise generate -s 64 --sigma 2.1 --seed 42

# Generate with verbose output
blue-noise generate -s 256 -v
```

## How It Works

Each pixel in the input image is compared against the corresponding blue noise threshold value. If brighter than the threshold, use the background colour; if darker, use the foreground colour. The noise texture tiles seamlessly across the image.

## Generating Blue Noise

Uses the **void-and-cluster algorithm** (Ulichney, 1993): identifies clusters and voids using Gaussian blur, then redistributes pixels until evenly spread. Each pixel gets a rank determining its threshold value.

The texture tiles seamlessly using torus topology. Power-of-two dimensions (64×64, 128×128) use FFT optimisation for ~50% faster generation.

**Performance:** 64×64 in ~2-5s, 128×128 in ~30-60s. Pre-generate textures for production use.

## References

- [Void-and-cluster method for dither array generation](https://doi.org/10.1117/12.152707) - Ulichney (1993)
- [Dithering with blue noise](https://doi.org/10.1109/5.3288) - Ulichney (1988)

## Additional Resources

- [Ditherpunk](https://surma.dev/lab/ditherpunk/) - Interactive dithering playground
- [Dithering - Part 1](https://visualrambling.space/dithering-part-1/) - Deep dive into dithering techniques
- [Dither Asteroids](https://dither.blode.co/) - Dithering asteroids game
