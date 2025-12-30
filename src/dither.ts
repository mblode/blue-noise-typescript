import sharp from "sharp";

interface DitherOptions {
  foreground: string;
  background: string;
  noisePath: string;
  width?: number;
  height?: number;
  contrast?: number;
}

const HEX_COLOR_REGEX = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i;

/**
 * Convert hex color to RGB
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = HEX_COLOR_REGEX.exec(hex);
  if (!result) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  return {
    r: Number.parseInt(result[1], 16),
    g: Number.parseInt(result[2], 16),
    b: Number.parseInt(result[3], 16),
  };
}

/**
 * Modulo wrap function
 */
function wrap(m: number, n: number): number {
  return n % m;
}

/**
 * Load blue noise texture from file
 */
async function loadBlueNoiseTexture(noisePath: string): Promise<{
  data: Buffer;
  width: number;
  height: number;
}> {
  const noiseImage = sharp(noisePath);
  const metadata = await noiseImage.metadata();

  if (!(metadata.width && metadata.height)) {
    throw new Error("Could not determine noise texture dimensions");
  }

  // Convert to grayscale
  const { data } = await noiseImage
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    data,
    width: metadata.width,
    height: metadata.height,
  };
}

/**
 * Apply blue noise dithering to an image
 */
export async function applyBlueNoiseDither(
  inputPath: string,
  outputPath: string,
  options: DitherOptions
): Promise<void> {
  const fg = hexToRgb(options.foreground);
  const bg = hexToRgb(options.background);

  // Load the blue noise texture
  const noise = await loadBlueNoiseTexture(options.noisePath);

  // Load and process the input image
  let image = sharp(inputPath);

  // Resize if width or height specified
  if (options.width || options.height) {
    image = image.resize(options.width, options.height, {
      fit: "inside",
      withoutEnlargement: false,
    });
  }

  // Apply contrast adjustment if specified
  if (options.contrast !== undefined) {
    image = image.linear(options.contrast, -(128 * options.contrast) + 128);
  }

  // Convert input image to grayscale raw pixel data and get actual dimensions
  const { data: imageData, info } = await image
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const width = info.width;
  const height = info.height;

  if (!(width && height)) {
    throw new Error("Could not determine image dimensions");
  }

  // Create output buffer (RGB)
  const output = Buffer.alloc(width * height * 3);

  // Apply dithering using the algorithm from the Rust code
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Wrap coordinates for tiling the noise texture
      const wrapX = wrap(noise.width, x);
      const wrapY = wrap(noise.height, y);

      // Get pixel values
      const noiseIdx = wrapY * noise.width + wrapX;
      const imageIdx = y * width + x;
      const outputIdx = imageIdx * 3;

      const noiseLuma = noise.data[noiseIdx];
      const imageLuma = imageData[imageIdx];

      // Compare: if picture is brighter than noise, use background color
      const isBright = imageLuma > noiseLuma;

      if (isBright) {
        output[outputIdx] = bg.r;
        output[outputIdx + 1] = bg.g;
        output[outputIdx + 2] = bg.b;
      } else {
        output[outputIdx] = fg.r;
        output[outputIdx + 1] = fg.g;
        output[outputIdx + 2] = fg.b;
      }
    }
  }

  // Save the output image
  await sharp(output, {
    raw: {
      width,
      height,
      channels: 3,
    },
  })
    .png()
    .toFile(outputPath);

  console.log(`Dithered image saved to: ${outputPath}`);
}
