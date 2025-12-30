#!/usr/bin/env node

import path from "node:path";
import { Command } from "commander";
import { applyBlueNoiseDither } from "./dither.js";
import { BlueNoiseGenerator, saveBlueNoiseToPNG } from "./generator.js";

interface DitherOptions {
  output: string;
  noise: string;
  foreground: string;
  background: string;
  width?: string;
  height?: string;
  contrast?: string;
}

interface GenerateOptions {
  size: string;
  output: string;
  sigma: string;
  seed?: string;
  verbose: boolean;
}

const FILE_EXTENSION_REGEX = /\.[^.]+$/;

const program = new Command();

program
  .name("blue-noise")
  .description("Blue noise dithering and generation tools")
  .version("1.0.0");

// Dither command (default behavior)
program
  .command("dither", { isDefault: true })
  .description("Apply blue noise dithering to an image")
  .argument("<input>", "Path to input image")
  .option("-o, --output <path>", "Output directory", "output")
  .option(
    "-n, --noise <path>",
    "Path to blue noise texture",
    "./blue-noise.png"
  )
  .option("-f, --foreground <hex>", "Foreground color (hex)", "#000000")
  .option("-b, --background <hex>", "Background color (hex)", "#ffffff")
  .option("-w, --width <number>", "Output width in pixels")
  .option("-h, --height <number>", "Output height in pixels")
  .option(
    "-c, --contrast <number>",
    "Contrast adjustment (1.0 = normal, >1 = more contrast, <1 = less contrast)"
  )
  .action(async (input: string, options: DitherOptions) => {
    try {
      const inputPath = path.resolve(input);
      const outputDir = path.resolve(options.output);
      const noisePath = path.resolve(options.noise);
      const inputFilename = path.basename(input);
      const outputFilename = inputFilename.replace(
        FILE_EXTENSION_REGEX,
        "-dithered.png"
      );
      const outputPath = path.join(outputDir, outputFilename);

      const width = options.width
        ? Number.parseInt(options.width, 10)
        : undefined;
      const height = options.height
        ? Number.parseInt(options.height, 10)
        : undefined;
      const contrast = options.contrast
        ? Number.parseFloat(options.contrast)
        : undefined;

      console.log(`Processing: ${inputPath}`);
      console.log(`Noise texture: ${noisePath}`);
      console.log(`Output: ${outputPath}`);
      if (width || height) {
        console.log(`Dimensions: ${width || "auto"}x${height || "auto"}`);
      }
      if (contrast !== undefined) {
        console.log(`Contrast: ${contrast}`);
      }
      console.log(`Foreground: ${options.foreground}`);
      console.log(`Background: ${options.background}`);

      await applyBlueNoiseDither(inputPath, outputPath, {
        foreground: options.foreground,
        background: options.background,
        noisePath,
        width,
        height,
        contrast,
      });

      console.log("Done!");
    } catch (error) {
      console.error("Error:", error instanceof Error ? error.message : error);
      process.exit(1);
    }
  });

// Generate command
program
  .command("generate")
  .description(
    "Generate a blue noise texture using the void-and-cluster algorithm"
  )
  .option("-s, --size <number>", "Texture size (width and height)", "128")
  .option("-o, --output <path>", "Output file path", "./blue-noise.png")
  .option(
    "--sigma <number>",
    "Gaussian sigma value (1.5-1.9, higher = more spread)",
    "1.9"
  )
  .option("--seed <number>", "Random seed for reproducibility")
  .option("-v, --verbose", "Show detailed generation progress", false)
  .action(async (options: GenerateOptions) => {
    try {
      const size = Number.parseInt(options.size, 10);
      const sigma = Number.parseFloat(options.sigma);
      const seed = options.seed ? Number.parseInt(options.seed, 10) : undefined;
      const outputPath = path.resolve(options.output);

      // Validate inputs
      if (Number.isNaN(size) || size < 8 || size > 512) {
        throw new Error("Size must be between 8 and 512");
      }
      if (Number.isNaN(sigma) || sigma < 1.0 || sigma > 3.0) {
        throw new Error("Sigma must be between 1.0 and 3.0");
      }

      if (!options.verbose) {
        console.log(`Generating ${size}×${size} blue noise texture`);
        console.log(`Sigma: ${sigma}`);
        if (seed !== undefined) {
          console.log(`Seed: ${seed}`);
        }
        console.log(`Output: ${outputPath}`);
        console.log("");
      }

      // Generate blue noise
      const generator = new BlueNoiseGenerator({
        width: size,
        height: size,
        sigma,
        seed,
        verbose: options.verbose,
      });

      const result = generator.generate();

      // Save to file
      await saveBlueNoiseToPNG(result, outputPath);

      console.log("");
      console.log("Done!");
    } catch (error) {
      console.error("Error:", error instanceof Error ? error.message : error);
      process.exit(1);
    }
  });

program.parse();
