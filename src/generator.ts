/**
 * Blue Noise Texture Generator using Void-and-Cluster Algorithm
 *
 * Implementation of Robert Ulichney's void-and-cluster method for generating
 * high-quality blue noise textures. Blue noise has evenly distributed energy
 * at high frequencies whilst minimising low-frequency content, avoiding the
 * clustering and void patterns found in white noise.
 *
 * ALGORITHM OVERVIEW
 * ==================
 *
 * Blue noise addresses issues with other dithering methods:
 * - White noise: Random distribution creates visible clusters and voids
 * - Bayer dithering: Regular patterns create repetitive artefacts
 * - Blue noise: Evenly distributed with minimal low-frequency patterns
 *
 * The void-and-cluster algorithm works by:
 * 1. Finding clusters (areas of high density) using Gaussian blur
 * 2. Finding voids (areas of low density) using the same blur
 * 3. Iteratively redistributing points to spread them evenly
 * 4. Assigning each pixel a rank based on its importance
 *
 * TORUS TOPOLOGY
 * ==============
 * All distance calculations wrap around at the edges (toroidal topology),
 * ensuring the resulting texture tiles seamlessly when repeated. This is
 * essential for dithering large images with small noise textures.
 *
 * FFT OPTIMISATION
 * ================
 * For power-of-two dimensions, Gaussian blur is performed in the frequency
 * domain using Fast Fourier Transform. Convolution becomes element-wise
 * multiplication in frequency space, providing ~50% performance improvement.
 *
 * GENERATION PHASES
 * =================
 *
 * Phase 0: Generate initial binary pattern
 *   - Place random points and redistribute until convergence
 *   - Convergence occurs when tightest cluster equals largest void
 *
 * Phase 1: Serialize initial points
 *   - Remove points from tightest clusters
 *   - Assign ranks from (initialPoints - 1) down to 0
 *
 * Phase 2: Fill to half capacity
 *   - Restore initial pattern and add points to largest voids
 *   - Assign ranks from initialPoints to area/2
 *
 * Phase 3: Fill to completion
 *   - Invert bitmap (0s become minority)
 *   - Remove minority points from tightest clusters
 *   - Assign ranks from area/2 to area-1
 *
 * Phase 4: Convert to threshold map
 *   - Map ranks [0, area-1] to threshold values [0, 255]
 *
 * REFERENCES
 * ==========
 * - Ulichney, R. (1993). "Void-and-cluster method for dither array generation"
 *   Proceedings of SPIE 1913, Human Vision, Visual Processing, and Digital
 *   Display IV. https://doi.org/10.1117/12.152707
 *
 * - Ulichney, R. (1988). "Dithering with blue noise"
 *   Proceedings of the IEEE, 76(1), 56-79.
 *
 * PERFORMANCE
 * ===========
 * 64×64 texture:  ~2-5 seconds
 * 128×128 texture: ~30-60 seconds
 * 256×256 texture: Several minutes
 *
 * For production use, pre-generate textures rather than generating at runtime.
 *
 * @author Implementation based on Ulichney's original research
 */

/**
 * Configuration for blue noise generation
 */
export interface BlueNoiseConfig {
  width: number;
  height: number;
  sigma?: number;
  initialDensity?: number;
  seed?: number;
  verbose?: boolean;
}

/**
 * Result of blue noise generation
 */
export interface BlueNoiseResult {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/**
 * Mulberry32 seeded random number generator
 * Fast, high-quality PRNG for reproducible results
 */
class SeededRandom {
  private seed: number;

  constructor(seed?: number) {
    this.seed = seed ?? Date.now();
  }

  next(): number {
    // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operations are intentional for Mulberry32 PRNG algorithm
    this.seed = (this.seed + 0x6d_2b_79_f5) | 0;
    // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operations are intentional for Mulberry32 PRNG algorithm
    let t = Math.imul(this.seed ^ (this.seed >>> 15), 1 | this.seed);
    // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operations are intentional for Mulberry32 PRNG algorithm
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operations are intentional for Mulberry32 PRNG algorithm
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  }
}

/**
 * Complex number for FFT calculations
 */
class Complex {
  real: number;
  imag: number;

  constructor(real: number, imag: number) {
    this.real = real;
    this.imag = imag;
  }

  add(other: Complex): Complex {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }

  sub(other: Complex): Complex {
    return new Complex(this.real - other.real, this.imag - other.imag);
  }

  mul(other: Complex): Complex {
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real
    );
  }

  magnitude(): number {
    return Math.sqrt(this.real * this.real + this.imag * this.imag);
  }
}

/**
 * In-place Cooley-Tukey FFT implementation
 * Optimized for power-of-two sizes
 */
class FFT {
  private readonly size: number;
  private readonly bitReversedIndices: number[];

  constructor(size: number) {
    if (!this.isPowerOfTwo(size)) {
      throw new Error("FFT size must be a power of two");
    }
    this.size = size;
    this.bitReversedIndices = this.computeBitReversedIndices();
  }

  private isPowerOfTwo(n: number): boolean {
    // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operation is intentional for power-of-two check
    return n > 0 && (n & (n - 1)) === 0;
  }

  private computeBitReversedIndices(): number[] {
    const bits = Math.log2(this.size);
    const indices = new Array(this.size);

    for (let i = 0; i < this.size; i++) {
      let reversed = 0;
      for (let j = 0; j < bits; j++) {
        // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operations are intentional for bit-reversal algorithm
        if (i & (1 << j)) {
          // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operations are intentional for bit-reversal algorithm
          reversed |= 1 << (bits - 1 - j);
        }
      }
      indices[i] = reversed;
    }

    return indices;
  }

  /**
   * Perform in-place FFT on complex array
   */
  forward(data: Complex[]): void {
    // Bit-reversal permutation
    for (let i = 0; i < this.size; i++) {
      const j = this.bitReversedIndices[i];
      if (i < j) {
        [data[i], data[j]] = [data[j], data[i]];
      }
    }

    // Cooley-Tukey decimation-in-time
    for (let len = 2; len <= this.size; len *= 2) {
      const halfLen = len / 2;
      const angle = (-2 * Math.PI) / len;

      for (let i = 0; i < this.size; i += len) {
        for (let j = 0; j < halfLen; j++) {
          const w = new Complex(Math.cos(angle * j), Math.sin(angle * j));

          const u = data[i + j];
          const v = data[i + j + halfLen].mul(w);

          data[i + j] = u.add(v);
          data[i + j + halfLen] = u.sub(v);
        }
      }
    }
  }

  /**
   * Perform in-place inverse FFT on complex array
   */
  inverse(data: Complex[]): void {
    // Conjugate
    for (let i = 0; i < this.size; i++) {
      data[i].imag = -data[i].imag;
    }

    // Forward FFT
    this.forward(data);

    // Conjugate and scale
    for (let i = 0; i < this.size; i++) {
      data[i].real /= this.size;
      data[i].imag = -data[i].imag / this.size;
    }
  }
}

/**
 * 2D FFT implementation using row-column decomposition
 */
class FFT2D {
  private readonly width: number;
  private readonly height: number;
  private readonly rowFFT: FFT;
  private readonly colFFT: FFT;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.rowFFT = new FFT(width);
    this.colFFT = new FFT(height);
  }

  /**
   * Convert 2D array to complex 2D array
   */
  private toComplex2D(data: Float32Array): Complex[][] {
    const result: Complex[][] = [];
    for (let y = 0; y < this.height; y++) {
      result[y] = [];
      for (let x = 0; x < this.width; x++) {
        result[y][x] = new Complex(data[y * this.width + x], 0);
      }
    }
    return result;
  }

  /**
   * Convert complex 2D array back to real values
   */
  private fromComplex2D(data: Complex[][]): Float32Array {
    const result = new Float32Array(this.width * this.height);
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        result[y * this.width + x] = data[y][x].real;
      }
    }
    return result;
  }

  /**
   * Perform 2D FFT using row-column algorithm
   */
  forward(data: Float32Array): Complex[][] {
    const complex2D = this.toComplex2D(data);

    // FFT on rows
    for (let y = 0; y < this.height; y++) {
      this.rowFFT.forward(complex2D[y]);
    }

    // FFT on columns
    for (let x = 0; x < this.width; x++) {
      const col = new Array(this.height);
      for (let y = 0; y < this.height; y++) {
        col[y] = complex2D[y][x];
      }
      this.colFFT.forward(col);
      for (let y = 0; y < this.height; y++) {
        complex2D[y][x] = col[y];
      }
    }

    return complex2D;
  }

  /**
   * Perform 2D inverse FFT
   */
  inverse(complex2D: Complex[][]): Float32Array {
    // Inverse FFT on columns
    for (let x = 0; x < this.width; x++) {
      const col = new Array(this.height);
      for (let y = 0; y < this.height; y++) {
        col[y] = complex2D[y][x];
      }
      this.colFFT.inverse(col);
      for (let y = 0; y < this.height; y++) {
        complex2D[y][x] = col[y];
      }
    }

    // Inverse FFT on rows
    for (let y = 0; y < this.height; y++) {
      this.rowFFT.inverse(complex2D[y]);
    }

    return this.fromComplex2D(complex2D);
  }
}

/**
 * Main class for generating blue noise textures
 */
export class BlueNoiseGenerator {
  // Constants
  private static readonly DEFAULT_SIGMA = 1.9;
  private static readonly DEFAULT_INITIAL_DENSITY = 0.1;
  private static readonly MAX_ITERATIONS_MULTIPLIER = 10;
  private static readonly THRESHOLD_MAP_LEVELS = 256;

  // Helper methods
  private static isPowerOfTwo(n: number): boolean {
    // biome-ignore lint/suspicious/noBitwiseOperators: Bitwise operation is intentional for power-of-two check
    return n > 0 && (n & (n - 1)) === 0;
  }

  // Configuration
  private readonly width: number;
  private readonly height: number;
  private readonly area: number;
  private readonly sigma: number;
  private readonly initialDensity: number;
  private readonly verbose: boolean;
  private readonly random: SeededRandom;

  // Working arrays
  private readonly bitmap: Uint8Array;
  private readonly rank: Int32Array;
  private energy: Float32Array;

  // Cached values for performance
  private onesCount = 0;

  // FFT optimization
  private readonly useFFT: boolean;
  private readonly fft2D: FFT2D | null = null;
  private readonly gaussianKernel: Complex[][] | null = null;

  constructor(config: BlueNoiseConfig) {
    // Input validation
    if (config.width <= 0 || config.height <= 0) {
      throw new Error("Width and height must be positive");
    }
    if (!(Number.isInteger(config.width) && Number.isInteger(config.height))) {
      throw new Error("Width and height must be integers");
    }
    if (config.sigma !== undefined && config.sigma <= 0) {
      throw new Error("Sigma must be positive");
    }
    if (
      config.initialDensity !== undefined &&
      (config.initialDensity <= 0 || config.initialDensity >= 1)
    ) {
      throw new Error("Initial density must be between 0 and 1");
    }

    this.width = config.width;
    this.height = config.height;
    this.area = this.width * this.height;
    this.sigma = config.sigma ?? BlueNoiseGenerator.DEFAULT_SIGMA;
    this.initialDensity =
      config.initialDensity ?? BlueNoiseGenerator.DEFAULT_INITIAL_DENSITY;
    this.verbose = config.verbose ?? false;

    // Initialize seeded random number generator
    this.random = new SeededRandom(config.seed);

    // Use FFT if dimensions are powers of two
    this.useFFT =
      BlueNoiseGenerator.isPowerOfTwo(this.width) &&
      BlueNoiseGenerator.isPowerOfTwo(this.height);

    if (this.useFFT) {
      this.fft2D = new FFT2D(this.width, this.height);
      this.gaussianKernel = this.createGaussianKernelFFT();
    }

    // Initialize working arrays
    this.bitmap = new Uint8Array(this.area);
    this.rank = new Int32Array(this.area);
    this.energy = new Float32Array(this.area);
  }

  /**
   * Create Gaussian kernel in frequency domain for FFT convolution
   *
   * Pre-computes the Gaussian kernel and transforms it to frequency space.
   * The kernel uses toroidal distance to ensure seamless tiling. By keeping
   * the kernel in frequency space, we only need to compute FFT once during
   * initialization rather than for every blur operation.
   */
  private createGaussianKernelFFT(): Complex[][] {
    const kernel = new Float32Array(this.area);
    const divisor = 2 * this.sigma * this.sigma;

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        // Compute distance from center with wrapping
        const dx = Math.min(x, this.width - x);
        const dy = Math.min(y, this.height - y);
        const distSq = dx * dx + dy * dy;

        kernel[y * this.width + x] = Math.exp(-distSq / divisor);
      }
    }

    // Normalize kernel
    let sum = 0;
    for (let i = 0; i < this.area; i++) {
      sum += kernel[i];
    }
    for (let i = 0; i < this.area; i++) {
      kernel[i] /= sum;
    }

    // Transform to frequency domain
    return this.fft2D?.forward(kernel);
  }

  /**
   * Apply Gaussian blur using FFT (frequency domain convolution)
   *
   * FFT optimization: Convolution in the spatial domain (O(n² × k²) where k is
   * kernel size) becomes element-wise multiplication in the frequency domain
   * (O(n² log n) for FFT). This provides ~50% performance improvement for
   * power-of-two dimensions.
   *
   * The convolution theorem states: convolution(A, B) = IFFT(FFT(A) × FFT(B))
   */
  private gaussianBlurFFT(data: Uint8Array): Float32Array {
    // Convert to float and transform to frequency domain
    const floatData = new Float32Array(this.area);
    for (let i = 0; i < this.area; i++) {
      floatData[i] = data[i];
    }

    const dataFreq = this.fft2D?.forward(floatData);

    // Element-wise multiplication in frequency domain (convolution)
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        dataFreq[y][x] = dataFreq[y][x].mul(this.gaussianKernel?.[y][x]);
      }
    }

    // Transform back to spatial domain
    return this.fft2D?.inverse(dataFreq);
  }

  /**
   * Apply Gaussian blur using spatial domain convolution (fallback)
   *
   * Used when dimensions are not powers of two. Applies the Gaussian kernel
   * directly in the spatial domain by computing weighted sums of neighbouring
   * pixels. Coordinates wrap around (torus topology) to ensure seamless tiling.
   */
  private gaussianBlurSpatial(data: Uint8Array): Float32Array {
    const blurred = new Float32Array(this.area);
    const kernelRadius = Math.ceil(3 * this.sigma);
    const divisor = 2 * this.sigma * this.sigma;

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        let sum = 0;
        let weightSum = 0;

        for (let ky = -kernelRadius; ky <= kernelRadius; ky++) {
          for (let kx = -kernelRadius; kx <= kernelRadius; kx++) {
            // Wrap coordinates (torus topology)
            const px = (x + kx + this.width) % this.width;
            const py = (y + ky + this.height) % this.height;

            const distSq = kx * kx + ky * ky;
            const weight = Math.exp(-distSq / divisor);

            sum += data[py * this.width + px] * weight;
            weightSum += weight;
          }
        }

        blurred[y * this.width + x] = sum / weightSum;
      }
    }

    return blurred;
  }

  /**
   * Apply Gaussian blur (chooses FFT or spatial based on size)
   */
  private gaussianBlur(data: Uint8Array): Float32Array {
    if (this.useFFT) {
      return this.gaussianBlurFFT(data);
    }
    return this.gaussianBlurSpatial(data);
  }

  /**
   * Find the tightest cluster: pixel with highest energy among all 1s
   *
   * A "cluster" is a region where pixels are densely packed. The Gaussian
   * blur creates an energy field where areas with many nearby 1s have high
   * energy. The tightest cluster is the 1-pixel with the most neighbours.
   *
   * @returns Index of the pixel in the tightest cluster
   */
  private findTightestCluster(): number {
    let maxEnergy = Number.NEGATIVE_INFINITY;
    let maxIdx = -1;

    for (let i = 0; i < this.area; i++) {
      if (this.bitmap[i] === 1 && this.energy[i] > maxEnergy) {
        maxEnergy = this.energy[i];
        maxIdx = i;
      }
    }

    return maxIdx;
  }

  /**
   * Find the largest void: pixel with lowest energy among all 0s
   *
   * A "void" is a region where pixels are sparse. In the energy field,
   * areas with few nearby 1s have low energy. The largest void is the
   * 0-pixel with the fewest neighbours.
   *
   * @returns Index of the pixel in the largest void
   */
  private findLargestVoid(): number {
    let minEnergy = Number.POSITIVE_INFINITY;
    let minIdx = -1;

    for (let i = 0; i < this.area; i++) {
      if (this.bitmap[i] === 0 && this.energy[i] < minEnergy) {
        minEnergy = this.energy[i];
        minIdx = i;
      }
    }

    return minIdx;
  }

  /**
   * Count number of 1s in the bitmap (returns cached value)
   */
  private countOnes(): number {
    return this.onesCount;
  }

  /**
   * Set a bit in the bitmap and update the cache
   */
  private setBit(idx: number, value: number): void {
    const oldValue = this.bitmap[idx];
    this.bitmap[idx] = value;
    this.onesCount += value - oldValue;
  }

  /**
   * Recalculate ones count from scratch (used for initialization)
   */
  private recalculateOnesCount(): void {
    let count = 0;
    for (let i = 0; i < this.area; i++) {
      count += this.bitmap[i];
    }
    this.onesCount = count;
  }

  /**
   * Recalculate energy field by applying Gaussian blur to bitmap
   *
   * The energy field is the key to finding clusters and voids. By applying
   * a Gaussian blur to the binary pattern, we create a smooth field where:
   * - High values indicate clusters (many nearby 1s)
   * - Low values indicate voids (few nearby 1s)
   *
   * The blur uses torus wrapping so edges connect seamlessly.
   */
  private recalculateEnergy(): void {
    this.energy = this.gaussianBlur(this.bitmap);
  }

  /**
   * Phase 0: Generate Initial Binary Pattern
   *
   * Creates a well-distributed set of initial points by:
   * 1. Randomly placing points (10% density by default)
   * 2. Iteratively swapping clustered points with void positions
   * 3. Stopping when convergence is reached (cluster = void position)
   *
   * This phase establishes the foundation for even distribution.
   */
  private phase0_generateInitialPattern(): void {
    const targetPoints = Math.floor(this.area * this.initialDensity);

    // Randomly place initial points
    while (this.countOnes() < targetPoints) {
      const idx = Math.floor(this.random.next() * this.area);
      if (this.bitmap[idx] === 0) {
        this.setBit(idx, 1);
      }
    }

    this.recalculateEnergy();

    // Redistribute points until convergence
    let iterations = 0;
    const maxIterations =
      this.area * BlueNoiseGenerator.MAX_ITERATIONS_MULTIPLIER;

    while (iterations < maxIterations) {
      iterations++;

      // Find tightest cluster and remove it
      const clusterIdx = this.findTightestCluster();
      this.setBit(clusterIdx, 0);

      // Find largest void with updated energy
      this.recalculateEnergy();
      const voidIdx = this.findLargestVoid();

      // Check for convergence
      if (voidIdx === clusterIdx) {
        this.setBit(clusterIdx, 1);
        this.recalculateEnergy();
        break;
      }

      // Place point in void
      this.setBit(voidIdx, 1);
      this.recalculateEnergy();
    }
  }

  /**
   * Phase 1: Serialize Initial Points
   *
   * Assigns ranks to the initial minority pattern by removing points from
   * tightest clusters first. These points are ranked from (initialPoints - 1)
   * down to 0, establishing which pixels are most important for creating
   * the blue noise distribution.
   */
  private phase1_serializeInitialPoints(): void {
    let rankCounter = this.countOnes() - 1;

    while (this.countOnes() > 0) {
      const clusterIdx = this.findTightestCluster();
      this.rank[clusterIdx] = rankCounter;
      rankCounter--;

      this.setBit(clusterIdx, 0);
      this.recalculateEnergy();
    }
  }

  /**
   * Phase 2: Fill to Half Capacity
   *
   * Restores the initial pattern and continues adding points to the largest
   * voids until the bitmap is 50% full. Ranks continue from initialPoints
   * to area/2. This builds up a minority pattern (less than half full).
   */
  private phase2_fillToHalf(
    prototype: Uint8Array,
    initialPoints: number
  ): void {
    this.bitmap.set(prototype);
    this.recalculateOnesCount();
    this.recalculateEnergy();

    let rankCounter = initialPoints;
    const halfArea = Math.floor(this.area / 2);

    while (this.countOnes() < halfArea) {
      const voidIdx = this.findLargestVoid();
      this.rank[voidIdx] = rankCounter;
      rankCounter++;

      this.setBit(voidIdx, 1);
      this.recalculateEnergy();
    }
  }

  /**
   * Phase 3: Fill to Completion
   *
   * Inverts the bitmap so 0s become the minority (more than half full).
   * Then removes the remaining minority pixels from their tightest clusters,
   * ranking them from area/2 to area-1. This clever inversion allows the
   * algorithm to work symmetrically for both minority and majority patterns.
   */
  private phase3_fillToCompletion(startRank: number): void {
    // Invert bitmap
    for (let i = 0; i < this.area; i++) {
      this.bitmap[i] = 1 - this.bitmap[i];
    }
    this.recalculateOnesCount();
    this.recalculateEnergy();

    let rankCounter = startRank;
    while (rankCounter < this.area) {
      const clusterIdx = this.findTightestCluster();
      this.rank[clusterIdx] = rankCounter;
      rankCounter++;

      this.setBit(clusterIdx, 0);
      this.recalculateEnergy();
    }
  }

  /**
   * Phase 4: Convert Ranks to Threshold Map
   *
   * Maps the rank values (0 to area-1) to grayscale threshold values (0-255).
   * Each pixel's rank determines its threshold value for dithering. Lower
   * ranked pixels will be turned "on" first when dithering bright images.
   */
  private phase4_convertToThresholdMap(): Uint8ClampedArray {
    const output = new Uint8ClampedArray(this.area);

    for (let i = 0; i < this.area; i++) {
      output[i] = Math.floor(
        (this.rank[i] * BlueNoiseGenerator.THRESHOLD_MAP_LEVELS) / this.area
      );
    }

    return output;
  }

  /**
   * Generate the blue noise texture
   */
  generate(): BlueNoiseResult {
    const startTime = this.verbose ? Date.now() : 0;

    if (this.verbose) {
      console.log(
        `Generating ${this.width}x${this.height} blue noise texture...`
      );
      console.log(
        `Using ${this.useFFT ? "FFT-optimized" : "spatial"} Gaussian blur`
      );
      console.log("Phase 0: Generating initial pattern...");
    }

    this.phase0_generateInitialPattern();

    const prototype = this.bitmap.slice();
    const initialPoints = this.countOnes();

    if (this.verbose) {
      console.log(`Initial pattern: ${initialPoints} points`);
      console.log("Phase 1: Serializing initial points...");
    }

    this.phase1_serializeInitialPoints();

    if (this.verbose) {
      console.log("Phase 2: Filling to half capacity...");
    }

    this.phase2_fillToHalf(prototype, initialPoints);

    if (this.verbose) {
      console.log("Phase 3: Filling to completion...");
    }

    const halfArea = Math.floor(this.area / 2);
    this.phase3_fillToCompletion(halfArea);

    if (this.verbose) {
      console.log("Phase 4: Converting to threshold map...");
    }

    const data = this.phase4_convertToThresholdMap();

    if (this.verbose) {
      const elapsed = Date.now() - startTime;
      console.log(
        `✓ Blue noise generation complete in ${(elapsed / 1000).toFixed(2)}s`
      );
    }

    return { data, width: this.width, height: this.height };
  }
}

/**
 * Convenience function to generate a blue noise texture
 */
export function generateBlueNoise(
  width = 64,
  height = 64,
  sigma = 1.9
): BlueNoiseResult {
  const generator = new BlueNoiseGenerator({ width, height, sigma });
  return generator.generate();
}

/**
 * Save blue noise texture to PNG file
 */
export async function saveBlueNoiseToPNG(
  result: BlueNoiseResult,
  filename: string
): Promise<void> {
  try {
    const sharp = await import("sharp");

    await sharp
      .default(result.data, {
        raw: {
          width: result.width,
          height: result.height,
          channels: 1,
        },
      })
      .png()
      .toFile(filename);

    console.log(` Saved blue noise texture to ${filename}`);
  } catch (error) {
    console.error("Error saving PNG:", error);
    throw error;
  }
}

/**
 * Apply ordered dithering using blue noise threshold map
 */
export function orderedDither(
  value: number,
  x: number,
  y: number,
  blueNoise: BlueNoiseResult,
  levels = 2
): number {
  const noiseX = x % blueNoise.width;
  const noiseY = y % blueNoise.height;
  const noiseIdx = noiseY * blueNoise.width + noiseX;
  const threshold = blueNoise.data[noiseIdx];

  const normalized = value / 255;
  const step = 1 / levels;
  const quantized = Math.floor(normalized / step);
  const fraction = (normalized % step) / step;

  const output =
    fraction > threshold / 255
      ? Math.min(quantized + 1, levels - 1)
      : quantized;

  return Math.floor((output * 255) / (levels - 1));
}
