// npm run verify:dither
// Builds the CLI, dithers a small in-memory fixture against the repo's own
// blue-noise.png, and checks a PNG came out. This is the "dither" feature's
// cli check.
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import sharp from "sharp";

const root = path.resolve(import.meta.dirname, "../..");
const PNG_SIGNATURE = "89504e470d0a1a0a";
const SIZE = 32;

execFileSync("npm", ["run", "build"], { cwd: root, stdio: "inherit" });

const dir = mkdtempSync(path.join(tmpdir(), "blue-noise-verify-"));
try {
  const fixture = path.join(dir, "input.png");
  const gradient = Buffer.alloc(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      gradient[y * SIZE + x] = Math.floor((x / SIZE) * 255);
    }
  }
  await sharp(gradient, {
    raw: { channels: 1, height: SIZE, width: SIZE },
  })
    .png()
    .toFile(fixture);

  execFileSync(
    "node",
    [
      "dist/cli.js",
      "dither",
      fixture,
      "-o",
      dir,
      "-n",
      path.join(root, "blue-noise.png"),
    ],
    { cwd: root, stdio: "inherit" }
  );

  const output = path.join(dir, "input-dithered.png");
  if (!existsSync(output)) {
    console.error(`expected ${output} to exist`);
    process.exit(1);
  }
  const bytes = readFileSync(output);
  if (bytes.subarray(0, 8).toString("hex") !== PNG_SIGNATURE) {
    console.error(`${output} is not a valid PNG`);
    process.exit(1);
  }
  console.log(`dither: wrote a ${bytes.length}-byte PNG to ${output}`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
