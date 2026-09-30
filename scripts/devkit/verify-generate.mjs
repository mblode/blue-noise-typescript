// npm run verify:generate
// Builds the CLI and runs `generate` into a temp file, then checks a PNG
// came out. This is the "generate" feature's cli check.
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const PNG_SIGNATURE = "89504e470d0a1a0a";

execFileSync("npm", ["run", "build"], { cwd: root, stdio: "inherit" });

const dir = mkdtempSync(path.join(tmpdir(), "blue-noise-verify-"));
try {
  const output = path.join(dir, "noise.png");
  execFileSync(
    "node",
    ["dist/cli.js", "generate", "-s", "16", "--seed", "1", "-o", output],
    { cwd: root, stdio: "inherit" }
  );

  if (!existsSync(output)) {
    console.error(`expected ${output} to exist`);
    process.exit(1);
  }
  const bytes = readFileSync(output);
  if (bytes.subarray(0, 8).toString("hex") !== PNG_SIGNATURE) {
    console.error(`${output} is not a valid PNG`);
    process.exit(1);
  }
  console.log(`generate: wrote a ${bytes.length}-byte PNG to ${output}`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
