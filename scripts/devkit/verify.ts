// npm run verify [-- --since <ref>] [--fast] [--ci]
// Maps the files changed since <ref> (default origin/main, plus uncommitted
// and untracked files) to features through their feature.json paths, runs
// each touched feature's checks and writes .factory/proof.json: per feature,
// the user paths covered, skipped (with the reason) and failed. --fast has
// no other method to skip yet, so it behaves the same as a full run. Exits 1
// when a check fails or a changed source file belongs to no feature.
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";

import { isSource, loadFeatures, mapFiles, repoFiles } from "./features.ts";
import type { Check, Loaded, Method } from "./features.ts";
import { coverage } from "./proof.ts";
import type { Run } from "./proof.ts";

const root = path.resolve(import.meta.dirname, "../..");
const { values } = parseArgs({
  options: {
    ci: { default: false, type: "boolean" },
    fast: { default: false, type: "boolean" },
    since: { default: "origin/main", type: "string" },
  },
});

const git = (...args: string[]) =>
  execFileSync("git", args, { cwd: root, encoding: "utf-8" }).trim();
const lines = (text: string) => text.split("\n").filter(Boolean);

let base: string;
try {
  base = git("merge-base", values.since, "HEAD");
} catch {
  console.error(
    `git cannot find a merge base with ${values.since}. Run \`git fetch origin\` or pass --since <ref>.`
  );
  process.exit(1);
}
const changed = [
  ...new Set([
    ...lines(git("diff", "--name-only", base)),
    ...lines(git("ls-files", "--others", "--exclude-standard")),
  ]),
].toSorted();

const { loaded, problems } = loadFeatures(root, repoFiles(root));
if (problems.length > 0) {
  console.error(
    `Fix the feature map first (npm run check-features):\n- ${problems.join("\n- ")}`
  );
  process.exit(1);
}
const { touched, uncovered } = mapFiles(loaded, changed);
const features = loaded.filter(({ feature }) => touched.has(feature.id));

const run: Method[] = new Set(["cli", "playwright"]);
const argv = (item: Check) =>
  item.method === "cli" ? (item.command ?? "").split(" ").slice(1) : [];

const results = new Map<string, Run>();
const runCheck = (item: Check): Run => {
  const args = argv(item);
  const key = args.join(" ");
  const cached = results.get(key);
  if (cached) {
    return cached;
  }
  console.log(`\n$ npm ${key}`);
  const started = Date.now();
  const child = spawnSync("npm", args, {
    cwd: root,
    env: { ...process.env, ...(values.ci && { CI: "1" }) },
    stdio: "inherit",
  });
  const result = {
    command: `npm ${key}`,
    exitCode: child.status ?? 1,
    ms: Date.now() - started,
  };
  results.set(key, result);
  return result;
};

const proofFeatures = features.map(({ file, feature }: Loaded) => {
  const ran = new Map<Check, Run>();
  for (const item of feature.checks.filter((c) => run.has(c.method))) {
    ran.set(item, runCheck(item));
  }
  return {
    file,
    id: feature.id,
    ...coverage(feature, ran, values.fast),
    files: touched.get(feature.id) ?? [],
  };
});

const uncoveredSource = uncovered.filter(isSource);
const failed = proofFeatures.filter((f) => f.failed.length > 0);
const proof = {
  base,
  changed: changed.length,
  dirty: git("status", "--porcelain") !== "",
  features: proofFeatures,
  mode: values.fast ? "fast" : "full",
  ok: failed.length === 0 && uncoveredSource.length === 0,
  sha: git("rev-parse", "HEAD"),
  since: values.since,
  uncovered,
};
const proofFile = path.join(root, ".factory", "proof.json");
mkdirSync(path.dirname(proofFile), { recursive: true });
writeFileSync(proofFile, `${JSON.stringify(proof, null, 2)}\n`);

console.log(
  `\nverify (${proof.mode}) since ${values.since}: ${changed.length} changed files, ${features.length} features`
);
for (const f of proofFeatures) {
  console.log(
    `  ${f.failed.length > 0 ? "FAIL" : "ok  "} ${f.id}: covered ${f.covered.length}, skipped ${f.skipped.length}, failed ${f.failed.length} (${f.methods.join(", ") || "no checks run"})`
  );
}
if (uncoveredSource.length > 0) {
  console.log(
    `  FAIL source files in no feature (add them to a feature.json's paths):\n    ${uncoveredSource.join("\n    ")}`
  );
}
console.log(`proof: ${path.relative(root, proofFile)}`);
if (values.ci) {
  console.log(JSON.stringify(proof));
}

// factory records the proof when its CLI is installed; verify never needs it.
const factory = spawnSync("factory", ["proof", proofFile], {
  stdio: "inherit",
});
if (factory.error) {
  console.log("factory CLI not on PATH; proof not recorded.");
}
if (!proof.ok) {
  process.exitCode = 1;
}
