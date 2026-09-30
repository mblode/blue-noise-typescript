// npm run doctor [-- --json]
// Read-only: is this checkout ready to build and verify? Checks Node and the
// sharp dependency, and prints the exact command that fixes each failure.
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

interface Result {
  detail: string;
  fix?: string;
  name: string;
  ok: boolean;
}

const root = path.resolve(import.meta.dirname, "../..");
const results: Result[] = [];
const pass = (name: string, detail: string) =>
  results.push({ detail, name, ok: true });
const fail = (name: string, detail: string, fix: string) =>
  results.push({ detail, fix, name, ok: false });
const read = (file: string) => readFileSync(path.join(root, file), "utf-8");

// Node version: engines in package.json is the source.
const manifest: { engines: { node: string } } = JSON.parse(
  read("package.json")
);
const nodeWant = Number(manifest.engines.node.replaceAll(/\D/gu, ""));
const nodeMajor = Number(process.versions.node.split(".")[0]);
if (nodeMajor >= nodeWant) {
  pass("node", `v${process.versions.node}`);
} else {
  fail(
    "node",
    `v${process.versions.node}, need ${manifest.engines.node}`,
    `nvm install ${nodeWant} && nvm use ${nodeWant}`
  );
}

// Dependencies: node_modules and the sharp native binding this CLI needs.
if (existsSync(path.join(root, "node_modules"))) {
  pass("dependencies", "node_modules present");
} else {
  fail("dependencies", "node_modules missing", "npm install");
}
if (existsSync(path.join(root, "node_modules/sharp"))) {
  pass("sharp", "installed");
} else {
  fail("sharp", "missing", "npm install");
}

if (process.argv.includes("--json")) {
  console.log(
    JSON.stringify({ ok: results.every((r) => r.ok), results }, null, 2)
  );
} else {
  console.log(`doctor: ${root}`);
  for (const r of results) {
    console.log(
      `  ${r.ok ? "ok  " : "FAIL"}  ${r.name.padEnd(11)} ${r.detail}`
    );
    if (r.fix) {
      console.log(`        fix: ${r.fix}`);
    }
  }
}
if (!results.every((r) => r.ok)) {
  process.exitCode = 1;
}
