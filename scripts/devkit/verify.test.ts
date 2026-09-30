import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";

const root = path.resolve(import.meta.dirname, "../..");

test("verify --all builds and runs every feature's checks, proving a push whose diff is empty (main pushed to itself) still checks something", () => {
  execFileSync("node", ["scripts/devkit/verify.ts", "--all", "--ci"], {
    cwd: root,
    encoding: "utf-8",
  });
  const proof = JSON.parse(
    readFileSync(path.join(root, ".factory/proof.json"), "utf-8")
  );
  assert.equal(proof.since, "all");
  assert.equal(proof.ok, true);
  assert.deepEqual(proof.features.map((f: { id: string }) => f.id).toSorted(), [
    "dither",
    "generate",
    "verification-harness",
  ]);
  for (const feature of proof.features) {
    assert.ok(
      feature.checks.length > 0,
      `${feature.id} ran no checks in --all mode`
    );
    assert.ok(
      feature.checks.every((c: { exitCode: number }) => c.exitCode === 0),
      `${feature.id} had a failing check in --all mode`
    );
  }
});
