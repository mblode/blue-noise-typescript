import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";

import { loadFeatures, mapFiles, repoFiles } from "./features.ts";

const root = path.resolve(import.meta.dirname, "../..");

test("a package.json-only diff runs the real CLI checks, not just the harness", () => {
  const { loaded, problems } = loadFeatures(root, repoFiles(root));
  assert.deepEqual(problems, []);
  const { touched, uncovered } = mapFiles(loaded, ["package.json"]);
  assert.deepEqual(uncovered, []);
  assert.deepEqual([...touched.keys()].toSorted(), [
    "dither",
    "generate",
    "verification-harness",
  ]);
});

test("a package-lock.json-only diff runs the real CLI checks, not just the harness", () => {
  const { loaded } = loadFeatures(root, repoFiles(root));
  const { touched, uncovered } = mapFiles(loaded, ["package-lock.json"]);
  assert.deepEqual(uncovered, []);
  assert.deepEqual([...touched.keys()].toSorted(), [
    "dither",
    "generate",
    "verification-harness",
  ]);
});

test("a blue-noise.png-only diff selects dither, not none", () => {
  const { loaded } = loadFeatures(root, repoFiles(root));
  const { touched, uncovered } = mapFiles(loaded, ["blue-noise.png"]);
  assert.deepEqual(uncovered, []);
  assert.deepEqual([...touched.keys()], ["dither"]);
});
