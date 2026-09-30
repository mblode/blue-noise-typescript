import assert from "node:assert/strict";
import { test } from "node:test";

import { verifyArgs } from "./verify-args.ts";

const root = "/repo";

test("a pull_request always diffs against the base branch", () => {
  assert.deepEqual(
    verifyArgs({ before: "abc123", eventName: "pull_request" }, root),
    ["--since", "origin/main"]
  );
});

test("a push with a real previous commit diffs against it, so a merge to main is actually checked", () => {
  assert.deepEqual(
    verifyArgs({ before: "abc123", eventName: "push" }, root, () => true),
    ["--since", "abc123"]
  );
});

test("a branch's first push (the null sha) falls closed to --all instead of checking nothing", () => {
  assert.deepEqual(
    verifyArgs(
      { before: "0000000000000000000000000000000000000000", eventName: "push" },
      root,
      () => true
    ),
    ["--all"]
  );
});

test("a push with no before at all falls closed to --all", () => {
  assert.deepEqual(
    verifyArgs({ eventName: "push" }, root, () => true),
    ["--all"]
  );
});

test("a force-push whose before commit no longer exists falls closed to --all", () => {
  assert.deepEqual(
    verifyArgs({ before: "abc123", eventName: "push" }, root, () => false),
    ["--all"]
  );
});
