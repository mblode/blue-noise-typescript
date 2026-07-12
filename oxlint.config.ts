import { defineConfig } from "oxlint";
import core from "ultracite/oxlint/core";

export default defineConfig({
  extends: [core],
  ignorePatterns: [...core.ignorePatterns, "dist/**"],
  // Rules relaxed for this codebase. `src/generator.ts` is a hot-path numeric
  // blue-noise / dithering algorithm where these are load-bearing, not style:
  // the mulberry32 PRNG relies on `| 0` / `>>> 0` uint32 coercions (Math.trunc
  // would change semantics), bitwise ops and `++` counters run in tight pixel
  // loops, `new Array(n)` preallocates, and `Math.sqrt(dx*dx+dy*dy)` is faster
  // than `Math.hypot` per iteration. `func-style` follows touchwood's
  // precedent; the CLI's `parseInt` differs from `Number` so coercion rewrites
  // would alter parsing. None have a safe autofix, so they're deferred rather
  // than hand-edited into potential correctness bugs.
  rules: {
    "no-plusplus": "off",
    "no-bitwise": "off",
    "func-style": "off",
    "sort-keys": "off",
    "max-classes-per-file": "off",
    "class-methods-use-this": "off",
    "prefer-named-capture-group": "off",
    "require-unicode-regexp": "off",
    "prefer-destructuring": "off",
    "unicorn/prefer-math-trunc": "off",
    "unicorn/prefer-modern-math-apis": "off",
    "unicorn/prefer-number-coercion": "off",
    "unicorn/no-new-array": "off",
    // `Uint8Array.prototype.slice()` returns a typed array; the spread rewrite
    // would silently produce a `number[]` and break the type.
    "unicorn/prefer-spread": "off",
  },
});
