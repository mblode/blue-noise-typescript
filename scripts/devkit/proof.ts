import type { Check, Feature, Method } from "./features.ts";

export interface Run {
  command: string;
  exitCode: number;
  ms: number;
}

const reason = (item: Check, fast: boolean) => {
  if (item.method === "playwright" && fast) {
    return "playwright: --fast runs cli checks only";
  }
  return `${item.method}: verify does not drive ${item.method} checks; run ${item.spec ?? "it"} by hand and record the result`;
};

/**
 * A feature's user paths are its coverage set. A path is covered when at
 * least one of its checks ran and every check that ran passed, failed when a
 * check that ran failed, and skipped (with the reason) otherwise. A check
 * that did not run is listed as skipped even when its path is covered by
 * another method, so the proof shows what was not driven.
 */
export const coverage = (
  feature: Feature,
  ran: Map<Check, Run>,
  fast: boolean
) => {
  const covered: string[] = [];
  const failed: { command: string; exitCode: number; path: string }[] = [];
  const skipped: { path: string; reason: string }[] = [];
  for (const userPath of feature.userPaths) {
    const checks = feature.checks.filter((c) => c.path === userPath);
    if (checks.length === 0) {
      skipped.push({ path: userPath, reason: "no check in feature.json" });
      continue;
    }
    const runs = checks.flatMap((c) => ran.get(c) ?? []);
    const failures = runs.filter((r) => r.exitCode !== 0);
    failed.push(
      ...failures.map((r) => ({
        command: r.command,
        exitCode: r.exitCode,
        path: userPath,
      }))
    );
    if (runs.length > 0 && failures.length === 0) {
      covered.push(userPath);
    }
    skipped.push(
      ...checks
        .filter((c) => !ran.has(c))
        .map((c) => ({ path: userPath, reason: reason(c, fast) }))
    );
  }
  const methods: Method[] = [...new Set([...ran.keys()].map((c) => c.method))];
  return {
    checks: [...ran.entries()].map(([item, result]) => ({
      method: item.method,
      path: item.path,
      ...result,
    })),
    covered,
    failed,
    methods,
    skipped,
  };
};
