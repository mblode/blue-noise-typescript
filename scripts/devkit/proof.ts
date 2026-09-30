import type { Check, Feature } from "./features.ts";

export interface Run {
  command: string;
  exitCode: number;
  ms: number;
}

/**
 * A feature's user paths are its coverage set. A path is covered when at
 * least one of its checks ran and every check that ran passed, failed when a
 * check that ran failed, and skipped (with the reason) otherwise.
 */
export const coverage = (feature: Feature, ran: Map<Check, Run>) => {
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
  }
  return {
    checks: [...ran.entries()].map(([item, result]) => ({
      method: item.method,
      path: item.path,
      ...result,
    })),
    covered,
    failed,
    skipped,
  };
};
