import { execFileSync } from "node:child_process";

const NULL_SHA = /^0+$/u;

export type CommitExists = (sha: string, root: string) => boolean;

export const realCommitExists: CommitExists = (sha, root) => {
  try {
    execFileSync("git", ["cat-file", "-e", `${sha}^{commit}`], {
      cwd: root,
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
};

export interface PushEvent {
  before?: string;
  eventName: string;
}

/**
 * The `npm run verify` arguments for a CI trigger. A pull_request always
 * diffs against the base branch. A push diffs against the commit it moved
 * from (github.event.before), so a merge landing on main is actually
 * checked instead of comparing HEAD to itself (which finds no diff and
 * silently runs zero checks). A branch's first push (before is the null
 * sha) and a force-push that rewrites history out from under that sha both
 * leave `before` unusable; either falls closed to --all rather than
 * checking nothing.
 */
export const verifyArgs = (
  event: PushEvent,
  root: string,
  commitExists: CommitExists = realCommitExists
): string[] => {
  if (event.eventName !== "push") {
    return ["--since", "origin/main"];
  }
  const { before } = event;
  if (!before || NULL_SHA.test(before) || !commitExists(before, root)) {
    return ["--all"];
  }
  return ["--since", before];
};
