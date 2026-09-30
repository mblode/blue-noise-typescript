import { execFileSync } from "node:child_process";
import { existsSync, globSync, readFileSync } from "node:fs";
import path from "node:path";

// The feature map: one feature.json per feature, describing it from the
// user's side (how they reach it) and how to prove it works (checks).
// `npm run verify` maps a diff to features through `paths` and runs their
// checks; `npm run check-features` keeps the map honest.

// This CLI has no browser surface, so the only supported check method is
// "cli": a command that builds and runs dist/cli.js. Add "playwright" (or
// another browser method) here, and its argv construction in verify.ts,
// only once there is an e2e/ suite to drive.
export const methods = ["cli"] as const;
export type Method = (typeof methods)[number];

export interface Check {
  command: string;
  method: Method;
  path: string;
}

export interface Feature {
  checks: Check[];
  gotchas: string[];
  id: string;
  paths: string[];
  preconditions: string[];
  title: string;
  userPaths: string[];
}

export interface Loaded {
  /** Repo-relative path of the feature.json. */
  file: string;
  feature: Feature;
}

const ID_REGEX = /^[a-z][a-z0-9-]*$/u;

/** Problems with a feature.json's shape, each naming the field at fault. */
const shapeProblems = (file: string, json: unknown): string[] => {
  if (typeof json !== "object" || json === null) {
    return [`${file}: must be a JSON object`];
  }
  const f = json as Record<string, unknown>;
  const problems: string[] = [];
  if (typeof f.id !== "string" || !ID_REGEX.test(f.id)) {
    problems.push(`${file}: "id" must be lowercase-kebab-case`);
  }
  if (typeof f.title !== "string" || f.title.length === 0) {
    problems.push(`${file}: "title" must be a non-empty string`);
  }
  if (!(Array.isArray(f.paths) && f.paths.length > 0)) {
    problems.push(`${file}: "paths" must be a non-empty array of strings`);
  }
  if (!(Array.isArray(f.userPaths) && f.userPaths.length > 0)) {
    problems.push(`${file}: "userPaths" must be a non-empty array of strings`);
  }
  if (Array.isArray(f.checks)) {
    for (const [i, item] of f.checks.entries()) {
      const c = item as Record<string, unknown>;
      if (typeof c?.path !== "string" || c.path.length === 0) {
        problems.push(`${file}: checks[${i}].path must be a non-empty string`);
      }
      if (!methods.includes(c?.method as Method)) {
        problems.push(
          `${file}: checks[${i}].method must be one of ${methods.join(", ")}`
        );
      }
    }
  } else {
    problems.push(`${file}: "checks" must be an array`);
  }
  if (!Array.isArray(f.preconditions)) {
    problems.push(`${file}: "preconditions" must be an array`);
  }
  if (!Array.isArray(f.gotchas)) {
    problems.push(`${file}: "gotchas" must be an array`);
  }
  return problems;
};

/** Tracked and untracked files, minus ignored and deleted ones. */
export const repoFiles = (root: string) =>
  execFileSync(
    "git",
    ["ls-files", "--cached", "--others", "--exclude-standard"],
    {
      cwd: root,
      encoding: "utf-8",
    }
  )
    .split("\n")
    .filter((file) => file && existsSync(path.join(root, file)));

export const loadFeatures = (root: string, files: string[]) => {
  const loaded: Loaded[] = [];
  const problems: string[] = [];
  for (const file of files.filter((f) => path.basename(f) === "feature.json")) {
    const text = readFileSync(path.join(root, file), "utf-8");
    let json: unknown;
    try {
      json = JSON.parse(text);
    } catch (error) {
      problems.push(`${file}: not valid JSON (${String(error)}).`);
      continue;
    }
    const shape = shapeProblems(file, json);
    if (shape.length > 0) {
      problems.push(...shape);
    } else {
      loaded.push({ feature: json as Feature, file });
    }
  }
  return { loaded, problems };
};

/** A path ending in / owns everything under it; anything else is one file. */
export const owns = (pattern: string, file: string) =>
  pattern.endsWith("/") ? file.startsWith(pattern) : file === pattern;

/** Files a feature must own: this CLI's source in src/. */
export const isSource = (file: string) =>
  file.startsWith("src/") && !file.endsWith(".md");

export const mapFiles = (loaded: Loaded[], files: string[]) => {
  const touched = new Map<string, string[]>();
  const uncovered: string[] = [];
  for (const file of files) {
    // A feature.json always belongs to its own feature.
    const owners = loaded.filter(
      (l) => l.file === file || l.feature.paths.some((p) => owns(p, file))
    );
    if (owners.length === 0) {
      uncovered.push(file);
    }
    for (const { feature: f } of owners) {
      touched.set(f.id, [...(touched.get(f.id) ?? []), file]);
    }
  }
  return { touched, uncovered };
};

interface Workspace {
  dir: string;
  scripts: Record<string, string>;
}

const readJson = (
  file: string
): { name?: string; scripts?: Record<string, string>; workspaces?: string[] } =>
  JSON.parse(readFileSync(file, "utf-8"));

export const workspaces = (root: string) => {
  const manifest = readJson(path.join(root, "package.json"));
  const map = new Map<string, Workspace>([
    ["", { dir: root, scripts: manifest.scripts ?? {} }],
  ]);
  for (const file of globSync(
    (manifest.workspaces ?? []).map((w) => `${w}/package.json`),
    { cwd: root }
  )) {
    const pkg = readJson(path.join(root, file));
    if (pkg.name) {
      map.set(pkg.name, {
        dir: path.join(root, path.dirname(file)),
        scripts: pkg.scripts ?? {},
      });
    }
  }
  return map;
};

const shape = "npm run <script> [-w <workspace>] [-- <args>]";

/** Problems with a cli check's command: its script, workspace and file arguments must exist. */
export const commandProblems = (
  command: string,
  spaces: Map<string, Workspace>
) => {
  const match = command.match(
    /^npm run (?<script>[\w:-]+)(?: -w (?<ws>[@\w/-]+))?(?: -- (?<rest>.+))?$/u
  );
  if (!match?.groups?.script) {
    return [`command "${command}" must be ${shape}, so verify can run it`];
  }
  const { script, ws = "", rest = "" } = match.groups;
  const space = spaces.get(ws);
  if (!space) {
    return [`command "${command}": no workspace named ${ws}`];
  }
  if (!space.scripts[script]) {
    return [
      `command "${command}": ${ws || "the root"} has no "${script}" script`,
    ];
  }
  return rest
    .split(/\s+/u)
    .filter((arg) => !arg.startsWith("-") && /[/.]/u.test(arg))
    .filter((arg) => !existsSync(path.join(space.dir, arg)))
    .map(
      (arg) =>
        `command "${command}": ${path.relative(path.dirname(space.dir), path.join(space.dir, arg))} does not exist`
    );
};

const checkProblems = (item: Check, spaces: Map<string, Workspace>) =>
  item.command
    ? commandProblems(item.command, spaces)
    : [`cli check "${item.path}" needs a command (${shape})`];

/** Every problem with the map, each naming its fix. */
export const validate = (root: string, loaded: Loaded[], files: string[]) => {
  const spaces = workspaces(root);
  const problems: string[] = [];
  const ids = new Map<string, string>();
  for (const { file, feature: f } of loaded) {
    const seen = ids.get(f.id);
    if (seen) {
      problems.push(
        `${file}: id "${f.id}" is already used by ${seen}. Pick a unique id.`
      );
    }
    ids.set(f.id, file);
    for (const pattern of f.paths.filter(
      (p) => !existsSync(path.join(root, p))
    )) {
      problems.push(
        `${file}: paths entry ${pattern} does not exist. Point it at files this feature owns (a directory ends in /).`
      );
    }
    for (const item of f.checks) {
      if (!f.userPaths.includes(item.path)) {
        problems.push(
          `${file}: check path "${item.path}" is not in userPaths. Name the user path it proves, or add it to userPaths.`
        );
      }
      problems.push(
        ...checkProblems(item, spaces).map((p) => `${file}: ${p}.`)
      );
    }
  }
  const { uncovered } = mapFiles(loaded, files.filter(isSource));
  for (const file of uncovered) {
    problems.push(
      `${file} belongs to no feature. Add it (or its directory, ending in /) to the paths of the feature.json that owns it.`
    );
  }
  return problems;
};
