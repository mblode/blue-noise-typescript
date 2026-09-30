# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/` with a small, focused module split: `src/cli.ts` is the command-line entrypoint, `src/dither.ts` handles image processing, and `src/generator.ts` builds blue-noise textures. Generated output goes to `dist/` (from `tsc`) and is gitignored. Local working folders `input/` and `output/` are also gitignored. Asset files for docs live in `img/`, and the default noise texture is `blue-noise.png` at the repo root. Core config is in `package.json`, `tsconfig.json`, `oxlint.config.ts`, `oxfmt.config.ts`, and `lefthook.yml`. The verification harness lives in `scripts/devkit/` (doctor, verify, feature map) with the feature map itself in `features/<id>/feature.json`.

## Build, Test, and Development Commands
- `npm install`: install dependencies.
- `npm run dither <input>`: run the CLI via `tsx` on a source image (outputs to `output/` by default).
- `npm run start generate -- --size 64 --sigma 1.9`: generate a blue-noise texture.
- `npm run build`: compile TypeScript to `dist/`.
- `npm run check:types`: typecheck only (no emit).
- `npm run lint`: check lint and formatting with Ultracite (oxlint + oxfmt).
- `npm run format`: auto-fix lint and formatting with Ultracite.
- `npm run doctor`: read-only check that Node and the `sharp` dependency are ready, with a fix for each failure.
- `npm run check-features`: validate `features/*/feature.json` against the source tree (unique ids, paths that exist, checks that reference a real script, every `src/` file owned by a feature).
- `npm run verify -- --since <ref>`: map files changed since `<ref>` (default `origin/main`) to features and run their checks, writing `.factory/proof.json`. `--all` skips the diff and runs every feature's checks instead; CI's push-to-main job picks this automatically (via `scripts/devkit/print-verify-args.ts`) whenever `github.event.before` is unusable (the branch's first push, or a force-push that rewrote history), so a push never silently diffs a commit against itself and checks nothing.
- `npm test`: Node's built-in test runner (`node --test`) over `scripts/devkit/*.test.ts` — unit tests for the feature map and the CI push-argument logic, plus one end-to-end test of `verify --all`. No test framework dependency.

## Coding Style & Naming Conventions
This repo uses TypeScript in ESM mode. Keep local imports with `.js` extensions (for example, `./dither.js`). Follow existing formatting: 2-space indentation, double quotes, and semicolons. Use `camelCase` for variables/functions, `PascalCase` for types/classes, and `SCREAMING_SNAKE_CASE` for constants. Ultracite (oxlint + oxfmt) is the source of truth for linting and formatting.

## Testing Guidelines
The CLI itself has no unit test framework; validate CLI changes by running `npm run check:types`, `npm run lint`, `npm run doctor`, and `npm run verify -- --since origin/main` (which builds and drives the CLI's `generate` and `dither` commands end to end). The verification harness in `scripts/devkit/` has its own tests: run `npm test`. If you add a new CLI command or module, add a `features/<id>/feature.json` describing it (copy an existing one) so `check-features` and `verify` cover it — remember that a dependency bump (`package.json`, `package-lock.json`) affects every feature that ships it, not just the harness, so list it in each of theirs.

## Commit & Pull Request Guidelines
Recent commits use short, plain-language summaries without prefixes. Keep messages concise and descriptive (for example, “Add generator seed option”). Lefthook runs `ultracite fix` on staged files at commit, so ensure formatting/linting passes before committing. CI runs `changeset status`, so add a changeset with `npx changeset` (or `npx changeset add --empty` for changes that need no release). For pull requests, include a brief description, the commands you ran, and (when output changes are visual) attach before/after images or link to generated files. Update `README.md` when CLI flags or usage change.
