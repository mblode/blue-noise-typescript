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
- `npm run verify -- --since <ref>`: map files changed since `<ref>` (default `origin/main`) to features and run their checks, writing `.factory/proof.json`.

## Coding Style & Naming Conventions
This repo uses TypeScript in ESM mode. Keep local imports with `.js` extensions (for example, `./dither.js`). Follow existing formatting: 2-space indentation, double quotes, and semicolons. Use `camelCase` for variables/functions, `PascalCase` for types/classes, and `SCREAMING_SNAKE_CASE` for constants. Ultracite (oxlint + oxfmt) is the source of truth for linting and formatting.

## Testing Guidelines
No unit test framework is configured yet. Validate changes by running `npm run check:types`, `npm run lint`, `npm run doctor`, and `npm run verify -- --since origin/main` (which builds and drives the CLI's `generate` and `dither` commands end to end). If you add a new command or module, add a `features/<id>/feature.json` describing it (copy an existing one) so `check-features` and `verify` cover it. If you add a unit test runner in the future, add it to `package.json` scripts and document the command here.

## Commit & Pull Request Guidelines
Recent commits use short, plain-language summaries without prefixes. Keep messages concise and descriptive (for example, “Add generator seed option”). Lefthook runs `ultracite fix` on staged files at commit, so ensure formatting/linting passes before committing. CI runs `changeset status`, so add a changeset with `npx changeset` (or `npx changeset add --empty` for changes that need no release). For pull requests, include a brief description, the commands you ran, and (when output changes are visual) attach before/after images or link to generated files. Update `README.md` when CLI flags or usage change.
