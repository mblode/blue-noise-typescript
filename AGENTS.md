# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/` with a small, focused module split: `src/cli.ts` is the command-line entrypoint, `src/dither.ts` handles image processing, and `src/generator.ts` builds blue-noise textures. Generated output goes to `dist/` (from `tsc`) and is gitignored. Local working folders `input/` and `output/` are also gitignored. Asset files for docs live in `img/`, and the default noise texture is `blue-noise.png` at the repo root. Core config is in `package.json`, `tsconfig.json`, and `biome.jsonc`.

## Build, Test, and Development Commands
- `npm install`: install dependencies.
- `npm run dither <input>`: run the CLI via `tsx` on a source image (outputs to `output/` by default).
- `npm run start generate -- --size 64 --sigma 1.9`: generate a blue-noise texture.
- `npm run build`: compile TypeScript to `dist/`.
- `npm run check-types`: typecheck only (no emit).
- `npm run lint` / `npm run lint:fix`: run Biome checks (optionally auto-fix).
- `npm run format` / `npm run format:check`: format source with Biome.

## Coding Style & Naming Conventions
This repo uses TypeScript in ESM mode. Keep local imports with `.js` extensions (for example, `./dither.js`). Follow existing formatting: 2-space indentation, double quotes, and semicolons. Use `camelCase` for variables/functions, `PascalCase` for types/classes, and `SCREAMING_SNAKE_CASE` for constants. Biome (with the Ultracite preset) is the source of truth for linting and formatting.

## Testing Guidelines
No automated test framework or coverage tooling is configured yet. For now, validate changes by running `npm run check-types`, `npm run lint`, and a manual CLI run on a sample image (for example, `npm run dither input/example.jpg`). If you add tests in the future, add the runner to `package.json` scripts and document the command here.

## Commit & Pull Request Guidelines
Recent commits use short, plain-language summaries without prefixes. Keep messages concise and descriptive (for example, “Add generator seed option”). Husky runs `lint-staged` on commit, so ensure formatting/linting passes before committing. For pull requests, include a brief description, the commands you ran, and (when output changes are visual) attach before/after images or link to generated files. Update `README.md` when CLI flags or usage change.
