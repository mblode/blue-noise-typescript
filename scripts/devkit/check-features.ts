// `npm run check-features` step: fails when a feature.json names a script,
// file or spec that does not exist, or a source file belongs to no feature.
// Each line names the fix.
import path from "node:path";

import { loadFeatures, repoFiles, validate } from "./features.ts";

const root = path.resolve(import.meta.dirname, "../..");
const files = repoFiles(root);
const { loaded, problems } = loadFeatures(root, files);
problems.push(...validate(root, loaded, files));

if (problems.length > 0) {
  console.error(
    `Feature map (feature.json) problems:\n- ${problems.join("\n- ")}`
  );
  process.exit(1);
}

console.log(`check-features: ${loaded.length} feature.json ok`);
