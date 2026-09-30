// Prints the `npm run verify` arguments for the current CI trigger, from
// GITHUB_EVENT_NAME and GITHUB_EVENT_BEFORE. See verify-args.ts.
import path from "node:path";

import { verifyArgs } from "./verify-args.ts";

const root = path.resolve(import.meta.dirname, "../..");
const args = verifyArgs(
  {
    before: process.env.GITHUB_EVENT_BEFORE,
    eventName: process.env.GITHUB_EVENT_NAME ?? "",
  },
  root
);
console.log(args.join(" "));
