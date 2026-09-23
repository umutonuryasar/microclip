/**
 * Assert the browser tokenizer matches the Python one.
 *
 * Fixtures are written by scripts/check_tokenizer_js.py: real COCO captions
 * plus the ids CaptionTokenizer produced for them. Any divergence in the BPE
 * merges, the byte mapping or the pre-tokenizer regex shows up as a mismatch.
 *
 * Usage: node scripts/check_tokenizer_js.mjs <fixtures.json>
 */
import { readFileSync } from "node:fs";
import { ByteLevelBPETokenizer } from "../demo/static/tokenizer.js";

const fixtures = JSON.parse(readFileSync(process.argv[2], "utf8"));
const tok = new ByteLevelBPETokenizer(JSON.parse(readFileSync(fixtures.spec_path, "utf8")));

let checked = 0;
const failures = [];
for (const { text, ids, mask } of fixtures.cases) {
  const got = tok.encode(text, fixtures.max_len);
  if (JSON.stringify(got.ids) !== JSON.stringify(ids) ||
      JSON.stringify(got.mask) !== JSON.stringify(mask)) {
    if (failures.length < 5) failures.push({ text, expected: ids.slice(0, 12), got: got.ids.slice(0, 12) });
  }
  checked++;
}

if (failures.length) {
  console.error(`MISMATCH on ${failures.length}+ of ${checked} cases`);
  for (const f of failures) console.error(JSON.stringify(f));
  process.exit(1);
}
console.log(`tokenizer parity ok — ${checked} cases, vocab ${tok.vocabSize}`);
