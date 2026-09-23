/**
 * Exercise the browser search path outside a browser.
 *
 * Runs the same tokenizer, the same int8 ONNX graphs and the same dot-product
 * ranking the page uses, against the built demo/_static bundle, and compares
 * the result with the reference ids produced by the PyTorch pipeline.
 *
 * Usage: python scripts/check_static.py   (writes the reference, then runs this)
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { ByteLevelBPETokenizer } from "../demo/static/tokenizer.js";

const reference = JSON.parse(readFileSync(process.argv[2], "utf8"));
const require = createRequire(process.argv[3] ?? import.meta.url);
const ort = require("onnxruntime-web");

const BUILD = new URL("../demo/_static/", import.meta.url);
const MAX_LEN = 64, DIM = 256, TOP_K = 8;

const spec = JSON.parse(readFileSync(new URL("assets/tokenizer/bpe16k.json", BUILD), "utf8"));
const index = JSON.parse(readFileSync(new URL("assets/index.json", BUILD), "utf8"));
const tokenizer = new ByteLevelBPETokenizer(spec);

function readEmbeddings(loss) {
  const buf = readFileSync(new URL(`assets/image_emb_${loss}.bin`, BUILD));
  const values = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (values.length !== index.length * DIM) throw new Error(`${loss}: bad embedding length`);
  return values;
}

function topMatches(images, queryVec) {
  const best = [];
  for (let i = 0; i < index.length; i++) {
    let dot = 0;
    const base = i * DIM;
    for (let d = 0; d < DIM; d++) dot += queryVec[d] * images[base + d];
    if (best.length < TOP_K || dot > best[best.length - 1].score) {
      const at = best.findIndex((b) => dot > b.score);
      best.splice(at === -1 ? best.length : at, 0, { index: i, score: dot });
      if (best.length > TOP_K) best.pop();
    }
  }
  return best;
}

let failures = 0;
for (const loss of ["softmax", "sigmoid"]) {
  const images = readEmbeddings(loss);
  const session = await ort.InferenceSession.create(
    new URL(`assets/text_encoder_${loss}.int8.onnx`, BUILD).pathname);

  for (const { query, expected } of reference[loss]) {
    const { ids, mask } = tokenizer.encode(query, MAX_LEN);
    const out = await session.run({
      token_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, MAX_LEN]),
      pad_mask: new ort.Tensor("int64", BigInt64Array.from(mask, BigInt), [1, MAX_LEN]),
    });
    const got = topMatches(images, out.embedding.data).map((m) => m.index);
    const overlap = got.filter((i) => expected.includes(i)).length;
    const status = got[0] === expected[0] && overlap >= 7 ? "ok" : "MISMATCH";
    if (status !== "ok") failures++;
    console.log(`${status.padEnd(9)} ${loss.padEnd(8)} top1 ${got[0] === expected[0] ? "same" : `${got[0]} vs ${expected[0]}`}`
      + `  overlap ${overlap}/${TOP_K}  "${query}"`);
  }
}
console.log(failures ? `\n${failures} mismatches` : "\nbrowser search path matches the PyTorch reference");
process.exit(failures ? 1 : 0);
