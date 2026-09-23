/**
 * MicroCLIP static demo: text-to-image search entirely in the browser.
 *
 * The image embeddings are precomputed, so a query is one text-encoder pass
 * (int8 ONNX, via onnxruntime-web) plus a dot product against 5,000 vectors.
 * Both models are loaded so every query can be answered by each of them.
 */
// onnxruntime-web is vendored rather than loaded from a CDN: a static host may
// block third-party scripts, and a CDN failure would leave the page stuck with
// no way to report why.
import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs";
import { ByteLevelBPETokenizer } from "./tokenizer.js";

// Single-threaded: threads need SharedArrayBuffer, which needs cross-origin
// isolation that static hosts generally do not set.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;

const MAX_LEN = 64;   // must match data.max_text_len in the training config
const EMBED_DIM = 256;
const TOP_K = 8;
const LOSSES = ["softmax", "sigmoid"];

const EXAMPLES = [
  "a man riding a surfboard on a wave",
  "two dogs playing in the snow",
  "a plate of pizza on a wooden table",
  "a red double decker bus on a city street",
  "a giraffe standing next to a tree",
  "a child holding an umbrella in the rain",
  "a kitchen with white cabinets and a window",
  "a tennis player about to hit the ball",
];

const el = {
  query: document.getElementById("query"),
  go: document.getElementById("go"),
  status: document.getElementById("status"),
  tokens: document.getElementById("tokens"),
  examples: document.getElementById("examples"),
  grids: Object.fromEntries(LOSSES.map((l) => [l, document.getElementById(`grid-${l}`)])),
};

const state = { tokenizer: null, index: null, sessions: {}, embeddings: {} };

// A silent failure looks identical to a slow load, so surface everything.
window.addEventListener("error", (e) => setStatus(`Error: ${e.message}`, true));
window.addEventListener("unhandledrejection", (e) =>
  setStatus(`Error: ${e.reason?.message ?? e.reason}`, true));

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.style.color = isError ? "#d33d3d" : "";
}

/** Fill the grids with grey boxes so the layout does not jump on first search. */
function renderPlaceholders() {
  for (const loss of LOSSES) {
    el.grids[loss].innerHTML = Array.from({ length: TOP_K })
      .map(() => '<div class="placeholder"></div>').join("");
  }
}

async function fetchChecked(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response;
}

async function fetchEmbeddings(url, count) {
  const buffer = await (await fetchChecked(url)).arrayBuffer();
  const values = new Float32Array(buffer);
  if (values.length !== count * EMBED_DIM) {
    throw new Error(`${url}: expected ${count * EMBED_DIM} floats, got ${values.length}`);
  }
  return values;
}

async function load() {
  renderPlaceholders();

  setStatus("Loading tokenizer and image index…");
  const [spec, index] = await Promise.all([
    fetchChecked("assets/tokenizer/bpe16k.json").then((r) => r.json()),
    fetchChecked("assets/index.json").then((r) => r.json()),
  ]);
  state.tokenizer = new ByteLevelBPETokenizer(spec);
  state.index = index;

  let done = 0;
  const steps = LOSSES.length * 2;
  const tick = (what) => setStatus(`Loading ${what}… (${++done}/${steps})`);

  for (const loss of LOSSES) {
    tick(`${loss} image embeddings`);
    state.embeddings[loss] = await fetchEmbeddings(`assets/image_emb_${loss}.bin`, index.length);
    tick(`${loss} text encoder`);
    // Fetch the graph ourselves so an HTTP failure reports the URL, which an
    // InferenceSession given a path would swallow into a generic wasm error.
    const graph = await (await fetchChecked(`assets/text_encoder_${loss}.int8.onnx`)).arrayBuffer();
    state.sessions[loss] = await ort.InferenceSession.create(
      new Uint8Array(graph), { executionProviders: ["wasm"] });
  }

  el.go.disabled = false;
  setStatus(`Ready — ${index.length.toLocaleString()} images indexed. Type a query or pick an example.`);
}

/** Encode the query with one model and return its unit-length embedding. */
async function encodeQuery(loss, text) {
  const { ids, mask } = state.tokenizer.encode(text, MAX_LEN);
  const feeds = {
    token_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, MAX_LEN]),
    pad_mask: new ort.Tensor("int64", BigInt64Array.from(mask, BigInt), [1, MAX_LEN]),
  };
  const out = await state.sessions[loss].run(feeds);
  return out.embedding.data;
}

/** Top-K image indices by cosine similarity. Both sides are already normalized. */
function topMatches(loss, queryVec) {
  const images = state.embeddings[loss];
  const best = [];
  for (let i = 0; i < state.index.length; i++) {
    let dot = 0;
    const base = i * EMBED_DIM;
    for (let d = 0; d < EMBED_DIM; d++) dot += queryVec[d] * images[base + d];
    // Keep a small sorted list rather than scoring-then-sorting 5,000 entries.
    if (best.length < TOP_K || dot > best[best.length - 1].score) {
      const entry = { index: i, score: dot };
      const at = best.findIndex((b) => dot > b.score);
      best.splice(at === -1 ? best.length : at, 0, entry);
      if (best.length > TOP_K) best.pop();
    }
  }
  return best;
}

function render(loss, matches) {
  el.grids[loss].innerHTML = matches.map(({ index, score }) => {
    const item = state.index[index];
    const caption = item.caption.replace(/[<>&]/g, (c) =>
      ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[c]);
    return `<figure>
      <img src="assets/thumbs/${item.file}" alt="${caption}" loading="lazy" />
      <figcaption><b>${score.toFixed(3)}</b> ${caption}</figcaption>
    </figure>`;
  }).join("");
}

function showTokens(text) {
  const pieces = state.tokenizer.pieces(text);
  el.tokens.innerHTML = `<strong>${pieces.length} tokens</strong> (plus [BOS] and [EOS]): `
    + pieces.map((p) => `<code>${p.replace(/Ġ/g, "_")
        .replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[c])}</code>`).join("");
}

async function search() {
  const text = el.query.value.trim();
  if (!text) {
    setStatus("Type a query first.");
    return;
  }
  el.go.disabled = true;
  setStatus("Searching…");
  showTokens(text);
  try {
    const started = performance.now();
    for (const loss of LOSSES) {
      render(loss, topMatches(loss, await encodeQuery(loss, text)));
    }
    setStatus(`Searched ${state.index.length.toLocaleString()} images in `
      + `${Math.round(performance.now() - started)} ms, with both models.`);
  } catch (error) {
    setStatus(`Search failed: ${error.message}`, true);
    throw error;
  } finally {
    el.go.disabled = false;
  }
}

el.examples.innerHTML = EXAMPLES
  .map((e) => `<button class="chip" type="button">${e}</button>`).join("");
el.examples.addEventListener("click", (event) => {
  if (!event.target.classList.contains("chip") || el.go.disabled) return;
  el.query.value = event.target.textContent;
  search();
});
el.go.addEventListener("click", search);
el.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !el.go.disabled) search();
});

load().catch((error) => {
  setStatus(`Could not load the models: ${error.message}`, true);
  throw error;
});
