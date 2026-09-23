/**
 * Byte-level BPE tokenizer, reading the same bpe16k.json the Python demo uses.
 *
 * The training tokenizer is a plain HF `tokenizers` BPE with a ByteLevel
 * pre-tokenizer, no normalizer and no post-processor, so re-implementing it is
 * about a hundred lines and avoids shipping a multi-megabyte library to do it.
 * scripts/check_tokenizer_js.mjs asserts this file agrees with Python on
 * thousands of real COCO captions.
 *
 * Special ids match src/microclip/data/tokenizer.py: PAD 0, BOS 1, EOS 2, UNK 3.
 */

export const PAD = 0, BOS = 1, EOS = 2, UNK = 3;

// GPT-2 pre-tokenization pattern, as used by ByteLevel(use_regex=true).
const PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** GPT-2 byte-to-unicode table: maps each byte to a printable character. */
function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const table = new Array(256);
  for (let i = 0; i < bs.length; i++) table[bs[i]] = String.fromCodePoint(cs[i]);
  return table;
}

export class ByteLevelBPETokenizer {
  /** @param {object} spec parsed bpe16k.json */
  constructor(spec) {
    this.vocab = spec.model.vocab;
    this.addPrefixSpace = spec.pre_tokenizer?.add_prefix_space ?? true;
    this.byteEncoder = bytesToUnicode();
    this.encoderUtf8 = new TextEncoder();
    this.ranks = new Map();
    spec.model.merges.forEach((merge, i) => {
      // Newer tokenizer files store merges as pairs; older ones as "a b".
      const [a, b] = Array.isArray(merge) ? merge : merge.split(" ");
      this.ranks.set(a + "\u0000" + b, i);
    });
    this.cache = new Map();
  }

  get vocabSize() {
    return Object.keys(this.vocab).length;
  }

  /** Apply merges to one pre-token, returning its sub-word pieces. */
  bpe(token) {
    const hit = this.cache.get(token);
    if (hit) return hit;

    let word = Array.from(token);
    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(word[i] + "\u0000" + word[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      word = [
        ...word.slice(0, bestIdx),
        word[bestIdx] + word[bestIdx + 1],
        ...word.slice(bestIdx + 2),
      ];
    }
    this.cache.set(token, word);
    return word;
  }

  /** Prefix space is added only to non-empty input, as HF ByteLevel does. */
  prepare(text) {
    if (text.length === 0) return "";
    return this.addPrefixSpace && !text.startsWith(" ") ? " " + text : text;
  }

  /** Text -> token id array, with no special tokens added. */
  encodeRaw(text) {
    const input = this.prepare(text);
    const ids = [];
    for (const match of input.matchAll(PATTERN)) {
      // Byte-level: encode as UTF-8, then map each byte to its printable char.
      let mapped = "";
      for (const byte of this.encoderUtf8.encode(match[0])) {
        mapped += this.byteEncoder[byte];
      }
      for (const piece of this.bpe(mapped)) {
        const id = this.vocab[piece];
        ids.push(id === undefined ? UNK : id);
      }
    }
    return ids;
  }

  /** Same contract as CaptionTokenizer.encode: [BOS] ids [EOS], padded. */
  encode(text, maxLen) {
    const ids = [BOS, ...this.encodeRaw(text).slice(0, maxLen - 2), EOS];
    const mask = new Array(ids.length).fill(1);
    while (ids.length < maxLen) {
      ids.push(PAD);
      mask.push(0);
    }
    return { ids, mask };
  }

  /** Human-readable pieces, for the "how your query is tokenized" display. */
  pieces(text) {
    const input = this.prepare(text);
    const out = [];
    for (const match of input.matchAll(PATTERN)) {
      let mapped = "";
      for (const byte of this.encoderUtf8.encode(match[0])) {
        mapped += this.byteEncoder[byte];
      }
      out.push(...this.bpe(mapped));
    }
    return out;
  }
}
