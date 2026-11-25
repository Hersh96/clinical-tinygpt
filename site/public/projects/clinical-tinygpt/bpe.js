// Byte-level BPE in the browser: the same algorithm as ctg/tokenizer/bpe.py,
// reading the merges.json the Python tokenizer saved. Used by the
// tokenizer playground so visitors can see how any sentence is split.
const utf8 = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });

// Python's `regex` pattern from ctg/tokenizer/bpe.py, written for JS (u flag for \p{...}).
const GPT2_PATTERN = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export function loadMerges(data) {
  const ranks = new Map();
  const vocab = [];
  for (let i = 0; i < 256; i++) vocab.push(new Uint8Array([i]));
  data.merges.forEach(([a, b], i) => {
    ranks.set(a * 65536 + b, i);
    const left = vocab[a], right = vocab[b];
    const joined = new Uint8Array(left.length + right.length);
    joined.set(left); joined.set(right, left.length);
    vocab.push(joined);
  });
  const cache = new Map();

  function encodeChunk(chunk) {
    if (cache.has(chunk)) return cache.get(chunk);
    let ids = Array.from(utf8.encode(chunk));
    while (ids.length >= 2) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = ids[best], b = ids[best + 1], id = 256 + bestRank, out = [];
      for (let i = 0; i < ids.length; ) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(id); i += 2; }
        else { out.push(ids[i]); i += 1; }
      }
      ids = out;
    }
    if (cache.size < 50000) cache.set(chunk, ids);
    return ids;
  }

  function encode(text) {
    const out = [];
    for (const m of text.matchAll(GPT2_PATTERN)) out.push(...encodeChunk(m[0]));
    return out;
  }

  return {
    vocabSize: data.vocab_size,
    encode,
    tokens: (text) => encode(text).map((id) => decoder.decode(vocab[id])),
    tokenBytes: (id) => vocab[id],
  };
}
