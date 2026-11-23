import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadMerges } from "../public/projects/clinical-tinygpt/bpe.js";

const dir = new URL("./fixtures/", import.meta.url);
const cases = JSON.parse(readFileSync(new URL("bpe_cases.json", dir), "utf8"));

for (const v of Object.keys(cases)) {
  const tok = loadMerges(JSON.parse(readFileSync(new URL(`merges_${v}.json`, dir), "utf8")));
  test(`encode matches Python for vocab ${v}`, () => {
    for (const c of cases[v]) assert.deepEqual(tok.encode(c.text), c.ids, JSON.stringify(c.text));
  });
  test(`token pieces match Python for ASCII text, vocab ${v}`, () => {
    for (const c of cases[v]) {
      if (/^[\x00-\x7f]*$/.test(c.text)) assert.deepEqual(tok.tokens(c.text), c.pieces);
    }
  });
}

test("schema wrapper from the export script is accepted", () => {
  const raw = JSON.parse(readFileSync(new URL("merges_300.json", dir), "utf8"));
  const tok = loadMerges({ schema_version: 1, generated_at: "x", ...raw });
  assert.deepEqual(tok.encode("Pt denies chest pain."), cases["300"][2].ids);
});
