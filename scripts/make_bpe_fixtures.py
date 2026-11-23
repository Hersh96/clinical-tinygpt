"""Write fixtures so the browser BPE (site/.../bpe.js) can be checked against Python."""
import json
from pathlib import Path

from ctg.tokenizer.bpe import BPETokenizer

CASES = ["", "a", "Pt denies chest pain.", "Metformin 500 mg PO BID", "µg °C – naïve 🙂",
         "HbA1c 7.2%; BP 140/90 mmHg", "  leading and trailing  \n\n", "don't won't it's", "x" * 300,
         "Lisinopril 10mg daily\tand  hydrochlorothiazide", "日本語のテキスト", "status post CABG x3"]
out = Path("site/tests/fixtures")
notes = [json.loads(l)["note"] for l in open("tests/fixtures/notes.jsonl")]
cases = {}
for v in (300, 400, 600):
    tok = BPETokenizer()
    tok.train(notes * 3, v)
    tok.save(out / f"merges_{v}.json")
    cases[str(v)] = [{"text": c, "ids": tok.encode(c), "pieces": tok.token_strings(c)} for c in CASES]
(out / "bpe_cases.json").write_text(json.dumps(cases, ensure_ascii=False))
