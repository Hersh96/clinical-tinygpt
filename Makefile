# Run targets in this order on the laptop: data tokenizer tokenizer-compare tokens sweep bench eval export
PY ?= python
.PHONY: test data tokenizer tokenizer-compare tokens
test:
	$(PY) -m pytest -q
data:
	$(PY) -m ctg.data.prepare --config configs/data.yaml --out data/processed
tokenizer:
	$(PY) -m ctg.tokenizer.train --config configs/tokenizer.yaml
tokenizer-compare:
	$(PY) -m ctg.experiments.tokenizer_compare
tokens:
	$(PY) -m ctg.data.tokens --tokenizer artifacts/tokenizer/8192/merges.json --out data/tokens/8192
