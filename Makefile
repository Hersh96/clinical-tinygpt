# Run targets in this order on the laptop: data tokenizer tokenizer-compare tokens sweep bench eval export
PY ?= python
.PHONY: test data tokenizer
test:
	$(PY) -m pytest -q
data:
	$(PY) -m ctg.data.prepare --config configs/data.yaml --out data/processed
tokenizer:
	$(PY) -m ctg.tokenizer.train --config configs/tokenizer.yaml
