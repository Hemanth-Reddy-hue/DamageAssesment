.PHONY: setup run bench ablation fixloop test clean

PYTHON = python
CAPTURE ?= data/raw/sample_room/lidar/run1
OUT ?= out

setup:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) scripts/run_capture.py $(CAPTURE) --out $(OUT)

bench:
	$(PYTHON) bench/harness.py

ablation:
	$(PYTHON) bench/ablation_drift.py

fixloop:
	$(PYTHON) bench/harness.py --fixloop

test:
	$(PYTHON) -m pytest tests/

clean:
	rm -rf out __pycache__ .pytest_cache
