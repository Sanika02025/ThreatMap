.PHONY: install test lint run download-attack build-index evaluate clean

install:
	pip install -e .

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

download-attack:
	python scripts/download_attack_data.py

build-index:
	python scripts/build_index.py

evaluate:
	python scripts/run_evaluation.py

run:
	streamlit run app.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
