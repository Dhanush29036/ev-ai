.PHONY: help install pipeline test dashboard clean lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install package in editable mode"
	@echo "  make pipeline   - Run complete end-to-end ML & GIS pipeline"
	@echo "  make test       - Run test suite with pytest"
	@echo "  make dashboard  - Launch Streamlit dashboard"
	@echo "  make lint       - Run ruff linter and style checks"
	@echo "  make clean      - Clean cache and temporary files"

install:
	python -m pip install -e . --no-deps

pipeline:
	python -m evplan.run --stage all

test:
	python -m pytest tests/ -v

dashboard:
	streamlit run app/streamlit_app.py

lint:
	ruff check src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
