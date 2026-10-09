BACKEND ?= cpu
PORT ?= 8000

.PHONY: install train serve test lint docs docs-build
install:
	uv sync --extra $(BACKEND) --group docs
train:
	uv run --extra $(BACKEND) python -m m5_forecast.train
serve:
	uv run --extra $(BACKEND) streamlit run app.py
test:
	uv run --extra $(BACKEND) pytest
lint:
	uv run --extra $(BACKEND) ruff check .
docs:
	uv run --extra $(BACKEND) --group docs mkdocs serve -a 127.0.0.1:$(PORT)
docs-build:
	uv run --extra $(BACKEND) --group docs mkdocs build --strict
