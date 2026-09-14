.PHONY: test lint notebooks phase1-smoke

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

notebooks:
	uv run jupyter nbconvert --to notebook --execute notebooks/00_project_overview.ipynb --output /tmp/00_project_overview.executed.ipynb --ExecutePreprocessor.timeout=120
	uv run jupyter nbconvert --to notebook --execute notebooks/01_iconclass_graph.ipynb --output /tmp/01_iconclass_graph.executed.ipynb --ExecutePreprocessor.timeout=120
	uv run jupyter nbconvert --to notebook --execute notebooks/02_visual_embeddings.ipynb --output /tmp/02_visual_embeddings.executed.ipynb --ExecutePreprocessor.timeout=120

phase1-smoke:
	uv run python scripts/build_iconclass_benchmark.py data/samples/iconclass_testset_excerpt.json --notations data/samples/iconclass_notations_fixture.txt --output-dir /tmp/iconclass-smoke
