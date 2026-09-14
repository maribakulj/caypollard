.PHONY: test lint notebooks notebooks-alignment phase1-smoke

# Notebooks that run from repository fixtures with the dev extra alone.
FIXTURE_NOTEBOOKS := \
	00_project_overview \
	01_iconclass_graph \
	02_visual_embeddings \
	03_kg_embeddings \
	04_visual_retrieval_baseline \
	05_graph_retrieval_baseline \
	06_multimodal_fusion \
	07_hard_pairs_evaluation

# Notebooks that additionally require the `alignment` extra (PyTorch).
ALIGNMENT_NOTEBOOKS := 06b_learned_joint_alignment

NBCONVERT = uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

notebooks:
	@for nb in $(FIXTURE_NOTEBOOKS); do \
		echo "executing $$nb"; \
		$(NBCONVERT) notebooks/$$nb.ipynb --output /tmp/$$nb.executed.ipynb || exit 1; \
	done

notebooks-alignment:
	@for nb in $(ALIGNMENT_NOTEBOOKS); do \
		echo "executing $$nb"; \
		$(NBCONVERT) notebooks/$$nb.ipynb --output /tmp/$$nb.executed.ipynb || exit 1; \
	done

phase1-smoke:
	uv run python scripts/build_iconclass_benchmark.py data/samples/iconclass_testset_excerpt.json --notations data/samples/iconclass_notations_fixture.txt --output-dir /tmp/iconclass-smoke
