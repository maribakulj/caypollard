.PHONY: test lint notebooks notebooks-alignment phase1-smoke viewer-build sketch-build viewer

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

# The viewer: a local page that shows the pool as thumbnails and, for any picture,
# the neighbours each representation returns. `viewer-build` writes everything it
# reads (thumbnails, item table, neighbour lists) under data/derived/viewer/;
# `viewer` only serves files. Rerun the build when a table changes.
VIEWER_MANIFESTS := data/derived/museums-v0.2/manifest.jsonl data/derived/emblematica-eval/manifest-cropped-paths.jsonl
VIEWER_TABLES := \
	--table "pixels=data/derived/shapes/pixels-mixte.npz" \
	--table "nœuds nommés=data/derived/shapes/nodes-mixte.npz" \
	--table "formes muettes=data/derived/shapes/v2-256/shape-signs.npz" \
	--table "palette=data/derived/shapes/palette-mixte.npz" \
	--table "signal=data/derived/shapes/signal-mixte.npz" \
	--table "composition=data/derived/shapes/composition-v3.npz" \
	--table "record (formes + composition + répétition)=data/derived/shapes/record-v3.npz" \
	--table "silhouette=data/derived/museums-v0.2/embeddings/dinov2-silhouette.npz" \
	--table "tout mélangé=data/derived/shapes/mix-tout.npz"

viewer-build:
	uv run python scripts/fetch_wikidata_labels.py data/derived/museums-v0.2/manifest.jsonl
	uv run python scripts/build_viewer.py $(VIEWER_MANIFESTS) $(VIEWER_TABLES) \
		--nodes data/derived/shapes/nodes-musees-vlm.jsonl --nodes data/derived/shapes/nodes-emblemes-vlm.jsonl
	uv run python scripts/fit_demo_models.py

# The stroke channels: every picture's contour drawing and its stroke signs, then the drawings
# through DINOv2. About ten minutes on this machine; rerun when caypollard.sketch changes.
sketch-build:
	uv run python scripts/build_sketch_channel.py \
		"musees=data/derived/museums-v0.2/manifest.jsonl" \
		"emblemes=data/derived/emblematica-eval/manifest-cropped-paths.jsonl" \
		--renders data/derived/sketch/renders --output data/derived/sketch
	uv run python scripts/embed_images.py data/derived/sketch/renders-manifest.jsonl \
		data/derived/sketch/renders data/derived/sketch/dinov2-sketch.npz --preset dinov2-base

viewer:
	uv run python scripts/serve_viewer.py
