# Changelog

## 0.4.0 — 2026-09-14

### Added

- frozen-encoder learned visual/KG alignment with linear or one-hidden-layer projection heads;
- symmetric InfoNCE with same-object positives and in-batch negatives;
- strict train/validation partitioning, early stopping, partition digests, and no test-driven checkpoint selection;
- learned-embedding collapse diagnostics using off-diagonal cosine statistics and effective rank;
- `protocol-v0.4`, `configs/alignment.yaml`, `scripts/train_alignment.py`, and executable `06b_learned_joint_alignment.ipynb`;
- dedicated `alignment` optional dependency group for PyTorch.

### Changed

- advanced package and citation metadata to v0.4.0;
- extended CI to execute the learned-alignment notebook with the alignment dependency group;
- declared an explicit ruff rule set and pinned the linter to a minor series so the lint
  contract belongs to the project rather than to whichever release CI resolves;
- split CI into a fast fixture job and a separate PyTorch alignment job, and moved the
  notebook list into `make notebooks` / `make notebooks-alignment` so CI and local runs
  execute the same set;
- recorded the full author name in `LICENSE` and `CITATION.cff`.

### Fixed

- `neighbor_overlap_at_k` annotated `EmbeddingTable` without importing it at module scope,
  leaving the annotation unresolvable to `typing.get_type_hints` and documentation tooling;
- removed a tautological `ndcg_at_10` conditional left over from the protocol-v0.2 endpoint
  freeze;
- `mean_average_precision` paired rankings with totals using a non-strict `zip`.

## 0.3.0 — 2026-09-14

### Added

- deterministic Node2Vec/DeepWalk-style PPMI-SVD and predicate-aware RDF2Vec-style graph controls;
- optional PyKEEN ComplEx/RotatE training adapter with provenance-bearing entity embeddings;
- canonical relation-triple IO, projection checksums, G2 target-edge masking, and projection audits;
- graph degree / embedding-neighbour hubness diagnostics;
- protocol-v0.3 hard-pair calibration and balanced mining with canonical frozen artifacts;
- validation-only late-fusion calibration, alpha selection, weighted-concatenation ranking, and graph reranking;
- executable `06_multimodal_fusion.ipynb` and `07_hard_pairs_evaluation.ipynb`;
- CLI entry points for generic KG embedding, hard-pair mining, and transparent fusion evaluation.

### Changed

- declared SciPy as a direct dependency rather than relying on scikit-learn to install it transitively;
- extended CI to execute eight committed research notebooks;
- updated graph and fusion roadmaps to distinguish implemented infrastructure from external-data-dependent result generation.

## 0.2.0 — 2026-09-14

### Changed

- unified the repository, Python distribution, import namespace, notebooks, scripts, documentation, and citation identity under `caypollard`;
- reset the local Git history so the current project identity contains no reachable legacy-branded files or commits.
- added protocol-aligned visual retrieval evaluation with hierarchical nDCG@10, Recall@1/5/10, MRR, mAP, coverage, and fixed depth/frequency strata;
- added executable `04_visual_retrieval_baseline.ipynb` and a CLI producing summary, per-query JSONL, and CSV artifacts;
- added an optional exact FAISS `IndexFlatIP` ranking backend with deterministic post-ordering.
- generalized embedding persistence so visual and graph representations share one storage layer;
- added an adjacency-SVD Iconclass taxonomy control, concept-to-image graph pooling, and visual/graph neighbour-overlap metric;
- added executable `03_kg_embeddings.ipynb` and `05_graph_retrieval_baseline.ipynb`;
- added protocol v0.2 and explicit graph projections preventing target Iconclass leakage from being mistaken for KG evidence.

## 0.1.1 — 2026-09-14

### Added

- working Iconclass AI `data.json` loader, audit, and canonical manifest builder;
- parser for the open Iconclass `notations.txt` hierarchy;
- SKOS graph export and transparent hierarchy-distance relevance;
- deterministic item/group splitting with exact-checksum and group leakage checks;
- guarded official 3.1 GB test-set downloader with published MD5 verification;
- pinned Iconclass core-data downloader with provenance metadata;
- executable `01_iconclass_graph.ipynb`;
- Iconclass data card and frozen pre-results protocol;
- CI execution of both committed notebooks.
- verified literature map, BibTeX bibliography, and literature-review protocol;
- average precision, mean average precision, and full-pool-aware nDCG evaluation utilities.
- Phase-2 visual embedding store, DINOv2/CLIP/SigLIP Hugging Face adapter, and exact cosine retrieval;
- executable `02_visual_embeddings.ipynb` offline smoke test and real-extraction command path;
- package/version metadata consistency fix.

## 0.1.0 — 2026-09-14

### Added

- initial research question and hypotheses;
- complete phased roadmap with decision gates;
- research and methodology documentation;
- FAIR/provenance/reproducibility policy;
- MIT licence and citation metadata;
- notebook plan and project overview notebook;
- transparent late-fusion utilities;
- basic retrieval metric utilities and tests;
- GitHub Actions CI skeleton;
- bootstrap publication script for GitHub CLI.
