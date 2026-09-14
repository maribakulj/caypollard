# Changelog

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
