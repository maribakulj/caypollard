# Notebooks

Notebooks are readable experiment interfaces. Reusable logic belongs in `src/caypollard/` and is covered by tests.

| Notebook | Status | Purpose |
| --- | --- | --- |
| `00_project_overview.ipynb` | executable | research design and transparent fusion smoke test |
| `01_iconclass_graph.ipynb` | **executable** | official annotation format, manifest, hierarchy parser, graded relevance, deterministic split smoke test |
| `02_visual_embeddings.ipynb` | **executable, Phase-2 foundation** | DINOv2 / CLIP / SigLIP extraction contract, embedding persistence, exact retrieval smoke test |
| `03_kg_embeddings.ipynb` | **executable** | taxonomy control, relation-aware walk controls, and G2 target masking |
| `04_visual_retrieval_baseline.ipynb` | **executable** | protocol-aligned visual-only evaluation and disagreement inspection |
| `05_graph_retrieval_baseline.ipynb` | **executable** | graph retrieval, visual/graph overlap, and degree/hubness diagnostics |
| `06_multimodal_fusion.ipynb` | **executable** | validation-calibrated late fusion and graph reranking smoke test |
| `07_hard_pairs_evaluation.ipynb` | **executable** | pre-fusion visual/iconographic disagreement benchmark |
| `08_emblematica_case_study.ipynb` | planned | historical emblem case study |
| `09_cross_collection_transfer.ipynb` | planned | external collection evaluation |

## Execution policy

Every committed notebook must run from a clean environment against repository fixtures or explicitly documented external data. Full external corpora are never implicit prerequisites for CI.
