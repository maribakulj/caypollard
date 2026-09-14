# caypollard

**Research code and reproducible notebooks for testing whether structured cultural-heritage knowledge changes and improves visual similarity.**

Status: **Phase 1 implemented; Phase 2 in progress / v0.2.0**

## Core research question

> Can knowledge-graph structure improve the retrieval of iconographically related cultural-heritage images beyond what visual embeddings alone recover?

The repository is designed around a deliberately testable distinction between:

- **visual similarity**: images look alike;
- **iconographic similarity**: images express related subjects, motifs, symbols, or concepts;
- **historical/contextual similarity**: images are related through creators, books, editions, places, dates, collections, or other graph relations.

The project does **not** assume that these notions of similarity coincide. Their disagreement is the research object.

## Working hypothesis

A visual model can retrieve formal resemblance. A knowledge graph can encode expert and contextual relations. A multimodal model should be useful only if it retrieves historically or iconographically meaningful neighbours that a visual-only baseline misses, while avoiding semantically misleading visual look-alikes.

Formally, the project compares:

- `S_v(i, j)`: visual similarity;
- `S_g(i, j)`: graph similarity;
- `S_t(i, j)`: textual similarity;
- `S_m(i, j)`: multimodal similarity.

The central test concerns cases where `S_v` and `S_g` disagree.

## Corpus strategy

The project uses corpora for different experimental roles rather than forcing everything into one graph.

| Corpus | Role | Why it matters |
| --- | --- | --- |
| **Iconclass AI Test Set** | controlled benchmark | expert-assigned iconographic labels and a hierarchical semantic structure |
| **Emblematica Online (UIUC/HAB)** | historical case study | pictura + motto + text + bibliographic context + Iconclass |
| **Rijksmuseum** | cross-collection validation | large-scale museum data, IIIF, Linked Art / structured classifications |
| **WJoconde** | secondary MMKG benchmark | multimodal French cultural-heritage KG suitable for comparison with KGC literature |

### Experimental progression

```text
Iconclass AI
    │
    ├── visual baselines
    ├── graph baselines
    ├── hard positive / hard negative benchmark
    └── multimodal fusion
            │
            v
     Emblematica Online
            │
            v
   Rijksmuseum transfer test
            │
            v
     WJoconde comparison
```

## Main hypotheses

- **H1 — Complementarity:** graph representations contain information not recoverable from visual embeddings alone.
- **H2 — Hard positives:** graph-aware retrieval improves retrieval of images that are iconographically close but visually dissimilar.
- **H3 — Hard negatives:** graph-aware retrieval suppresses visually close but iconographically unrelated results.
- **H4 — Cross-collection transfer:** at least part of the gain survives transfer to a collection not used to fit the fusion model.
- **H5 — Simple baselines matter:** a learned multimodal model must outperform transparent baselines such as weighted late fusion and graph reranking to justify its complexity.

## Experiments

The planned model matrix is:

| ID | Vision | KG | Text | Purpose |
| --- | :---: | :---: | :---: | --- |
| V | ✓ |  |  | visual baseline |
| G |  | ✓ |  | graph baseline |
| T |  |  | ✓ | textual baseline |
| V+T | ✓ |  | ✓ | conventional multimodal baseline |
| V+G | ✓ | ✓ |  | test graph contribution to image retrieval |
| G+T |  | ✓ | ✓ | isolate graph/text interaction |
| V+G+T | ✓ | ✓ | ✓ | full model |

Initial representation families:

- vision: DINOv2, CLIP/OpenCLIP, SigLIP;
- graph: Node2Vec (control), RDF2Vec, ComplEx, RotatE;
- retrieval: exact cosine baseline, FAISS for indexed retrieval;
- fusion: weighted late fusion, graph reranking, learned shared projection;
- evaluation: Recall@K, MRR, mAP, nDCG, hierarchical Iconclass relevance.

## Why hard positives and hard negatives are central

Random nearest-neighbour examples are easy to make impressive and hard to interpret. This repository therefore treats disagreement cases as a first-class benchmark.

| | Iconographically close | Iconographically distant |
| --- | --- | --- |
| **Visually close** | easy positive | **hard negative** |
| **Visually distant** | **hard positive** | easy negative |

The two bold cells are the main scientific target.

## Leakage controls

Naive random image splits are not acceptable when collections contain copies, editions, reprints, or near-duplicate plates. Splits should therefore be evaluated by:

- book / work;
- edition;
- creator;
- collection;
- date range;
- and, where possible, visual near-duplicate groups.

The repository will keep split definitions as versioned data artifacts.

## Repository structure

```text
caypollard/
├── README.md
├── ROADMAP.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_iconclass_graph.ipynb              # executable
│   ├── 02_visual_embeddings.ipynb            # executable Phase-2 foundation
│   ├── 03_kg_embeddings.ipynb                # planned
│   ├── 04_visual_retrieval_baseline.ipynb    # planned
│   ├── 05_graph_retrieval_baseline.ipynb     # planned
│   ├── 06_multimodal_fusion.ipynb            # planned
│   ├── 07_hard_pairs_evaluation.ipynb        # planned
│   ├── 08_emblematica_case_study.ipynb       # planned
│   └── 09_cross_collection_transfer.ipynb    # planned
├── src/caypollard/
├── configs/
├── data/
│   ├── manifests/
│   └── samples/
├── results/
├── tests/
├── docs/
└── .github/workflows/
```

Notebooks are **interfaces to experiments**, not the implementation layer. Reusable logic belongs in `src/caypollard/` so that notebooks remain readable, testable, and executable from a clean environment.

## Phase 1 now implemented

The repository now includes a working Iconclass data layer rather than only a plan:

- parser for the official `data.json` annotation format;
- canonical deterministic JSONL manifests;
- parser for the open `iconclass/data` `notations.txt` hierarchy;
- SKOS export and parent/child edge export;
- hierarchy-aware semantic-distance baseline;
- deterministic item/group split utilities;
- exact-checksum and group leakage checks;
- explicit downloader for the ~3.1 GB official test set with MD5 verification;
- pinned vocabulary downloader with provenance sidecar;
- `01_iconclass_graph.ipynb`, executable entirely from tiny repository fixtures;
- a frozen pre-results protocol in `docs/protocol-v0.1.md`.

### Phase 2 foundation now included

The repository also contains the first visual-baseline infrastructure: a lazy Hugging Face encoder adapter for DINOv2, CLIP, and SigLIP; a provenance-bearing NPZ+JSON embedding format; exact cosine retrieval with query self-match exclusion; and an executable `02_visual_embeddings.ipynb` smoke test. Full heritage-model runs remain explicit because neither model weights nor the 3.1 GB image archive belong in CI.

The full image corpus is intentionally not committed or downloaded by CI. See `docs/ICONCLASS_DATA_CARD.md`.

## Quick start

The project uses Python 3.11+ and is prepared for `uv`.

```bash
git clone <repository-url>
cd caypollard
uv sync --extra dev
uv run pytest
uv run jupyter lab
```

A minimal sample dataset will be kept in `data/samples/`. Full collections should be reconstructed from versioned manifests and source identifiers rather than committed to Git.

### Phase-1 smoke run

Without downloading the full image archive:

```bash
uv run python scripts/build_iconclass_benchmark.py \
  data/samples/iconclass_testset_excerpt.json \
  --notations data/samples/iconclass_notations_fixture.txt \
  --output-dir /tmp/iconclass-smoke

uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_iconclass_graph.ipynb \
  --output /tmp/01_iconclass_graph.executed.ipynb
```

For the real corpus, read `docs/ICONCLASS_DATA_CARD.md` and `docs/protocol-v0.1.md` first. The project deliberately refuses to make a 3.1 GB research-data download an invisible side effect of setup.

## Reproducibility and FAIR principles

The project will prefer **identifiers and manifests over copied heritage assets**. Dataset manifests should record, where available:

- persistent object identifier;
- source institution;
- IIIF manifest / image service;
- source metadata URL;
- rights and licence statement;
- retrieval timestamp;
- checksum for downloaded derivatives;
- transformation history;
- labels / concepts used by the experiment;
- split membership.

Large embeddings and derived datasets should be versioned outside GitHub, ideally through a DOI-bearing research repository, while GitHub retains code, schemas, manifests, configuration, and small reproducible samples.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

The research positioning is maintained in [`docs/LITERATURE.md`](docs/LITERATURE.md), with a publication-grade search and screening procedure in [`docs/LITERATURE_REVIEW_PROTOCOL.md`](docs/LITERATURE_REVIEW_PROTOCOL.md). A machine-readable bibliography is provided in [`references.bib`](references.bib).

## Scientific success criteria

The project will not treat a visually pleasing demo as sufficient evidence. Before a more complex joint model is justified, it should meet all of the following:

1. outperform the strongest visual-only baseline on a predeclared semantic retrieval metric;
2. outperform or materially complement simple late fusion;
3. show improvement specifically on hard-positive and/or hard-negative subsets;
4. remain useful under at least one leakage-resistant split;
5. show at least partial cross-collection transfer;
6. provide interpretable evidence for why KG information changed a ranking;
7. report failures, not just favourable examples.

## Non-goals for the first release

The first release will **not** attempt to:

- create a universal ontology of cultural heritage;
- replace Iconclass or curatorial description;
- infer historical influence from embedding similarity alone;
- deploy a production graph database or distributed vector stack;
- train a foundation model from scratch;
- claim that vector proximity is equivalent to art-historical interpretation.

The initial target is narrower: determine whether graph-structured heritage knowledge makes measurable, interpretable, and transferable contributions to image retrieval.

## Research outputs envisaged

1. reproducible benchmark and notebooks;
2. curated hard-positive / hard-negative evaluation set;
3. multimodal retrieval baselines;
4. Emblematica historical case study;
5. cross-collection evaluation;
6. citable software/data release;
7. research paper on graph-conditioned iconographic retrieval;
8. optional interactive demonstrator exposing visual, graph, and fused rankings side by side.

## Roadmap

The detailed roadmap, milestones, exit criteria, risks, and release targets are in [`ROADMAP.md`](ROADMAP.md).

## Licence

Code in this repository is released under the **MIT License**. Source images, metadata, vocabularies, and external datasets retain their own rights and licences; see dataset-specific provenance documentation before redistribution.
