# Research Roadmap

This roadmap is organised as a sequence of **decision gates**, not merely a feature list. Each phase must produce evidence that justifies the next layer of complexity.

## Guiding rule

> Do not build a sophisticated multimodal architecture until simple baselines demonstrate that the knowledge graph contributes complementary signal.

---

## Phase 0 — Research design and repository foundation

**Target release:** `v0.1.0`

### Research objective

Turn the broad idea of “image + KG embeddings for heritage” into falsifiable questions, documented assumptions, and reproducible infrastructure.

### Tasks

- [x] Define primary research question.
- [x] Define H1–H5 hypotheses.
- [x] Define corpus roles: Iconclass AI, Emblematica, Rijksmuseum, WJoconde.
- [x] Define model families and baseline matrix.
- [x] Define leakage controls.
- [x] Define notebook/source-code separation.
- [x] Add MIT licence.
- [x] Add citation metadata.
- [x] Add CI skeleton.
- [x] Add data provenance and reproducibility policy.
- [x] Freeze first experimental protocol as `docs/protocol-v0.1.md` before model fitting; revise only by versioning.
- [x] Establish a verified literature map and publication-grade review protocol before originality claims.

### Deliverables

- repository skeleton;
- `RESEARCH_PLAN.md`;
- `ROADMAP.md`;
- project overview notebook;
- sample tested utility functions.

### Exit criterion

A third party should be able to understand **what would falsify the central hypothesis** without running a model.

---

## Phase 1 — Corpus audit and Iconclass benchmark construction

**Target release:** `v0.2.0`

### Research objective

Establish a controlled benchmark with expert-assigned semantic structure before experimenting on historically richer but messier data.

### Tasks

- [x] Document source, rights caveats, verified archive checksum, and explicit acquisition procedure.
- [x] Implement canonical image/concept manifest builder (full-corpus materialisation pending download).
- [x] Parse the open Iconclass hierarchy and derive parent/ancestor paths from source edges.
- [x] Implement annotation, co-occurrence, missingness, and hierarchy-depth audit code.
- [ ] Run and freeze the full-corpus frequency/depth/co-occurrence/imbalance report after archive acquisition.
- [x] Implement exact SHA-256 duplicate grouping/checks when image bytes are available.
- [ ] Detect and cluster near-duplicate images at full-corpus scale.
- [x] Implement deterministic diagnostic train/validation/test partitioning.
- [x] Implement group-aware splitting and leakage checks; trustworthy source groups still need recovery.
- [x] Implement canonical JSONL serialisation and SHA-256 manifest digests.
- [x] Define and test transparent hierarchy-distance relevance baseline in `docs/protocol-v0.1.md`.
- [x] Add non-image official annotation excerpt and documented hierarchy fixture for CI.

### Planned notebook

`01_iconclass_graph.ipynb`

### Deliverables

- versioned corpus manifest;
- Iconclass graph export;
- split definitions;
- corpus audit figures;
- benchmark data card.

### Exit criterion

The benchmark must support evaluation that cannot be trivially solved by exact/near duplicate leakage.

### Main risks

- long-tail concept distribution;
- ambiguous or multi-label annotation;
- inconsistent image availability;
- circular use of Iconclass as both training input and evaluation target.

### Mitigation

Maintain evaluation regimes in which held-out examples, branches, books, or collections are not exposed to the fusion model during fitting.

---

## Phase 2 — Visual representation baselines

**Target release:** `v0.3.0`

### Research objective

Measure what visual models already capture before attributing any gain to graph structure.

### Candidate models

- DINOv2;
- CLIP/OpenCLIP;
- SigLIP.

### Tasks

- [x] Implement model-native deterministic preprocessing through versioned Hugging Face processors.
- [x] Implement extraction and cache format; full-corpus extraction remains data/model dependent.
- [x] Record model identifier, resolved revision, processor config, library versions, pooling, manifest checksum, and vector dimension.
- [x] Build exact cosine search baseline with self-match exclusion.
- [x] Add optional exact FAISS `IndexFlatIP` as a scaling backend while retaining NumPy exact cosine as the reference implementation.
- [x] Implement standard retrieval evaluation (hierarchical nDCG@10, Recall@1/5/10, MRR, mAP); full-model result tables remain pending.
- [x] Implement fixed depth/frequency stratification; populate it with real encoder runs once the full corpus is reconstructed.
- [ ] Collect representative successes and failures.

### Planned notebooks

- `02_visual_embeddings.ipynb`
- `04_visual_retrieval_baseline.ipynb`

### Exit criterion

At least two substantially different visual representation families have reproducible results under the same split and metric definitions.

---

## Phase 3 — Knowledge-graph embedding baselines

**Target release:** `v0.4.0`

### Research objective

Determine whether graph proximity captures an interpretable semantic structure that is complementary to visual proximity.

### Candidate models

- Node2Vec as a deliberately simple control;
- RDF2Vec;
- ComplEx;
- RotatE.

### Tasks

- [x] Define graph projections and target-leakage policy explicitly (`G0` taxonomy oracle, `G1` context-only, `G2` masked-label).
- [x] Separate taxonomy-only oracle/control experiments from richer evidence-bearing relational projections.
- [x] Implement fixed-seed relation-aware KGE code paths: predicate-aware RDF2Vec-style control plus optional PyKEEN ComplEx/RotatE; full-corpus training remains pending external data/model runs.
- [ ] Evaluate full-corpus graph neighbourhood quality using hierarchical relevance; the executable fixture pipeline is implemented.
- [x] Implement degree-vs-neighbour-hubness diagnostics; populate full-corpus correlations once real G1/G2 embeddings are available.
- [x] Implement graph-neighbour vs visual-neighbour overlap metric; populate full-corpus results after real embeddings are available.

### Planned notebooks

- `03_kg_embeddings.ipynb`
- `05_graph_retrieval_baseline.ipynb`

### Exit criterion

Graph retrieval must provide measurable information not reducible to the visual nearest-neighbour ranking.

### Stop condition

If KG neighbours provide no complementary signal under robust evaluation, do **not** proceed directly to a complex joint model. Revisit graph construction and research assumptions first.

---

## Phase 4 — Hard-pair benchmark

**Target release:** `v0.5.0`

### Research objective

Create an evaluation subset where visual resemblance and iconographic relatedness deliberately diverge.

### Four pair classes

1. visually close + iconographically close;
2. visually close + iconographically distant (**hard negative**);
3. visually distant + iconographically close (**hard positive**);
4. visually distant + iconographically distant.

### Tasks

- [x] Freeze validation-only visual threshold calibration (95th percentile close, median distant) in protocol v0.3.
- [x] Freeze semantic thresholds in protocol v0.3 (`>=0.5` close, `<=0.2` distant).
- [x] Implement deterministic balanced mining from visual top-k, semantic BFS, and random distant candidates; full-corpus artifact pending.
- [ ] Manually inspect an evaluation subset.
- [x] Persist pair class, visual/semantic scores, labels, calibration metadata, and checksum in canonical JSONL/JSON artifacts.
- [ ] Freeze the **real full-corpus** hard-pair test artifact before reporting fusion results; the procedure is frozen and fixture-tested.

### Planned notebook

`07_hard_pairs_evaluation.ipynb`

### Exit criterion

The benchmark contains enough high-confidence disagreement cases to distinguish genuine semantic improvement from generic retrieval gains.

---

## Phase 5 — Transparent multimodal fusion

**Target release:** `v0.6.0`

### Research objective

Test whether graph signal improves retrieval using methods simple enough to interpret.

### Baseline A — weighted late fusion

`score = alpha * visual_similarity + (1 - alpha) * graph_similarity`

Evaluate a predeclared alpha grid and tune only on validation data.

### Baseline B — graph reranking

1. retrieve top-N candidates visually;
2. rerank with graph similarity;
3. measure gains/losses relative to visual retrieval.

### Tasks

- [x] Implement validation-pair min-max score calibration with no test-derived bounds.
- [x] Implement preregistered alpha sweep `[0, .25, .5, .75, 1]` with validation nDCG@10 selection and conservative tie-break.
- [x] Implement fixed visual candidate-pool reranking with validation-calibrated modality scores.
- [ ] Compare overall and hard-pair performance.
- [ ] Quantify how often graph information changes a top-K result.
- [ ] Produce explanations for changed rankings.

### Planned notebook

`06_multimodal_fusion.ipynb`

### Exit criterion

At least one transparent fusion strategy improves a preregistered semantic metric without unacceptable degradation of visual relevance.

---

## Phase 6 — Learned joint alignment

**Target release:** `v0.7.0`

### Research objective

Test whether a learned shared space offers value beyond transparent fusion.

### Initial architecture

Frozen encoders + small projection heads + contrastive objective.

```text
visual embedding -> projection --+
                                 +--> shared space
KG embedding     -> projection --+
```

### Tasks

- [ ] Implement projection heads.
- [ ] Define positive/negative sampling strategy.
- [ ] Train with fixed data partitions.
- [ ] Compare against late fusion and reranking.
- [ ] Run seed sensitivity analysis.
- [ ] Evaluate representation collapse/hubness.
- [ ] Conduct modality ablation.

### Model matrix

- V
- G
- T
- V+T
- V+G
- G+T
- V+G+T

### Exit criterion

The learned model must outperform the strongest simple fusion baseline on at least one central preregistered metric and retain interpretable behaviour on hard pairs.

### Stop condition

If a complex model only matches weighted fusion, prefer the transparent method in the main paper.

---

## Phase 7 — Emblematica historical case study

**Target release:** `v0.8.0`

### Research objective

Move from controlled taxonomy evaluation to historically meaningful, relational material.

### Graph scope

Model only relations needed by the research question, for example:

- emblem -> pictura;
- emblem -> motto;
- emblem -> subscriptio;
- emblem -> Iconclass concept;
- emblem -> book;
- book -> author;
- book -> printer;
- book -> place;
- book -> date.

### Tasks

- [ ] Audit UIUC/HAB data access and licences.
- [ ] Build IIIF-first manifest where possible.
- [ ] Construct minimal RDF graph.
- [ ] Map concepts to existing vocabularies when stable mappings exist.
- [ ] Compare visual, graph, and fused neighbours for emblem queries.
- [ ] Identify interpretable cross-book or cross-edition relations.
- [ ] Document cases where retrieval suggests a hypothesis rather than established influence.

### Planned notebook

`08_emblematica_case_study.ipynb`

### Exit criterion

The method produces interpretable retrieval differences that are historically meaningful enough for expert assessment, without presenting vector similarity as proof of influence.

---

## Phase 8 — Cross-collection transfer

**Target release:** `v0.9.0`

### Research objective

Test whether observed gains survive outside the collection/cataloguing environment used to develop the method.

### Preferred target

Rijksmuseum, using its structured metadata, classifications, and image/IIIF infrastructure where licensing permits.

### Tasks

- [ ] Define target subset and mapping coverage.
- [ ] Freeze model before target evaluation.
- [ ] Test zero-shot or minimally calibrated transfer.
- [ ] Compare image-only and multimodal ranking.
- [ ] Measure performance by object type, date, and concept coverage.
- [ ] Conduct failure analysis for metadata and vocabulary mismatch.

### Planned notebook

`09_cross_collection_transfer.ipynb`

### Exit criterion

At least one graph-aware method provides evidence of transferable value, or the project clearly characterises why the gain is collection-specific.

---

## Phase 9 — Secondary benchmark on WJoconde

**Target release:** `v0.10.0`

### Research objective

Position the retrieval work against multimodal cultural-heritage KG literature and test portability to a different graph schema.

### Tasks

- [ ] Reproduce a relevant published WJoconde baseline where feasible.
- [ ] Map the repository's representation/fusion interface onto WJoconde.
- [ ] Separate retrieval claims from KG-completion claims.
- [ ] Report where methods transfer and where task definitions diverge.

### Exit criterion

The project can state precisely how it differs from and complements multimodal KG completion/enrichment work.

---

## Phase 10 — Human/expert evaluation and interpretability

**Target release:** `v0.11.0`

### Research objective

Assess whether metric improvements correspond to useful cultural-heritage discovery.

### Proposed protocol

For a stratified sample of queries, evaluators judge retrieved pairs on separate dimensions:

- formal/visual similarity;
- iconographic relatedness;
- contextual/historical relevance;
- novelty/usefulness for research;
- confidence.

Do not collapse these dimensions into one vague “relevance” score.

### Tasks

- [ ] Define annotation protocol.
- [ ] Pilot inter-rater agreement.
- [ ] Blind method labels during evaluation.
- [ ] Compare human rankings with automatic metrics.
- [ ] Analyse disagreements rather than hiding them.

### Exit criterion

The project can distinguish “metric improvement” from “useful scholarly retrieval.”

---

## Phase 11 — Research release and demonstrator

**Target release:** `v1.0.0`

### Outputs

- [ ] frozen benchmark manifests and splits;
- [ ] reproducible experiment configurations;
- [ ] executed notebooks with clean outputs;
- [ ] citable software release;
- [ ] DOI for release/data artifacts where possible;
- [ ] research paper preprint;
- [ ] model/results cards;
- [ ] interactive demonstration.

### Demonstrator design

A query image should expose three rankings side by side:

- visual;
- graph;
- multimodal.

Each result should show the evidence responsible for retrieval, such as shared concepts or a graph path. The demonstrator is an explanatory layer over the experiments, not the evidence itself.

---

# Evaluation plan

## Retrieval metrics

Primary candidates:

- Recall@1 / @5 / @10;
- MRR;
- mAP;
- nDCG@10 with graded hierarchical relevance.

## Statistical analysis

Where appropriate:

- bootstrap confidence intervals over queries;
- paired comparisons between methods;
- effect sizes, not only p-values;
- results across multiple random seeds for learned models;
- subgroup analysis by concept frequency/depth and corpus source.

## Qualitative analysis

Every major release should include:

- representative true improvements;
- representative regressions;
- hard positives;
- hard negatives;
- unexplained retrievals;
- graph-driven changes that are historically misleading.

---

# Compute strategy

## Stage A: laptop/small GPU feasible

- corpus analysis;
- precomputed/small-batch visual embeddings;
- graph construction;
- Node2Vec/RDF2Vec;
- exact retrieval on samples;
- metric development.

## Stage B: single GPU

- full visual embedding extraction;
- KGE experiments;
- learned projection heads;
- ablations.

## Stage C: only if justified

- larger multimodal encoders;
- graph neural networks;
- domain-specific fine-tuning.

No distributed infrastructure is a milestone. It is merely a cost.

---

# Publication decision gates

A paper-oriented release should proceed only if at least one of these claims survives rigorous evaluation:

1. KG-aware fusion improves iconographic retrieval beyond strong vision-language baselines.
2. KG-aware fusion specifically improves hard positives / hard negatives even when aggregate metrics change little.
3. Graph information enables useful cross-collection transfer or interpretability unavailable from image embeddings alone.
4. Negative result: current KG representations do **not** improve retrieval, with a careful explanation of why. A clean negative result is still more useful than decorative multimodality.

