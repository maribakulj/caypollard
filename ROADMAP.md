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
- [x] Run and freeze the full-corpus frequency/depth/co-occurrence/imbalance report after archive acquisition.
- [x] Implement exact SHA-256 duplicate grouping/checks when image bytes are available.
- [x] Detect and cluster near-duplicate images at full-corpus scale.
- [x] Implement deterministic diagnostic train/validation/test partitioning.
- [x] Implement group-aware splitting and leakage checks; book-level source groups recovered from filenames for 16.7% of the corpus.
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

**Met.** The frozen v0.1 benchmark covers all 87 744 annotated images with SHA-256 and
two-stage perceptual hashes, groups them by the union of book identifier, exact bytes, and
confirmed near duplicate, and reports zero group and zero checksum leakage across the
70 304 / 8 539 / 8 901 partition. Near-duplicate detection resolves 130 multi-image groups
covering 269 images, the largest holding 4.

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
- [x] Collect representative successes and failures (`scripts/analyse_rank_changes.py` persists
      the largest gains and the largest regressions with the evidence for each).

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
- [x] Evaluate graph neighbourhood quality using hierarchical relevance on the real `G1` sub-corpus.
- [x] Implement degree-vs-neighbour-hubness diagnostics; real `G1` embeddings now available for population.
- [x] Implement graph-neighbour vs visual-neighbour overlap metric; populated on real embeddings (G1 vs visual 0.15-0.18, visual vs visual 0.32-0.37).

### Planned notebooks

- `03_kg_embeddings.ipynb`
- `05_graph_retrieval_baseline.ipynb`

### Exit criterion

Graph retrieval must provide measurable information not reducible to the visual nearest-neighbour ranking.

**Met on the `G1` sub-corpus (1 714 test images).** `G1` reaches nDCG@10 0.727 against a
0.329 random control through the identical ranking and metric path (+0.398, d = 1.40), while
sharing only 0.15-0.18 of its top-10 neighbours with a visual encoder - against 0.32-0.37
between two visual encoders. The signal is therefore both real and not a restatement of
visual proximity. `G1` alone remains slightly below every visual baseline on nDCG
(-0.015 to -0.041), which is a level, not a redundancy, result.

### Stop condition

If KG neighbours provide no complementary signal under robust evaluation, do **not** proceed directly to a complex joint model. Revisit graph construction and research assumptions first.

---

## Phase 4 — Hard-pair benchmark

**Target release:** `v0.5.0`

### Research objective

Create an evaluation subset where visual resemblance and iconographic relatedness deliberately diverge.

### Constraint discovered in execution: H3 is not testable under Iconclass

Iconclass is predominantly denotative. Two lions — one for fortitude, one for defeat — both carry
`25F23`, so the vocabulary scores them semantically *close* and the hard-negative rule
(similarity ≤ 0.2) can never select them. Division 5, *Abstract Ideas and Concepts*, does supply a
connotative layer (17.1% of Emblematica assignments), which is why hard **positives** work: they
meet on shared concepts such as `56F2` Love or `57B1` Praise, 193 of 250 sharing an exact
notation. Hard **negatives** instead come out as pairs sharing a page layout — 1 of 140 shares any
notation. H3 as stated needs an annotation separating a motif from its allegorical reading, which
this ground truth does not provide.

### Four pair classes

1. visually close + iconographically close;
2. visually close + iconographically distant (**hard negative**);
3. visually distant + iconographically close (**hard positive**);
4. visually distant + iconographically distant.

### Tasks

- [x] Freeze validation-only visual threshold calibration (95th percentile close, median distant) in protocol v0.3.
- [x] Freeze semantic thresholds in protocol v0.3 (`>=0.5` close, `<=0.2` distant).
- [x] Implement deterministic balanced mining from visual top-k, semantic BFS, and random distant candidates; full-corpus artifact pending.
- [x] Manually inspect an evaluation subset.
- [x] Persist pair class, visual/semantic scores, labels, calibration metadata, and checksum in canonical JSONL/JSON artifacts.
- [x] Freeze the real hard-pair test artifact on the `G1` sub-corpus before reporting fusion results.

### Planned notebook

`07_hard_pairs_evaluation.ipynb`

### Exit criterion

The benchmark contains enough high-confidence disagreement cases to distinguish genuine semantic improvement from generic retrieval gains.

**Met, and it did that job twice — first against the fusion result, then against the
benchmark's own relevance definition.**

The first mining exposed a defect rather than a finding: bracketed Iconclass text keys were
stripped, so notation `86` (proverbs, emblems, mottoes) collapsed 5 972 distinct mottoes onto
one node and every emblem scored as perfectly relevant to every other. Only 1 of 250 mined
hard positives shared an exact notation with its partner. Those pairs were not hard positives,
and the analysis built on them is void. See [`protocol-v0.5.md`](docs/protocol-v0.5.md).

Re-mined under the corrected `keep` policy, 186 of 250 hard positives share an exact notation —
genuine iconographic matches that look nothing alike. The conclusion below is drawn from those,
and is unchanged from the void analysis, which is itself worth recording: the defect had
inflated the pairs without reversing the verdict. The separation is structural:

| pair class | same book (`keep`, DINOv2 mining) |
| --- | ---: |
| easy positive | 243 / 250 |
| hard negative | 174 / 250 |
| **hard positive** | **14 / 250** |
| easy negative | 0 / 250 |

A hard positive is visually distant but iconographically close, and such a pair is almost
never two plates of one volume. `G1` encodes nothing but volume membership and plate order,
so it holds **no edge at all** between the images H2 is about. Measured as AUC of the graph
score, restricted to visually distant pairs where only iconography separates them:

- hard positives vs easy negatives: **0.498** — chance;
- hard positives vs hard negatives: **0.241** — inverted, because hard negatives *are* mostly
  same-volume pairs and the graph rewards them.

By contrast the graph scores easy positives at 0.981 and easy negatives at 0.416 — it is a
competent detector of exactly one thing, co-membership of a volume.

**Bibliographic enrichment does not change this.** 505 Munich volumes were ingested from IIIF
manifests, yielding 209 GND creators, 163 printers, 66 places and 136 works. They reach
**0 of 250** hard positives, under both relevance policies and both mining encoders, because
hard positives are emblem-book pairs across St Andrews, UIUC, Wolfenbüttel, Mnemosyne and
Utrecht while the enriched volumes are Munich incunabula.

**Consequence for the phase-5 result.** The +0.025 to +0.044 nDCG@10 gain is real and
reproducible, but it is a *generic* retrieval gain: plates of one book usually share
iconography, and the graph recovers that. It is not evidence for H2 or H3, which concern
precisely the pairs this graph cannot reach. Publication gate 2 is therefore not supported by
the current projection, and any hard-case claim needs cross-volume relations — shared creator,
printer, place, date — which requires the UIUC/HAB catalogue join described in
`GRAPH_PROJECTIONS.md`.

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
- [x] Compare overall performance against visual baselines; hard-pair comparison pending the frozen artifact.
- [x] Quantify how often graph information changes a top-K result: 51-67% of queries, 1.2-3.4
      items swapped, improvements outnumbering regressions 1.6 to 2.4 to one.
- [x] Produce explanations for changed rankings: each promoted item is classified by what it
      shares with the query. On Emblematica 55-63% share an Iconclass notation as well as a
      bibliographic attribute; on Iconclass only 21-26% do and about half share nothing but
      the volume. Regressions have one mechanism, a volume plateau flooding the top-10.

### Planned notebook

`06_multimodal_fusion.ipynb`

### Exit criterion

At least one transparent fusion strategy improves a preregistered semantic metric without unacceptable degradation of visual relevance.

**Met for all three encoders.** Preregistered late fusion at the validation-selected
`alpha = 0.25` improves test nDCG@10 over visual-only by +0.044 (DINOv2), +0.025 (CLIP), and
+0.030 (SigLIP), all with p <= 0.0002 and d = 0.26-0.33 - an order of magnitude above the
d ~ 0.05 that separates the visual encoders from each other. MRR improves for all three.
MAP improves for DINOv2 (+0.019) and is inconclusive for CLIP and SigLIP, so the gain is
concentrated in graded hierarchical relevance rather than exact-label precision.

**Scope of the claim.** This holds on the sub-corpus where `G1` exists - 1 714 of 8 901 test
images. `G1` carries book membership and plate order only, so the most conservative reading
is that plates of one volume share iconography, and that this contextual fact is not
recoverable from pixels. That is a genuine non-target signal, not target leakage, but it is
narrower than the full contextual hypothesis.

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

- [x] Implement small linear/MLP projection heads over frozen embeddings.
- [x] Define positive/negative sampling strategy (same-object positives, in-batch negatives).
- [x] Implement fixed train/validation partition training with validation-loss checkpoint selection; full-corpus runs remain pending.
- [x] Compare real-corpus results against late fusion: the learned space loses nDCG@10 by
      0.0051 (p = 0.0010, d = -0.081) and wins MAP by 0.0231 (p = 0.0001, d = 0.259).
- [x] Run the preregistered >=3-seed sensitivity analysis on real embeddings: spread across
      three seeds is 0.0006 of nDCG@10, so the comparison is not seed noise.
- [x] Implement representation-collapse diagnostics (off-diagonal cosine + effective rank); combine with existing hubness diagnostics in real runs.
- [x] Conduct modality ablations on real embeddings: T alone 0.6281, G 0.6829, V 0.7178,
      G+T 0.7026, V+T 0.7319, V+G 0.7450, V+G+T 0.7450 — the validation sweep gives the text
      arm zero weight once the graph is present, so text is redundant with it, not additive.
- [x] Conduct capacity ablations (linear vs MLP heads) on real embeddings: the MLP is worse on
      both metrics, and effective rank falls from 20.5 of 128 dimensions to 15.3.

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

**Reached, and applied.** The learned model beats late fusion on MAP (+0.0231, d = 0.259) and
loses the primary endpoint (-0.0051, d = -0.081), while reproducing rather than escaping the
`G1` unreachability artifact on hard pairs. Validation loss rises from the first epoch in all six
runs because the objective amounts to predicting an item's volume from its pixels, and the
group-aware split places every validation volume outside training. The exit criterion is
therefore half met, the stop condition applies, and the transparent method stands as the
project's recommended system. See [`docs/RESULTS.md`](docs/RESULTS.md).

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

### Status: reframed from a case study into the corpus that makes H2 testable

Phase 7 was planned as a historical case study to follow validation on Iconclass AI. The
hard-pair analysis moved it forward: hard positives are emblem-book pairs, the Iconclass sample
is drawn from a subscription database whose identifiers do not resolve publicly, and
bibliographic enrichment there reached 0 of 250 hard positives. Emblematica Online carries
images, Iconclass notations, emblem texts and resolvable book identifiers in one open corpus,
so it is ingested as a second benchmark rather than as an illustration. See
[`EMBLEMATICA_DATA_CARD.md`](docs/EMBLEMATICA_DATA_CARD.md).

### Tasks

- [x] Audit UIUC/HAB data access and licences.
- [x] Build a manifest from the open API (IIIF is not exposed; the SPINE record is).
- [x] Construct minimal RDF-style graph: `part_of`, `adjacent_to`, `created_by`,
      `published_at`, `published_in`, `instance_of`.
- [x] Map concepts to existing vocabularies when stable mappings exist. The Iconclass SKOS
      export carries only notation/broader/narrower, so the alignment comes from Wikidata's
      P1256: 4 125 distinct notations, covering 37.5% of Iconclass assignments and 68.3% of
      its images, and 28.2% of Emblematica assignments and 83.0% of its emblems. Enough to
      name a concept in readable words, not enough to serve as ground truth.
- [x] Establish whether the emblem's interpretive verse can serve as ground truth for H3: it
      cannot. 455 of 31 041 records carry a transcribed subscriptio and 454 of those carry no
      Iconclass annotation, so verse and ground truth coexist on one record. Expert judgement
      (phase 10) is the only remaining route.
- [x] Compare visual, graph, and fused neighbours for emblem queries. Mean top-10 overlap:
      two visual encoders 0.228, graph against visual 0.081-0.100, text against visual 0.042,
      text against graph 0.042, and the fused ranking against its own visual arm **0.811** --
      the fusion is four fifths the visual system, which matches the 1.89 items of 10 that
      the rank-change analysis measures independently.
- [x] Establish that cross-volume paths exist at all: 272 of 368 emblem books (74%) share a
      creator with another book and 254 (69%) share a place, against 0% reachable in the
      Iconclass sample.
- [x] Identify interpretable cross-book or cross-edition relations. Of 28 040 top-10 slots on
      the test corpus only 346 point outside the query's volume, and each is classified by
      the evidence joining the pair.
- [x] Document cases where retrieval suggests a hypothesis rather than established influence.
      Of those 346, 87 share a creator and 124 a place or decade -- recovery of a catalogue
      fact, not discovery -- 118 share nothing at all, and **17** join two volumes with no
      shared creator, place or decade through an Iconclass notation. Only those 17 can
      suggest a hypothesis, and they are listed with their mottoes in
      `results/emblematica/cross-book-retrievals.json`.

### Planned notebook

`08_emblematica_case_study.ipynb`

### Exit criterion

The method produces interpretable retrieval differences that are historically meaningful enough for expert assessment, without presenting vector similarity as proof of influence.

### Result: the fusion gain replicates; H2 does not, and the reason is now precise

On 2 804 Emblematica test queries under protocol v0.5, preregistered late fusion improves
nDCG@10 over visual-only for all three encoders and both graph variants — +0.031 (DINOv2),
+0.019 (CLIP), +0.020 (SigLIP), every one at p = 0.0002 with d = 0.24-0.29. That is an
independent replication: a different corpus, a different ingestion pipeline, effect sizes
matching the 0.24-0.33 measured on Iconclass AI.

H2 fails again, and not for the reasons previously suspected. Both have been eliminated:

- **not the corpus** — this one carries creator, place, date and work for 99.6% of volumes;
- **not the graph construction** — the `pure` projection separates same-author cross-volume
  pairs at AUC 0.963, against 0.457 for the volume-membership graph that failed before.

The measurement that settles it is what hard positives actually share. Of 250 mined with
DINOv2, **243 share no bibliographic attribute at all** and 2 share an author; of 250 mined
with CLIP, **0 share an author** and the 66 that share anything share only a *decade*. That
weak, broad attribute is what produces CLIP's apparent H2 AUC of 0.759 against DINOv2's 0.521 —
chronological coincidence, not iconographic knowledge. Hard *negatives*, meanwhile, do share
attributes (36 and 20 shared authors), which pushes the score the wrong way.

Mottoes do not bridge the gap either: at most 11 of 250 hard-positive pairs share a single
content word, and 2 share two.

**This reasoning was confounded and has been corrected.** The pairs it rests on were mined
without a format control. Holding libraries scan differently — full pages with a ruler and colour
chart in one, cropped picturae in another — so "visually distant" collapsed into "different
library": 247 of 250 CLIP-mined hard positives crossed collections against 0 of 250 hard
negatives, and 243 against 0 on Iconclass AI. Pairs drawn from different libraries are
necessarily from different books and authors, so their bibliographic unrelatedness followed from
the sampling, not from motif circulation. That interpretation is withdrawn.

Re-mined within a collection (`--stratum-key collection`) and compared against a control matched
on visual similarity, the graph reaches AUC 0.449 [0.387, 0.510], 0.633 [0.567, 0.697],
0.691 [0.646, 0.736] and 0.582 [0.532, 0.633] across the four corpus-encoder conditions — weakly
above chance in three, inconclusive and below it in the fourth. H3 stays wrong-signed
(0.10-0.43). Publication gate 2 remains unsupported, on a weaker and better-founded basis. See
[`docs/RESULTS.md`](docs/RESULTS.md).

---

## Phase 8 — Cross-collection transfer

**Target release:** `v0.9.0`

### Research objective

Test whether observed gains survive outside the collection/cataloguing environment used to develop the method.

### Preferred target

Rijksmuseum, using its structured metadata, classifications, and image/IIIF infrastructure where licensing permits.

### Tasks

- [x] Define target subset and mapping coverage: 1 864 Rijksmuseum works reached without an API
      key, through Wikidata `depicts` joined to Iconclass by P1256. See
      [`RIJKSMUSEUM_DATA_CARD.md`](docs/RIJKSMUSEUM_DATA_CARD.md).
- [x] Freeze model before target evaluation: encoders are pretrained and never fitted, and the
      zero-shot arm carries the source corpus's `alpha` unchanged.
- [x] Test zero-shot and minimally calibrated transfer, both.
- [x] Compare image-only and multimodal ranking across three encoders and two graph variants.
- [x] Measure performance by object type, date, and concept coverage.
- [x] Conduct failure analysis for metadata and vocabulary mismatch.

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

- [x] Reproduce a relevant published WJoconde baseline where feasible — it is **not** feasible.
      The paper's abstract page links no dataset, so the baseline cannot be reproduced from
      materials this pipeline can reach. Recorded rather than left open.
- [x] Map the repository's representation/fusion interface onto WJoconde: the interface ports,
      and the Rijksmuseum ingestion proves it in a day. The **relevance definition** does not.
- [x] Separate retrieval claims from KG-completion claims. See
      [`POSITIONING.md`](docs/POSITIONING.md).
- [x] Report where methods transfer and where task definitions diverge, with measurement rather
      than assertion: `scripts/audit_joconde.py` samples 22 957 records across seven byte ranges
      of the 1.2 GB national export. `Sujet_Represente` is a free-text faceted phrase, not a
      classification — 58.3% of its 12 801 terms occur exactly once, and no column carries an
      image URL. Graded hierarchical relevance, the project's primary endpoint, is undefined on
      it.

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

- [x] Define annotation protocol: five dimensions, a 1-5 ordinal scale with an explicit
      "cannot judge", tercile-stratified sampling, and the analysis fixed in advance. See
      [`EXPERT_EVALUATION_PROTOCOL.md`](docs/EXPERT_EVALUATION_PROTOCOL.md).
- [ ] Pilot inter-rater agreement — **needs experts, which is a decision outside this repository.**
      Everything else is built and tested so that the ask is an afternoon rather than a project.
- [x] Blind method labels during evaluation: `scripts/build_expert_evaluation.py` pools the
      methods' top-5 per query, shuffles with a fixed seed, and writes the method/rank mapping to
      a separate key the analysis reads and the rater never sees.
- [x] Compare human rankings with automatic metrics: Kendall's tau-b between each dimension and
      the graded relevance assigned to the same pair, in `scripts/analyse_expert_evaluation.py`.
- [x] Analyse disagreements rather than hiding them: pairs differing by two points or more are
      returned whole and worst-first, not summarised as a variance.

### Exit criterion

The project can distinguish “metric improvement” from “useful scholarly retrieval.”

**Instrumented, not yet answered.** The protocol fixes in advance what each outcome would mean,
including the one the project expects: given that the fused ranking shares 0.811 of its top-10
with its own visual arm and fails every hard-case test, a null human result is the likely finding
and is to be reported as plainly as a positive one. The measurement chain is built and exercised
end to end on synthetic judgements; only the raters are missing.

---

## Phase 11 — Research release and demonstrator

**Target release:** `v1.0.0`

### Outputs

- [x] frozen benchmark manifests and splits;
- [x] reproducible experiment configurations;
- [x] executed notebooks with clean outputs;
- [x] citable software release;
- [ ] DOI for release/data artifacts where possible;
- [ ] research paper preprint;
- [x] model/results cards;
- [x] interactive demonstration.

### Built

The demonstrator shows eight test emblems chosen for maximum disagreement between the three
rankings, each result carrying the evidence that produced it — shared Iconclass notations,
shared volume, shared author, shared place — rather than a bare similarity score. Rankings are
real, computed on the frozen 2 804-emblem test pool with DINOv2 and the `pure` graph, with
calibration and `alpha` taken from validation only.

`docs/RESULTS.md` is the results card; `docs/ICONCLASS_DATA_CARD.md` and
`docs/EMBLEMATICA_DATA_CARD.md` are the data cards. A DOI and a preprint remain outstanding and
require actions outside this repository.

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

