# Changelog

## Unreleased

### Added

- **a viewer for the pool.** `viewer/` is a local page, served by `scripts/serve_viewer.py`,
  that shows the 21 128 pictures of the museum and emblem benchmarks as thumbnails, filters
  them by corpus, collection, kind, genre or century, and for any picture lists the hundred
  nearest neighbours under each representation table, saying what each neighbour shares with
  the query — subject, kind, collection, century, genre, hand, book — so a regime can be read
  off a list rather than a table. `scripts/build_viewer.py` writes everything it reads;
  `scripts/fetch_wikidata_labels.py` turns the manifests' Wikidata ids into words, once.
  Notations carried by more than 3.2% of the pool are treated as categories rather than
  subjects, the cut the regime scripts apply;
- **the demo engine, "sous le capot".** `caypollard.demo` recomputes every representation
  for one picture — dropped in, or taken from the pool — through the experiments' own
  functions, shows each intermediate step with its parameters and its explanation, lets the
  parameters be edited, and searches the pool by channel or by weighted mix with exact
  cosine. `scripts/fit_demo_models.py` recovers the four k-means models the tables were
  built on (their centres had never been saved) and verifies each against its frozen table:
  formes muettes and formes v4 at 1.2e-7, répétition at 3e-8, signes composites at 0 — the
  last only once the outline block is scaled after stacking, in float32, as the script does.
  On a picture of the pool every channel reproduces its frozen row at cosine 1.0000. The
  named nodes are produced by Claude (Sonnet) through Claude Code in non-interactive mode
  rather than by the Mistral namer of the frozen tables, and the reproduction cosine on that
  step (0.45 on the first picture tried) records how differently two namers see. The two
  channels the record set aside are shown as well, so they can be seen before being
  dropped: the relations between signs (the 64-sign vocabulary and the 9 518 keys of
  `relations-wide64` recovered and verified at 0) and the pose (Keypoint R-CNN, skeleton
  drawn on the picture, angles per segment; its search table is the partial one of 1 903
  pictures, the only one that exists).

- **the strokes.** `caypollard.sketch` reduces a picture to its contours at a coarse scale
  (smoothed gradient, non-maximum suppression, hysteresis: two colours, no fill), thins them,
  walks them into strokes that continue through crossings along the straightest branch,
  simplifies each to a polyline, and rations them: the sketch keeps the forty longest, the
  pictogram the twelve. It starts from edges where every other channel starts from tone.
  `scripts/build_sketch_channel.py` draws every picture of the pool that way, describes its
  sketch strokes (size, direction, bending, place: 16 numbers), fits a vocabulary of 128
  stroke signs balanced across corpora with its centres saved, and writes two tables: the
  stroke-sign histogram weighted by length, and the contour drawings through DINOv2. Both
  are search channels in the demo, whose "traits" step shows the four levels and the
  description of each stroke, with every parameter editable.

### Found

- `docs/VISUAL_RECORD.md` says regions are cut by a watershed; `scripts/segment_shapes.py`
  thresholds at two polarities and labels connected components. The "formes" block of
  `record-v3` is the composite signs of `v3-groups`, not a shape-sign table, and `mix-tout`
  uses the balanced `v4-256` vocabulary rather than `v2-256`. The "mixte" tables have no
  producing script: they are row concatenations of the per-corpus tables.

## 0.7.0 — 2026-09-17

Three phases close and one central claim is overturned. The release is dominated by controls
that were missing rather than by new capability.

### Added

- **a third corpus.** 1 864 Rijksmuseum works ingested from Wikidata and Wikimedia Commons with
  no API key, subject terms joined to Iconclass through Wikidata's P1256, zero group and checksum
  leakage. See [`RIJKSMUSEUM_DATA_CARD.md`](docs/RIJKSMUSEUM_DATA_CARD.md);
- **the text modality** the model matrix has declared since phase 0 with nothing behind it:
  multilingual sentence encoders with mean pooling over the attention mask, filling T, V+T, G+T
  and V+G+T for the first time;
- **four fusion rules beside the weighted sum** — rank-linear, reciprocal rank, elementwise
  maximum and geometric mean — each run through the identical calibration and evaluation path;
- **visually matched controls** (`benchmarks.matched_controls`) with a balanced caliper draw,
  repeated across seeds;
- **the expert evaluation, built and tested end to end**: Krippendorff's alpha with an ordinal
  difference function, Kendall's tau-b, item-level disagreement analysis, a blinded package
  builder and its sealed key. See
  [`EXPERT_EVALUATION_PROTOCOL.md`](docs/EXPERT_EVALUATION_PROTOCOL.md);
- [`POSITIONING.md`](docs/POSITIONING.md), stating how retrieval differs from KG completion;
- the Wikidata concept alignment: 4 125 Iconclass notations, reaching 37.5% of Iconclass
  assignments and 28.2% of Emblematica's.

### Changed

- **H2 is not supported, and the reason is structural rather than empirical.** Two controls
  overturned the previous reading. Volume size alone reaches AUC 0.578 on hard positives, a real
  confound in the aggregate figures. And `G1` is a disjoint union of 1 246 components, one per
  volume, with no path between items of different volumes — so the 0.649 it scores on such pairs
  cannot be information. Emblematica's `pure` projection, where volumes genuinely are connected,
  gives 0.505. Where the graph can carry cross-volume information it carries none; where it
  appears to, it structurally cannot;
- **the fusion gain does not transfer.** Of six encoder-by-graph conditions on the Rijksmuseum,
  one gains, one loses significantly, three select `alpha = 1.0` — no fusion at all — and one is
  inconclusive, against 9 of 9 gains on the two development corpora. Carrying the source `alpha`
  across costs 0.095 to 0.143 of nDCG@10 at p = 0.0001;
- **phase 6's stop condition applied.** The learned alignment beats late fusion on MAP (+0.0231,
  d = 0.259) and loses the primary endpoint (-0.0051), while reproducing rather than escaping the
  reachability artifact. Validation loss rises from the first epoch in all six runs. The
  transparent method stands as the recommended system;
- the phase-9 comparison is **declined**, with both reasons measured: the published dataset is
  not linked from the paper, and Joconde's `Sujet_Represente` is a faceted free-text phrase with
  58.3% of its terms occurring once, on which graded hierarchical relevance is undefined.

### Fixed

- **the matched-control draw was biased.** Hard positives sit in the low tail of the similarity
  distribution, so a symmetric caliper offered more candidates above the target than below, and
  controls came out reliably more similar to the query than the positives were. The visual arm,
  which the design requires at chance, read 0.395-0.468; a balanced draw puts it at 0.500-0.502;
- a quarter of the frozen Emblematica mottoes still carried undecoded character references, so a
  text encoder read five literal characters where a letter belonged;
- five `OPTIONAL` clauses in one Wikidata SPARQL query return 502; attributes are fetched one
  property at a time.

## 0.6.0 — 2026-09-16

### Added

- full-corpus Iconclass AI benchmark: 87 744 images, two-stage perceptual near-duplicate
  detection, book identifiers recovered from filenames, and a leakage-free 70 304/8 539/8 901
  partition;
- inferential statistics (`caypollard.statistics`): percentile bootstrap intervals, paired
  bootstrap differences, sign-flip permutation tests, Cohen's d and Cliff's delta;
- a second benchmark from Emblematica Online — 25 463 annotated emblems, 264 volumes, 271 902
  Iconclass assignments, and 30 402 transcribed mottoes, the project's first text modality;
- context-only graph projections and the bibliographic enrichment built from Munich IIIF
  manifests and the Emblematica catalogue;
- `protocol-v0.5`, `docs/RESULTS.md`, `docs/EMBLEMATICA_DATA_CARD.md`.

### Changed

- **relevance definition.** Bracketed Iconclass text keys are no longer stripped: each observed
  key is attached as a child of its base notation. Under the old definition notation 86 folded
  5 972 distinct mottoes onto one node, so any two emblems scored graded relevance 1.0 and the
  hard-pair benchmark was compromised — 1 of 250 mined hard positives shared an exact notation,
  against 186 of 250 after the fix. Both definitions remain selectable via `--key-policy`, and
  protocol v0.5 requires every nDCG claim to be reported under both;
- phase 7 reframed from an illustrative case study into the corpus that makes H2 testable.

### Fixed

- `--device auto` detected only CUDA and silently fell back to CPU on Apple silicon;
- `get_image_features` returns an output object rather than a tensor in Transformers 5, which
  broke the CLIP and SigLIP presets;
- SigLIP loaded a multimodal processor, requiring SentencePiece for a text branch never used;
- image extraction wrote nothing until completion and held every vector in memory, so a
  multi-hour run was indistinguishable from a stalled one and a single crash lost all of it;
- Emblematica parsing read a commented-out cataloguer's template as a real notation, inflating
  measured Iconclass coverage from 82.0% to 90.5%.

### Results

H1 holds on both corpora. Transparent late fusion improves test nDCG@10 over visual-only in all
six encoder-corpus combinations (d = 0.24-0.33, p = 0.0002 throughout). H2 and H3 are refuted on
both: hard positives are bibliographically unrelated — 243 of 250 share no attribute at all — so
a context-only graph cannot reach them. See [`docs/RESULTS.md`](docs/RESULTS.md).

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
