# Iconclass phase-1 data card

**Status:** source integration implemented; full 3.1 GB image archive not vendored in this repository.

## Purpose

The Iconclass AI Test Set is the controlled benchmark for testing whether graph-structured iconographic knowledge adds retrieval signal beyond visual embeddings. It is not treated as a complete historical corpus and it is not used to infer historical influence.

## Official source

- Dataset landing page: https://iconclass.org/testset/
- Suggested source citation: Etienne Posthumus, “Iconclass AI Test Set”, February 2020.
- Official archive: approximately 3.1 GB.
- Published archive MD5: `779ba2ca9e977c58d818e3823a676973`.
- Published size: 87,749 images.
- Images are described by the source as having at most 500 pixels on the longest side.
- `data.json` maps image filenames to lists of assigned Iconclass notations.

The project downloader verifies the official MD5 and requires the explicit `--accept-large-download` flag. CI never downloads the full archive.

## Vocabulary source

The hierarchy is sourced independently from the open `iconclass/data` repository:

- https://github.com/iconclass/data
- license: CC0-1.0
- default reproducibility pin in `scripts/fetch_iconclass_core.py`: `0aeced694cf0a57dd5ff5dfb4587da99f698b686`

The parser reads the repository's `notations.txt` directly. This avoids silently changing hierarchy semantics when the online vocabulary is updated.

## Rights and redistribution

The Iconclass site states that the test-set images were digitized by Arkyves or supplied by partners under open licences, while also noting that more detailed per-image reuse information is desirable. Consequently this repository does **not** redistribute the full image archive by default. It stores source identifiers, checksums, derived manifests, code, and tiny non-image fixtures.

Iconclass classification codes in the test-set page are released with a CC0 waiver. The separate `iconclass/data` repository is also CC0-1.0.

## Known limitations

### Partial grouping provenance in `data.json`

The public annotation map exposes filenames and labels but supplies no book, edition, creator, or source-object grouping **field**. The filenames themselves, however, are not opaque: a minority encode the digitised volume a plate was cut from, as `<collection>_<shelfmark>_<plate>` — `uiuc_832w91o_pic012`, `embhab_hab_li77441_pic142`, `bsb00027852_559`, `120d13_002r_min_a1`.

`caypollard.datasets.iconclass_provenance` recovers that identifier conservatively, returning `None` wherever the volume cannot be established. Measured on the full test set:

| | images | share |
| --- | ---: | ---: |
| with a recoverable book identifier | 14 645 | 16.7% |
| of which emblem-book plates | 5 159 | 5.9% |
| without recoverable provenance | 73 099 | 83.3% |

across 1 246 books, the largest holding 299 images. The unattributed majority is dominated by flat per-item identifiers — `IIHIM_-859728949` alone accounts for 63 524 images, alongside `ursicula`, `folger`, `dpd`, and `biblia_sacra`. Reading those leading tokens as volumes would pool tens of thousands of unrelated images into one group and destroy the very splits grouping is meant to protect, so they are left null.

Two consequences follow. First, book-level grouping is available for evaluation on a sub-corpus but not corpus-wide, so split-regime reporting must state which regime applies to which subset. Second, `partOf(image, book)` is the only non-target relation this source carries, which makes it the only material from which a context-only `G1` projection could be built here at all — see [`GRAPH_PROJECTIONS.md`](GRAPH_PROJECTIONS.md).

### Near-duplicate detection

Exact byte duplicates are caught by SHA-256. Near duplicates — rescans, recompressions, a plate photographed twice — are caught by a two-stage perceptual procedure in `caypollard.datasets.near_duplicates`, deliberately independent of the visual encoder under evaluation so that cleaning the benchmark does not entangle it with the model being tested.

1. A 64-bit difference hash generates candidate pairs through banding. With `threshold + 1` bands the shortcut is exact rather than approximate: by the pigeonhole principle no pair within the threshold can be missed.
2. A 256-bit difference hash confirms each candidate. This stage is required, not optional: the coarse hash reads *layout* as much as content, and on this corpus alone it merged 211 unrelated paintings that share a photographic mount, and pages of blackletter text that share a two-column frame.

The confirmation threshold of 28 bits was calibrated by manual inspection rather than chosen for roundness. Byte-identical images score 0; sampled genuine near duplicates — one woodcut in two states, an ink drawing and its red-chalk version, a plate rescanned at a different tone — still confirmed at distances up to 25, while the first false merges appeared from 35.

Grouping signals are then merged into one transitively closed partition (`combine_group_keys`), because splitting on either signal alone still lets the other leak: a rescan of a plate under a new filename, or a second plate from the same book.

Results should still be reported under more than one split regime.

### Multi-label long tail

Images can have several Iconclass notations, often at different depths and with keys/qualifiers. Frequency, depth, and co-occurrence must be reported rather than collapsing the task to a flat single-label problem.

### Recall@K is not informative on this benchmark

Relevance here is *shared Iconclass notation*, and notations are widely shared, so the
relevant pool per query is large: measured on the 8 901-image test split, the median query
has **176 exactly relevant candidates** and the mean has 368, up to 1 995.

Recall@K is therefore bounded far below 1 by construction — the mean theoretical ceiling on
Recall@10 is 0.247, so an observed 0.034 represents 13.7% of what is attainable, not 3.4% of
the relevant material. Reporting it as a headline number invites reading a strong ranker as
a failure.

Report **precision-oriented** quantities instead: nDCG@10 with graded hierarchical relevance,
MAP, MRR, and precision@10 (≈ 0.37 for the visual baselines, i.e. roughly four of the top ten
results share a notation with the query). Recall@K is retained in the artifacts for
comparability with published work, but should carry the pool size whenever it is quoted.

A second consequence concerns significance. With more than 8 000 queries, differences of
under one nDCG point reach p < 0.001 while Cohen's d stays near 0.05. Every paired comparison
in this project must therefore report an effect size beside the p-value; `caypollard.statistics`
returns both, plus Cliff's delta, from a single call for exactly this reason.

### Circular evaluation risk

Iconclass cannot simultaneously be hidden target, direct model input, and sole evaluation authority without care. Experiments must state exactly which graph edges/labels are visible during fitting and which are held out.

## Phase-1 generated artifacts

A full run of `scripts/build_iconclass_benchmark.py` produces:

- `manifest.jsonl` — deterministic canonical image records;
- `audit.json` — annotation and hierarchy coverage statistics;
- `iconclass_edges.csv` — parent/child graph;
- `iconclass_skos.ttl` — compact SKOS graph export.

These derived artifacts must be regenerated from pinned sources for publication releases.
