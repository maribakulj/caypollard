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

### Missing grouping provenance in `data.json`

The public annotation map exposes filenames and labels but does not itself supply book, edition, creator, or source-object grouping fields. Therefore a random or stable item-hash split is only a **diagnostic baseline**. It is not sufficient evidence against leakage between copies, editions, or visually duplicated material.

The benchmark design therefore requires, before headline multimodal results:

1. exact byte-level duplicate checking after image acquisition;
2. near-duplicate grouping using perceptual and/or embedding-based similarity;
3. collection/work/edition grouping whenever trustworthy provenance can be recovered;
4. reporting results under more than one split regime.

### Multi-label long tail

Images can have several Iconclass notations, often at different depths and with keys/qualifiers. Frequency, depth, and co-occurrence must be reported rather than collapsing the task to a flat single-label problem.

### Circular evaluation risk

Iconclass cannot simultaneously be hidden target, direct model input, and sole evaluation authority without care. Experiments must state exactly which graph edges/labels are visible during fitting and which are held out.

## Phase-1 generated artifacts

A full run of `scripts/build_iconclass_benchmark.py` produces:

- `manifest.jsonl` — deterministic canonical image records;
- `audit.json` — annotation and hierarchy coverage statistics;
- `iconclass_edges.csv` — parent/child graph;
- `iconclass_skos.ttl` — compact SKOS graph export.

These derived artifacts must be regenerated from pinned sources for publication releases.
