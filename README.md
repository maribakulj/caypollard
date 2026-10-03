# caypollard

[![CI](https://github.com/maribakulj/caypollard/actions/workflows/ci.yml/badge.svg)](https://github.com/maribakulj/caypollard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**A research prototype for iconographic image retrieval: finding pictures that show the same
thing, across prints and other media, and showing why two pictures were judged close.**

Status: **exploratory prototype.** Not a finished study, not a tool for production use. The
experiments below were run, their results are reported with their limits, and some of the paths
tried were abandoned.

## What has been tested, and what it found

Three representations of an image were compared and combined: its pixels (DINOv2, CLIP,
SigLIP), a knowledge graph of its context (book, creator, date, place), and its text (emblem
mottoes). Relevance was defined by shared Iconclass notations.

- **Fusing graph and pixels helps on the two development corpora** (Iconclass AI Test Set,
  Emblematica Online): +0.019 to +0.039 nDCG@10, p = 0.0002.
- **The gain does not transfer** to a third collection (1 864 Rijksmuseum works): 1 condition in
  6 gains, and importing a mixing weight from another corpus costs 0.095–0.143 nDCG@10.
- **The hard-pair hypotheses (H2, H3) could not be supported.** Hard positives are mostly
  bibliographically unrelated, so a context graph holds no edge between them; and Iconclass
  describes what is shown, not what it means, so it cannot score a hard negative of meaning.
- **Mottoes see something else.** They retrieve well on their own but share only 4 % of their
  neighbours with the graph, and Iconclass relevance does not reward what they find.

| hypothesis | outcome |
| --- | --- |
| H1 — the graph carries signal the pixels do not | holds, on two corpora |
| H2 — the graph recovers iconographically close, visually distant pictures | not supported: no graph path between most such pairs |
| H3 — the graph suppresses visually close, iconographically distant pictures | not testable under Iconclass |
| H4 — the gain transfers to an unseen collection | does not hold |
| H5 — a learned model must beat transparent fusion to be worth it | it does not; transparent fusion is kept |

Details, controls and the errors corrected along the way: [`docs/RESULTS.md`](docs/RESULTS.md).

## What was tried and abandoned

An intermediate "visual language" (shapes, composition, pose, palette, named nodes) meant to
make a print and a painting comparable was explored on 21 128 objects. Its units, inspected one
picture at a time, did not correspond to what a person sees in the image, so its measurements do
not answer the question they were built for. It is kept for the record in
[`docs/archive/`](docs/archive/), not as a result.

## Where it goes next

The next aim is narrower than the original plan: **find the same figure or the same composition
across engravings and media** — copies, re-engravings, reused woodblocks, a print taken up in a
painting — and show which parts of the two pictures the match rests on.

Two rules follow from what went wrong before. Every representation is first inspected on real
pictures, decomposed one by one, before any aggregate measurement is made of it. And a failed
construction is reported as a failed construction, not as the refutation of an idea.

The study of emblems as such — how picture, motto, verse and iconographic code answer one
another — is a separate project.

## Corpora

| corpus | used for |
| --- | --- |
| **Iconclass AI Test Set** | first benchmark: expert iconographic labels on a hierarchy |
| **Emblematica Online (UIUC/HAB)** | second benchmark: pictures, mottoes, bibliographic context, Iconclass |
| **Rijksmuseum** (via Wikidata) | transfer test |

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

## Quick start

The project uses Python 3.11+ and is prepared for `uv`.

```bash
git clone https://github.com/maribakulj/caypollard.git
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

For the real corpus, read `docs/ICONCLASS_DATA_CARD.md` and `docs/protocol-v0.4.md` first. The project deliberately refuses to make a 3.1 GB research-data download an invisible side effect of setup.

For the learned projection-head experiments, install the optional PyTorch layer explicitly:

```bash
uv sync --extra dev --extra alignment
```

### Looking at the pool

Once the museum and emblem benchmarks and the representation tables exist, a local page shows
every picture as a thumbnail and, for any one of them, the neighbours each representation
returns, with what each neighbour shares with the query. It computes nothing and annotates
nothing; see [`viewer/README.md`](viewer/README.md).

```bash
make viewer-build   # thumbnails, item table, neighbour lists, Wikidata labels
make viewer         # http://127.0.0.1:8765/viewer/
```

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

## Non-goals

The prototype does **not** attempt to:

- create a universal ontology of cultural heritage;
- replace Iconclass or curatorial description;
- infer historical influence from embedding similarity alone;
- deploy a production graph database or distributed vector stack;
- train a foundation model from scratch;
- claim that vector proximity is equivalent to art-historical interpretation.

## Roadmap

[`ROADMAP.md`](ROADMAP.md) records the original phased plan of the graph-fusion study. It is
kept as history; it is no longer the plan.

## Licence

Code in this repository is released under the **MIT License**. Source images, metadata, vocabularies, and external datasets retain their own rights and licences; see dataset-specific provenance documentation before redistribution.
