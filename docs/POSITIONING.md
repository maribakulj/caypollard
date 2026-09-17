# Positioning: retrieval is not completion

Phase 9's exit criterion is that the project can state precisely how it differs from and
complements multimodal knowledge-graph completion and enrichment work. This document does that,
using the project's own measured results as the contrast material rather than asserting a
difference in the abstract. Citations are the verified entries in
[`LITERATURE.md`](LITERATURE.md).

## The two tasks are not versions of each other

| | multimodal KG completion / enrichment | this project |
| --- | --- | --- |
| Question | given a graph, what triple is missing? | given an image, which other images are iconographically related? |
| Evidence used | entity images help predict a relation | graph structure helps rank images |
| Direction | images → graph | graph → images |
| What is held out | triples | **images, grouped by volume or creator** |
| Success | a correct head or tail entity | a better ranking under graded relevance |
| Failure mode guarded against | overfitting to graph structure | **target leakage: the annotation being ranked must not enter the graph** |

The last row is the one that matters most and it is not symmetric. A completion model may use
every attribute of an entity, because the held-out item is a *triple*. A retrieval model
evaluated against iconographic annotation may not use that annotation, because it is the ground
truth. This project's `G1` and `pure` projections therefore exclude Iconclass entirely, and the
Rijksmuseum projection excludes `depicts` for the same reason. A result obtained without that
exclusion is not comparable with one obtained with it, whatever the metric is called.

## What this project adds that a completion benchmark cannot show

Three of its findings are invisible to a completion evaluation, because they concern the
*ranking*, not the graph:

- **The fused system is four fifths its visual arm.** Top-10 overlap between the fused ranking
  and visual-only is 0.811, against 0.228 between two visual encoders. A completion metric has
  no place to report this; a retrieval system's user experiences it directly.
- **A gain can be real and generic at once.** Late fusion improves nDCG@10 in 9 of 9 conditions
  on two corpora, and the classification of *which* results move shows the gain is largely
  volume co-membership on one corpus and largely shared notation on the other. "The graph helps"
  is true in both and means different things.
- **Reachability bounds what a graph can be credited with.** `G1` is a disjoint union of one
  component per volume. Any similarity it reports between items of different volumes is an
  artifact of embedding a disconnected graph, not information. Completion work rarely needs this
  check because its queries are inside the graph by construction; retrieval work does, because
  its queries are pairs the graph may simply not connect.

## What completion work establishes that this project relies on

- That visual features can carry information a graph does not (Xie et al., 2017). This project's
  H1 measurement is the mirror image, and confirms it from the other side: graph neighbourhoods
  share only 0.08-0.18 of their top-10 with visual ones.
- That an extra modality is not automatically useful (Zhang, Zaporojets et al., 2026). This is
  why the protocol requires modality ablations, and it is what the V+G+T result shows: adding a
  text arm to V+G is selected at `alpha = 1.0`, weight zero.
- That French heritage data can be assembled into a multimodal KG at all (Zhang, Mimouni et al.,
  2026, WJoconde).

## Why no empirical comparison on WJoconde is offered

Two separate obstacles, recorded so the gap is legible rather than silent.

**The published dataset is not linked from the paper's abstract page**, so the baseline cannot be
reproduced without materials this pipeline does not have. The roadmap's own wording — "reproduce
a relevant published WJoconde baseline *where feasible*" — anticipated this.

**The underlying source diverges at the vocabulary level.** Joconde's national export is openly
licensed and reachable without a key, and `scripts/audit_joconde.py` samples it across seven byte
ranges of the 1.2 GB file:

| | measured on 22 957 sampled records |
| --- | ---: |
| records carrying a subject | 46.1% |
| distinct subject terms | 12 801 |
| subject-term occurrences | 78 700 |
| **terms occurring exactly once** | **58.3%** |
| records flagged as having an image | 79.5% |
| columns carrying an image URL | **none** |

`Sujet_Represente` is a free-text faceted phrase — a head term with parenthesised qualifiers,
such as `scène (intérieur : atelier d'artiste, peintre, homme, debout, chevalet)`. It is a
description, not a classification: **there is no parent relation to walk**, so graded
hierarchical relevance, this project's primary endpoint, is undefined on it. Exact-match
relevance degenerates as well, since most terms occur once.

This is the phase-9 answer in its most useful form. The interface ports — the manifest schema,
the splitting, the fusion, the evaluation all accept any corpus, as the Rijksmuseum ingestion
demonstrated in a day. **What does not port is the relevance definition**, and it is the
relevance definition, not the architecture, that makes a retrieval claim mean anything. A number
computed on Joconde under a facet-overlap relevance would not be comparable with the numbers in
[`RESULTS.md`](RESULTS.md), and reporting them in one table would be the error this document
exists to prevent.

## The claim the project is entitled to make

Graph-aware late fusion improves graded iconographic retrieval on two curated emblem and
book-illustration corpora, by an effect size an order of magnitude larger than the difference
between visual encoders, and the gain is concentrated in results that share a volume or a
notation with the query. It does not transfer to a third collection, the mixing weight is
actively harmful when carried across, and no configuration tested improves the disagreement
cases the project was built to study.

That is a retrieval result about two corpora. It is not a statement about knowledge-graph
completion, and nothing here should be read as one.
