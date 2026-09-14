# Experimental Protocol v0.3 — hard-pair and transparent-fusion freeze

**Status:** frozen before full-corpus hard-pair mining and multimodal test evaluation.  
**Supersedes:** `protocol-v0.2.md` for hard-pair construction and transparent fusion.  
**Date:** 2026-09-14.

Protocol v0.3 retains every target-leakage rule from v0.2 and adds predeclared
procedures for relation-aware graph baselines, hard visual/semantic disagreement
pairs, and transparent visual+graph fusion.

## 1. Graph baselines

The graph model family is separated into controls and evidence-bearing runs:

- `G0`: taxonomy oracle/control only;
- `G1`: context-only graph, excluding evaluation Iconclass edges and text;
- `G2`: masked-label graph, with evaluation image -> Iconclass target edges removed;
- topology controls: adjacency-SVD and Node2Vec/DeepWalk-style PPMI-SVD;
- predicate-aware control: RDF2Vec-style predicate walks + PPMI-SVD;
- relation-aware KGE: ComplEx and RotatE through a pinned PyKEEN environment.

The PPMI-SVD controls are deliberately labelled `*-style` and are not reported
as canonical reproductions of the original Word2Vec implementations.

## 2. Hard-pair threshold calibration

Visual thresholds are derived from **validation embeddings only**. Using a fixed
seed, sample up to 20,000 unique random validation pairs:

- visually close: similarity >= 95th percentile;
- visually distant: similarity <= 50th percentile.

Semantic thresholds use the frozen hierarchy relevance `1 / (1 + d)`:

- semantically close: relevance >= 0.5 (same concept or one hierarchy edge);
- semantically distant: relevance <= 0.2 (distance >= 4 or disconnected).

No test embedding or test label may be used to set these cutoffs.

## 3. Four hard-pair classes

- easy positive: visual close + semantic close;
- hard negative: visual close + semantic distant;
- hard positive: visual distant + semantic close;
- easy negative: visual distant + semantic distant.

Mining combines visual top-k candidates, local semantic BFS candidates, and a
fixed-seed random distant-candidate sample. The final artifact is deduplicated,
balanced up to a declared per-class cap, persisted as canonical JSONL, and
checksummed before fusion experiments.

## 4. Late fusion

For visual similarity `V` and graph similarity `G`:

`score = alpha * scale(V) + (1 - alpha) * scale(G)`

The alpha grid is fixed to `[0, 0.25, 0.5, 0.75, 1]` for the first benchmark.
Each modality's min/max bounds are fitted on the same fixed validation-pair
sample only. Scores are not clipped.

Alpha is selected by validation mean hierarchical nDCG@10. Exact ties are
resolved in favour of the **higher visual weight**, requiring graph complexity
to earn its influence. The selected alpha is then evaluated exactly once on the
test split.

Because external min-max scaling is affine, ranking is equivalent to a weighted
sum of raw cosine similarities. Caypollard implements this ranking exactly as a
weighted concatenated cosine space, which allows reuse of the same evaluator.

## 5. Graph reranking

The second transparent fusion baseline retrieves a fixed visual candidate pool
(default top 100) and reranks only those candidates using the same
validation-calibrated visual/graph scores. Candidate-pool size and alpha are
fixed on validation data before test evaluation.

## 6. Required diagnostics

Every graph/fusion result must report or preserve enough information to compute:

- hierarchical nDCG@10, Recall@1/5/10, MRR, and mAP;
- graph degree vs neighbour-occurrence (hubness) correlation;
- visual/graph neighbourhood overlap;
- performance on all four hard-pair classes;
- the fraction of top-k rankings changed by graph information;
- projection ID, graph checksum, masked-edge count, model, seed, and dependency versions.

## 7. Interpretation gate

A fusion gain counts as evidence only for `V+G1` or `V+G2`. Improvements from
`V+G0` or any graph representation containing the evaluated image's own target
Iconclass edge are oracle/control results and cannot support the headline claim.
