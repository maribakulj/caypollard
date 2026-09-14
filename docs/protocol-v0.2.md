# Experimental Protocol v0.2 — target-leakage controlled

**Status:** frozen before full-model benchmark results.  
**Supersedes:** `protocol-v0.1.md` for headline KG/multimodal claims.  
**Date:** 2026-09-14.

This protocol retains the metrics and split discipline of v0.1 and adds an
explicit separation between oracle graph controls and evidence-bearing graph
conditions.

## 1. Primary question

Can graph-structured cultural-heritage context improve iconographic image
retrieval beyond visual embeddings **without directly exposing the evaluation
labels for test images to the graph model**?

## 2. Primary endpoint

Hierarchical nDCG@10, using held-out Iconclass structure exactly as defined in
v0.1 (`1 / (1 + d)`, maximum over image-label pairs).

Secondary endpoints remain exact-label Recall@1/5/10, MRR, mAP, neighbour-space
overlap, hard-positive/hard-negative performance, and depth/frequency strata.

## 3. Representation conditions

- `V`: visual-only.
- `G0-taxonomy-oracle`: direct taxonomy-derived image representation. Control
  only; cannot support the main claim.
- `G1-context`: contextual KG excluding target Iconclass edges and text.
- `G2-masked`: graph may contain Iconclass structure, but target edges for
  evaluated images are masked before fitting/inference according to the declared
  regime.
- `V+G1`, `V+G2`: evidence-bearing fusion conditions.
- `T`, `V+T`, `G+T`, `V+G+T`: explicit text ablations.

## 4. Target-leakage rule

No headline KG or multimodal result may use a test image's evaluation Iconclass
assignment as an input feature, graph edge, generated prompt, retrieval key, or
training target that remains visible at inference time.

Any oracle condition that violates this rule must be labeled `oracle` in result
files, figures, and prose.

## 5. Graph projection artifacts

Each run must record:

- projection identifier (`G0`, `G1`, `G2`, etc.);
- entity and relation types included;
- whether target semantic edges are present;
- masked-edge artifact and checksum where applicable;
- whether the regime is inductive or transductive;
- graph node/edge counts after masking;
- model and seed.

See `docs/GRAPH_PROJECTIONS.md`.

## 6. Split and duplicate discipline

The v0.1 split hierarchy remains in force: provenance-group and duplicate-group
splits are preferred; stable item-hash splits are diagnostic only. No headline
claim may rely only on item-hash splitting.

## 7. Tuning

Hyperparameters are selected on validation data only. Test labels may be read
only by the evaluator. Fusion weights, graph masking policy, model selection,
and stopping criteria must be fixed before final test evaluation.

## 8. Interpretation

A gain from `G0-taxonomy-oracle` demonstrates only that the evaluation taxonomy
contains useful ranking information, which is nearly definitional. The research
claim requires gains from `G1-context` or `G2-masked`, ideally surviving
cross-collection transfer and expert assessment.
