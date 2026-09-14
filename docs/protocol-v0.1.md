# Experimental protocol v0.1

**Frozen:** 2026-09-14
**Scope:** Phase 1–6 controlled Iconclass benchmark
**Status:** pre-results protocol; changes affecting headline metrics require a new protocol version.

## 1. Primary question

Can graph-structured iconographic knowledge improve retrieval of iconographically related heritage images beyond visual embeddings alone, particularly when visual and semantic similarity disagree?

## 2. Units

- Query unit: one heritage image.
- Candidate unit: another image in the same evaluation pool.
- Semantic evidence: expert-assigned Iconclass notations and their hierarchy.
- Visual evidence: frozen pretrained image representation unless a later protocol explicitly permits fine-tuning.

## 3. Model families

Primary comparison:

1. `V`: visual-only retrieval;
2. `G`: graph-only retrieval;
3. `V+G-late`: normalized weighted late fusion;
4. `V+G-rerank`: visual candidate generation with graph reranking;
5. `V+G-joint`: learned shared-space projection, only after the transparent baselines are established.

Textual descriptions are a separate ablation (`T`, `V+T`, `G+T`, `V+G+T`) and must not be silently folded into the graph condition.

## 4. Primary endpoint

**nDCG@10 using graded hierarchy relevance.**

The phase-1 transparent relevance function maps shortest taxonomic distance `d` through a common ancestor to:

`relevance = 1 / (1 + d)`

Exact concept identity therefore scores 1, a parent/child relation scores 0.5, and unresolved/disconnected concepts score 0. The function is deliberately simple and must be compared with alternatives before publication.

For multi-label images, pair relevance is the maximum relevance over all query-label/candidate-label pairs. A later protocol may additionally test mean or optimal-matching aggregation.

## 5. Secondary endpoints

- Recall@1/5/10 under exact-label overlap;
- MRR;
- mAP;
- visual-vs-graph neighbour overlap;
- performance stratified by label frequency and hierarchy depth;
- hard-positive and hard-negative subsets.

## 6. Split regimes

### Diagnostic split A: stable item hash

Used for pipeline debugging and early baselines only. Assignment is deterministic from item ID and a published seed.

### Preferred split B: provenance groups

Objects sharing a work, book, edition, or other defensible source group must remain in the same partition whenever such metadata is available.

### Preferred split C: duplicate groups

Exact and near-duplicate images must remain in the same partition. Any residual cross-split duplicate rate must be reported.

**No headline claim may rely only on split A.**

## 7. Tuning discipline

- Hyperparameters, including late-fusion alpha, are selected on validation data only.
- Test data must not be used to select normalization bounds.
- Model revisions, preprocessing, seeds, and vector dimensions are recorded in configs and result metadata.
- Repeated stochastic experiments report seed-level values and aggregate uncertainty.

## 8. Hard-pair construction

Hard positives: semantically close according to held-out expert structure but visually distant according to a frozen visual baseline.

Hard negatives: visually close according to the same frozen baseline but semantically distant.

Thresholds are selected on training/validation partitions and then frozen before test evaluation.

## 9. Falsification / stop criteria

H1 is weakened if graph-only neighbourhoods show no interpretable or quantitative semantic signal beyond chance/frequency baselines.

The multimodal claim fails if `V+G` does not outperform `V` on the primary endpoint under leakage-controlled splits, or if gains disappear after controlling for text/provenance leakage.

A learned joint architecture is not justified unless it beats transparent late fusion or reranking under the same protocol.

## 10. External validity

Only after the controlled benchmark succeeds should the method be evaluated on Emblematica and an external collection such as Rijksmuseum. Cross-collection results are reported separately from in-domain results.

## 11. Reproducibility artifacts

Every reported run should identify:

- dataset manifest SHA-256;
- Iconclass vocabulary ref and file SHA-256;
- split seed/version;
- model identifier and immutable revision where available;
- config file;
- code commit;
- environment lockfile;
- raw per-query ranking/metric outputs where redistribution permits.
