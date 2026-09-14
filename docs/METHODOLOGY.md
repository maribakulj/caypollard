# Methodology Notes

## Principle

The project compares representation spaces before attempting to collapse them into one. Each modality should first be evaluated independently.

## Representation pipeline

```text
IMAGE -----------------> visual encoder -----------------> v_i
  |
  +--------------------> text/caption encoder -----------> t_i
  |
  +-> object/entity ---> heritage knowledge graph -------> g_i

v_i, t_i, g_i
      |
      +-> baselines / reranking / learned projections
      |
      +-> retrieval ranking + explanations
```

## Baseline 1: visual cosine similarity

For normalised visual vectors `v_i` and `v_j`:

`S_v(i,j) = v_i · v_j`

## Baseline 2: graph cosine similarity

Graph experiments are split into oracle/control and evidence-bearing conditions. Direct image → Iconclass target edges may be used only in an explicitly labeled taxonomy oracle. Headline graph conditions must use context-only structure or mask evaluation-image target edges as defined in `GRAPH_PROJECTIONS.md` and `protocol-v0.3.md`.

For graph vectors `g_i` and `g_j`:

`S_g(i,j) = g_i · g_j`

If the KGE model's native scoring function is relational rather than entity-neighbourhood similarity, report clearly how an entity-to-entity similarity is derived.

## Baseline 3: late fusion

After calibration/normalisation:

`S(i,j) = alpha * S_v(i,j) + (1-alpha) * S_g(i,j)`

Tune `alpha` on validation data only. The initial grid is `[0, 0.25, 0.5, 0.75, 1]`; ties on validation hierarchical nDCG@10 favour the larger visual weight. Min/max similarity bounds are also fitted from validation pairs only.

## Baseline 4: graph reranking

1. Retrieve top `N` by `S_v`.
2. Compute graph score within the candidate set.
3. Rerank with a fixed/tuned combination.

This is attractive for production heritage systems because graph computation can be restricted to plausible visual candidates.

## Learned alignment

Use frozen source encoders initially:

`z_v = P_v(v)`

`z_g = P_g(g)`

Protocol v0.4 trains two deliberately small projection heads with symmetric InfoNCE. The same object across visual and graph modalities is the positive pair; other objects in the minibatch are negatives. Optimisation uses train IDs only, checkpoint selection uses validation InfoNCE only, and test rows may be projected only after fitting.

Every learned run records train/validation partition digests plus off-diagonal cosine and effective-rank collapse diagnostics. At least three predeclared seeds are required before a learned result is treated as confirmatory. If the learned model cannot beat transparent fusion, increasing parameter count is not evidence of progress.

## Hard-pair evaluation

Let `d_v(i,j)` be visual distance and `d_h(i,j)` hierarchical iconographic distance.

Candidate hard positives satisfy:

- high `d_v`;
- low `d_h`.

Candidate hard negatives satisfy:

- low `d_v`;
- high `d_h`.

Thresholds are fixed by `protocol-v0.3`: visual close/distant thresholds come from the 95th and 50th percentiles of a fixed validation random-pair similarity sample, while semantic close/distant thresholds are `>=0.5` and `<=0.2` under the frozen hierarchy relevance. The resulting test-pair artifact is checksummed before fusion evaluation.

## Hierarchical relevance

Potential relevance functions include:

- inverse shortest path;
- depth-normalised path distance;
- lowest-common-ancestor depth;
- information-content similarity.

For multi-label images, define an explicit aggregation strategy such as best-match average rather than silently taking the maximum.

## Explainability output

For each graph-aware ranking change, retain:

```json
{
  "query_id": "...",
  "candidate_id": "...",
  "visual_score": 0.0,
  "graph_score": 0.0,
  "fused_score": 0.0,
  "shared_concepts": [],
  "graph_path": [],
  "rank_visual": 0,
  "rank_fused": 0
}
```

These records can support both quantitative analysis and qualitative case studies.
