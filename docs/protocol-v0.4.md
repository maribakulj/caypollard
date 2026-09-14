# Experimental protocol v0.4

**Status:** frozen before any full-corpus learned-alignment test evaluation.

Protocol v0.4 inherits every leakage, hard-pair, and transparent-fusion rule from
[`protocol-v0.3.md`](protocol-v0.3.md). It adds the first learned V+G condition.

## 1. Scope of the learned model

The visual and graph encoders remain frozen. Only two small projection heads are trained:

```text
visual embedding -> projection --+
                                 +--> shared space
graph embedding  -> projection --+
```

The default head is a single linear projection. A one-hidden-layer MLP is an allowed
predeclared ablation, not the default headline model.

## 2. Positive and negative pairs

A positive is the same heritage object represented in the visual and graph modalities.
Negatives are other aligned objects in the same training minibatch. Training uses symmetric
InfoNCE, so both visual->graph and graph->visual directions contribute to the loss.

No Iconclass relevance label is consumed directly by the alignment loss. Whether a graph
condition is evidence-bearing still depends on the v0.2/v0.3 `G0`/`G1`/`G2` leakage rules.
A projection trained from a `G0` taxonomy oracle remains an oracle experiment.

## 3. Partition discipline

- optimisation uses `train` identifiers only;
- epoch/checkpoint selection uses symmetric InfoNCE loss on `validation` identifiers only;
- `test` identifiers may be projected after fitting, but their labels and retrieval scores are
  not inspected during optimisation or checkpoint selection;
- the final test comparison is performed only after hyperparameters and seeds are frozen.

## 4. Default optimisation configuration

Unless a versioned experiment config states otherwise:

- projection dimension: 128;
- hidden layer: none;
- temperature: 0.07;
- AdamW learning rate: 1e-3;
- weight decay: 1e-4;
- batch size: 256;
- maximum epochs: 100;
- early-stopping patience: 15 epochs;
- seed: 42.

These defaults are engineering starting points, not results-derived choices.

## 5. Model selection and comparison

The learned condition does **not** replace transparent fusion. Its scientific comparison is:

1. strongest visual-only baseline;
2. strongest valid graph-only `G1` or `G2` baseline;
3. preregistered weighted late fusion;
4. preregistered graph reranking;
5. learned shared projection.

A learned model earns a headline claim only if it improves at least one central preregistered
metric or the hard-pair benchmark beyond the strongest transparent fusion baseline without a
materially unacceptable regression elsewhere.

## 6. Seed sensitivity

Before reporting a learned-model result, run at least three predeclared seeds. Report mean,
spread, and per-seed values. A single favourable seed is exploratory evidence only.

## 7. Collapse diagnostics

For every learned embedding artifact, report at minimum:

- mean/std/max off-diagonal cosine similarity;
- effective rank of the centered embedding matrix;
- retrieval hubness diagnostics already defined for graph baselines.

Very high mean cosine combined with low effective rank is treated as possible representation
collapse even when one retrieval metric looks good.

## 8. Ablations

The minimum learned-model ablations are:

- linear projection vs one-hidden-layer projection;
- V+G with valid `G1`/`G2` graph input;
- V only projected through an equivalently sized head when needed to isolate capacity effects;
- modality removal at inference where meaningful.

Text (`T`) enters only after V+G is understood. Adding every modality at once would make causal
interpretation needlessly miserable.

## 9. Frozen interpretation rule

The learned shared space tests whether a compact cross-modal alignment improves retrieval. It
does not establish historical influence, transmission, or iconographic causation. Those remain
claims requiring external historical evidence and, later, expert evaluation.
