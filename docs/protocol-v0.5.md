# Experimental protocol v0.5

**Status:** frozen after a defect was found in the v0.1–v0.4 relevance definition, and before
any result is reported under the corrected definition.

Protocol v0.5 inherits every leakage, hard-pair, fusion, and learned-alignment rule from
[`protocol-v0.4.md`](protocol-v0.4.md). It changes one thing: how bracketed Iconclass keys
enter graded relevance. Because that change moves a headline metric, both definitions are
retained and every result is reported under both.

## 1. The defect

Iconclass brackets carry two different things:

- `(+34)` — a **plus key**, a structural modifier of how a subject is shown;
- `(RHUBARB)`, `(HERMES TRISMEGISTOS)`, `(USQUE RECURRIT)` — a **text key**, naming what is
  depicted.

`normalize_notation` removed both. For most notations that is a harmless coarsening. For
notation **86 — proverbs, sayings, emblems** — it is not: the motto *is* the content, and
5 972 distinct mottoes in this corpus collapse onto the single node `86`. Any two emblem
images then scored graded relevance 1.0 to each other.

The damage was concentrated, not diffuse. Exact-match relevance was never affected, because
`exact_relevant_ids` indexes raw notations — so MAP, MRR and Recall@K are identical under both
definitions. Only nDCG and the hard-pair mining, which both consume
`image_hierarchical_similarity`, were wrong.

The consequence for the hard-pair benchmark was total. Mining selects visually distant pairs
whose semantic similarity is high, and the `86` collision is the cheapest way for two
unrelated emblems to reach exactly 1.0. Under the old definition **1 of 250** mined hard
positives shared an exact notation with its partner; under the corrected one, **186 of 250**.

## 2. The two definitions

| policy | treatment | flag |
| --- | --- | --- |
| `strip` | remove plus keys and text keys (v0.1–v0.4 behaviour) | `--key-policy strip` |
| `keep` | remove plus keys only; attach each **observed** text key as a child of its base notation | `--key-policy keep` |

Under `keep`, two different mottoes are siblings rather than synonyms: distance 2, graded
relevance 1/3, while the same motto repeated still scores 1.0. Only keys present in the corpus
are added — 12 851 of them — so the hierarchy grows by what the data contains rather than by
everything the vocabulary permits. A text key already carrying its own vocabulary node keeps
its real parent instead of being re-attached.

`keep` is the definition of record for v0.5. `strip` is retained so that v0.1–v0.4 numbers stay
reproducible and so the sensitivity analysis below can be re-run.

## 3. Mandatory sensitivity reporting

No nDCG-based claim may be published under one policy alone. Every reported nDCG table states
both values and their difference. A claim survives only if it holds under `keep`; agreement
under both is reported as a robustness result, disagreement as a finding in its own right.

Measured on the frozen v0.1 benchmark:

| condition | nDCG@10 `strip` | nDCG@10 `keep` | Δ |
| --- | ---: | ---: | ---: |
| DINOv2, full test split | 0.5352 | 0.5004 | −0.035 |
| CLIP, full test split | 0.5439 | 0.5029 | −0.041 |
| SigLIP, full test split | 0.5675 | 0.5274 | −0.040 |
| random control, `G1` sub-corpus | 0.3273 | 0.1966 | −0.131 |
| `G1` graph only, sub-corpus | 0.7270 | 0.5238 | −0.203 |
| DINOv2, sub-corpus | 0.7420 | 0.5639 | −0.178 |

Two things follow. Model ranking is unchanged on the full split — SigLIP leads under both — so
the defect never inverted a visual comparison. But `G1` loses the most of any condition
(−0.203), which says its apparent strength was disproportionately the motto artefact: emblem
plates of one volume all carry `86` keys and were scoring as mutually identical.

## 4. What survives

The transparent-fusion gain is robust to the definition:

| encoder | gain `strip` | gain `keep` |
| --- | ---: | ---: |
| DINOv2 | +0.0437 | +0.0387 |
| CLIP | +0.0248 | +0.0241 |
| SigLIP | +0.0301 | +0.0261 |

The validation-selected `alpha` is 0.25 under both. A conclusion that held only under the
defective definition would have been an artifact; this one is not.

## 5. What does not survive, and what replaces it

The hard-pair analysis reported before this protocol is void: it was computed on pairs that
were not hard positives. Re-mined under `keep`, with genuine pairs, the conclusion is
nonetheless the same and now properly founded:

- hard positives are **cross-volume**: 14 of 250 share a book, against 174 of 250 for hard
  negatives and 243 of 250 for easy positives;
- the `G1` graph score separates hard positives from easy negatives at AUC **0.498** — chance;
- it separates hard positives from hard negatives at AUC **0.241** — inverted, because hard
  negatives are largely same-volume pairs that the graph rewards.

So `G1` improves aggregate retrieval by recovering volume membership, and cannot address H2 or
H3, which concern pairs it holds no edge between. Publication gate 2 remains unsupported.

## 6. Constraint on graph enrichment

The bibliographic enrichment built from Munich IIIF manifests (505 volumes, 209 GND creators,
163 printers, 66 places) reaches **0 of 250** hard positives, under both policies and both
mining encoders. The reason is structural rather than a coverage shortfall: hard positives are
emblem-book pairs drawn across St Andrews, UIUC, Wolfenbüttel, Mnemosyne and Utrecht, whereas
the enriched volumes are Munich incunabula.

Any enrichment intended to test H2 must therefore target the emblem collections, and must
supply a relation that actually links two emblem books by different authors printed decades
apart. Shared creator, printer and place will not do so. The candidate relation is
`instance_of` — editions and translations of one source work — which is implemented and
awaiting catalogue coverage for those collections.
