# Results

All figures below come from frozen benchmarks under protocol v0.5, with graded relevance
computed under the `keep` key policy unless a `strip` column says otherwise. Every nDCG figure
carries a bootstrap interval over queries in the artifact it comes from; intervals are omitted
here only where the comparison is the point rather than the level.

## What the project set out to test

> Can knowledge-graph structure improve the retrieval of iconographically related
> cultural-heritage images beyond what visual embeddings alone recover?

Two claims were preregistered as the central targets: **H1**, that graph representations carry
information not recoverable from pixels, and **H2/H3**, that graph-aware retrieval specifically
improves the disagreement cases — images that are iconographically close but visually unlike,
and vice versa.

## Headline

**H1 holds and replicates on two independent corpora. The transparent fusion gain replicates in
every encoder-corpus combination tested, and the text modality adds an independent smaller gain.
H2 is not supported, and the reason is structural rather than empirical: on the corpus where the
graph could reach a cross-volume pair it separates hard positives at 0.505, and on the corpus
where it appears to separate them at 0.649 it provably holds no path between them. H3 is not
testable under this ground truth at all.**

This document has reported H2 wrongly twice, each time because a control was missing. Both
readings are retained below rather than deleted, with the control that overturned each.

## The two benchmarks

| | Iconclass AI Test Set | Emblematica Online |
| --- | ---: | ---: |
| images / emblems | 87 744 | 25 463 |
| Iconclass assignments | 391 244 | 271 902 |
| labels per item | 4.46 | 10.68 |
| volumes with recoverable provenance | 1 246 (16.7% of images) | 264 (100%) |
| volumes with a cross-volume edge | see below | 263 of 264 |
| text modality | none | 30 402 mottoes |

The first corpus is a sample of Arkyves, a subscription database, so its filenames carry
internal collection codes with no public key to any catalogue. Book-level provenance was
recovered for 16.7% of images by parsing shelfmarks out of filenames; the bibliographic
enrichment built on it (505 Munich IIIF manifests, 209 GND creators) reached **0 of 250** hard
positives. The second corpus was ingested precisely to remove that limitation.

Both benchmarks report zero group leakage and zero checksum leakage across their splits.

## H1 — the graph carries complementary signal

On the Iconclass `G1` sub-corpus (1 714 test images), the graph shares only **0.15-0.18** of its
top-10 neighbours with a visual encoder, against **0.32-0.37** between two visual encoders. The
information is real, not a restatement of visual proximity: `G1` reaches nDCG@10 0.727 against a
0.329 random control run through the identical ranking and metric path (d = 1.40).

On Emblematica (2 804 test queries), every condition clears the random control decisively:

| condition | nDCG@10 | MAP | MRR |
| --- | ---: | ---: | ---: |
| random control | 0.4312 | 0.1312 | 0.2677 |
| graph `book` | 0.7036 | 0.3554 | 0.6215 |
| graph `full` | 0.7020 | 0.3555 | 0.6208 |
| graph `pure` | 0.7049 | **0.3643** | 0.6210 |
| DINOv2 | 0.7311 | 0.3430 | 0.7110 |
| CLIP | 0.7293 | 0.3515 | 0.6933 |
| SigLIP | **0.7368** | 0.3511 | **0.7132** |

Graphs trail the visual encoders on nDCG and MRR but **beat them on MAP** — `pure` at 0.3643
against DINOv2's 0.3430.

## The transparent fusion gain replicates

Preregistered late fusion, with `alpha` selected on validation only, improves test nDCG@10 over
visual-only in **every one of nine encoder-corpus-graph combinations tested**, each at
p = 0.0002.

| corpus | encoder | graph | alpha | fusion | visual | gain | d |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Iconclass | DINOv2 | `G1` | 0.25 | 0.6026 | 0.5639 | **+0.0387** | 0.33 |
| Iconclass | CLIP | `G1` | 0.25 | 0.5980 | 0.5739 | **+0.0241** | 0.26 |
| Iconclass | SigLIP | `G1` | 0.25 | 0.6015 | 0.5754 | **+0.0261** | 0.27 |
| Emblematica | DINOv2 | `pure` | 0.75 | 0.7624 | 0.7311 | **+0.0313** | 0.29 |
| Emblematica | CLIP | `pure` | 0.75 | 0.7483 | 0.7293 | **+0.0190** | 0.24 |
| Emblematica | SigLIP | `pure` | 0.75 | 0.7571 | 0.7368 | **+0.0203** | 0.26 |

Two independent corpora, two ingestion pipelines, effect sizes of 0.24-0.33 throughout. For
scale, the difference *between* visual encoders on the same data reaches significance at
d ≈ 0.05 — an order of magnitude smaller. The selected `alpha` differs by corpus (0.25 on
Iconclass, 0.75 on Emblematica), so the balance between modalities is corpus-specific even
where the direction of the gain is not.

The gain also survives the relevance-definition change that invalidated the hard-pair analysis
(see below): on Iconclass it is +0.0437/+0.0248/+0.0301 under `strip` and
+0.0387/+0.0241/+0.0261 under `keep`.

One caveat belongs here rather than in a footnote. The visual encoders partly encode
*digitisation format* — full-page scan versus cropped pictura — and format correlates with
holding library, which correlates with the book attributes the graph carries. Aggregate nDCG is
scored against Iconclass relevance, which is independent of format, so the gain is not an
artifact of it; but no claim about *why* the modalities are complementary should rest on the
aggregate alone.

## The model matrix, filled in

Phase 0 declared seven conditions — V, G, T, V+T, G+T, V+G, V+G+T — and until now T had no
implementation at all. Mottoes are transcribed for 97.7% of Emblematica emblems, in Latin,
German, French, Dutch and English, and 5 544 of the 5 889 evaluation emblems carry one. They are
encoded with LaBSE under mean pooling over the attention mask.

All rows below are computed on the emblems that carry a motto, so V reads 0.7178 here rather
than the 0.7293 it reaches on the full evaluation set; comparing a partially covered modality
against a fully covered one on different query sets would not be a comparison.

| condition | nDCG@10 | against | difference | p | d |
| --- | ---: | --- | ---: | ---: | ---: |
| T (mottoes alone) | 0.6281 | — | — | — | — |
| G (`pure`) | 0.6829 | — | — | — | — |
| V (DINOv2) | 0.7178 | — | — | — | — |
| G+T | 0.7026 | G | +0.0197 | 0.0001 | 0.163 |
| V+T | 0.7319 | V | +0.0140 | 0.0001 | 0.173 |
| V+G | 0.7450 | V | +0.0272 | 0.0001 | — |
| V+G+T | 0.7450 | V+G | **+0.0000** | 1.0000 | 0.000 |

Three readings follow. **Text is a real and independent signal**: mottoes alone rank at 0.628,
far above the 0.431 random control, and improve both other modalities significantly. **Text is
weaker than either** visual or graph on its own. And **text is redundant once the graph is
present**: the validation sweep for V+G+T selects `alpha = 1.0`, giving the text arm zero weight,
so the condition is V+G exactly. Whatever the mottoes contribute, the bibliographic graph has
already contributed it — unsurprising once one notices that emblems of one volume share a
compositor's phrasing as well as a printer.

The dissociation matters for the hard cases: this same text modality scores **0.422** on hard
positives, below chance. A multilingual semantic encoder, which does not need shared words to
match paraphrases, does not bring together two emblems that share an Iconclass concept.

## Does the combination rule matter?

The weighted sum is one rule among several, and the flooding behaviour that costs hard positives
their place — a plateau of near-equal graph scores from a query's own volume taking the top of
the list — is a property of summation. Four alternatives were run through the identical
calibration, alpha-selection and evaluation path. The `linear` row reproduces the frozen
published figure exactly, which is the check that nothing else changed.

| rule | Iconclass · DINOv2 | Iconclass · CLIP | Iconclass · SigLIP | Emblematica · DINOv2 |
| --- | ---: | ---: | ---: | ---: |
| `linear` (frozen baseline) | **0.6026** | **0.5980** | **0.6015** | 0.7624 |
| `product` (geometric mean) | 0.6025 | 0.5979 | 0.6013 | **0.7637** |
| `rrf` (reciprocal rank) | 0.5848 | 0.5835 | 0.5917 | 0.7469 |
| `rank_linear` | 0.5767 | 0.5874 | 0.5864 | 0.7311 → collapses to V |
| `max` | 0.5247 | 0.5739 → collapses to V | 0.5754 → collapses to V | 0.7311 → collapses to V |

No rule beats the weighted sum by more than 0.0013, and three of them lose to it. Rank-based
rules discard score magnitude, which is most of the information; `max` is frequently sent to
`alpha = 1.0` by the validation sweep, meaning the sweep prefers no fusion at all to that rule.
A collapsed rule is labelled as such in the artifacts rather than printed as a fusion result.

On hard positives the ordering is the same, so the combination rule is not what stands between
the current system and H2. What stands there is that the graph holds no edge to reach the pair.

## What the gain is made of

A mean gain of +0.03 is compatible with a nudge on every query or a rescue of a few. Comparing
the visual and fused top-10 of every test query, under the identical relevance path, separates
them. The `linear` rule reproduces the published nDCG exactly through this second code path,
which is the check that nothing else changed.

| corpus | encoder | top-10 changed | items swapped | improved | degraded | unchanged | top decile of the gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iconclass | DINOv2 | 66.5% | 3.37 | 697 | 286 | 723 | 35.7% |
| Iconclass | CLIP | 54.4% | 2.23 | 577 | 243 | 886 | 35.4% |
| Iconclass | SigLIP | 51.2% | 2.31 | 560 | 225 | 921 | 35.1% |
| Emblematica | DINOv2 | 64.9% | 1.89 | 939 | 542 | 1 323 | 39.7% |
| Emblematica | CLIP | 58.8% | 1.20 | 808 | 509 | 1 487 | 45.1% |
| Emblematica | SigLIP | 62.8% | 1.43 | 852 | 565 | 1 387 | 42.9% |

The gain is broad rather than concentrated — a third to a half of it comes from the top decile of
improving queries, and a third of queries are untouched — and promoted items are more relevant
than the items they displace in every condition, by 0.11 to 0.16 of graded relevance.

**What a promoted item shares with its query is the substantive finding**, because it separates
an iconographic retrieval from the generic contextual one.

| corpus | encoder | notation and context | context only | notation only | nothing |
| --- | --- | ---: | ---: | ---: | ---: |
| Emblematica | DINOv2 | 54.7% | 45.2% | — | 0.1% |
| Emblematica | CLIP | 62.7% | 37.2% | — | 0.1% |
| Emblematica | SigLIP | 63.3% | 36.6% | — | 0.1% |
| Iconclass | DINOv2 | 25.8% | 54.0% | 1.4% | 18.8% |
| Iconclass | CLIP | 20.6% | 50.6% | 2.6% | 26.2% |
| Iconclass | SigLIP | 24.5% | 45.9% | 2.5% | 27.1% |

On Emblematica the majority of promotions share an Iconclass notation with the query as well as a
bibliographic attribute, so the gain there is largely iconographic. On Iconclass only a fifth to
a quarter do, and roughly half share nothing but the volume — the generic contextual gain
described earlier, now measured rather than inferred.

**Regressions have a single mechanism.** In the worst case on Emblematica, nDCG falls 0.43
because the graph promotes five plates of the query's own volume at graded relevance 0.14 to
0.25 and evicts a cross-volume emblem at relevance 1.0 that shares notation `46C215`. That is
flooding by a saturated context score, and it is the same behaviour that makes hard positives
difficult: the items H2 is about are exactly the ones a volume plateau displaces.

## H2 and H3 — not supported, and now for a structural reason

This section has been rewritten twice. Both earlier readings are recorded at the end, because
each was overturned by a control that should have been there from the start, and the sequence is
the useful part.

### The measurement as it now stands

Hard positives are mined within one collection (`--stratum-key collection`), so both items share
a scanning format. Each is paired with a control drawn from the same collection at the same
visual similarity to the query, within ±0.02, sharing no notation with it. The draw is balanced
above and below the target, and repeated over five seeds. Visual similarity then carries no
information about which candidate is the positive, and its AUC sits at 0.500-0.502 in every
draw, which is the check that the design works.

| corpus · encoder | volume size alone | visual | graph | linear fusion | graph, cross-volume only | cross-volume reachable |
| --- | ---: | ---: | ---: | ---: | ---: | :--: |
| Iconclass · DINOv2 | 0.578 | 0.502 | 0.677 | **0.693** | 0.649 | **no** |
| Iconclass · SigLIP | 0.551 | 0.500 | 0.659 | 0.642 | 0.639 | **no** |
| Iconclass · CLIP | 0.444 | 0.502 | 0.488 | 0.477 | 0.593 | **no** |
| Emblematica · DINOv2 | 0.588 | 0.502 | 0.543 | 0.519 | **0.505** | yes |
| Emblematica · DINOv2 + mottoes | 0.668 | 0.500 | 0.422 | 0.398 | 0.393 | — |

Read without the last column, the Iconclass rows say H2 holds: fusion reaches 0.693
[0.632, 0.749] across a 0.680-0.706 spread over five control draws, and keeps 0.653 when
restricted to pairs whose two items sit in different volumes. Two controls say otherwise.

### Control 1 — volume size

A PPMI-SVD embedding places nodes partly by degree, and hard positives drawn across volumes come
from volumes of far more unequal size than random same-collection pairs do: median size ratio 4.0
against 1.9. Scoring the pairs on volume-size proximity alone reaches 0.578 on Iconclass and
0.588 on Emblematica, so this is a genuine confound in the aggregate figures. It is **not** the
explanation of the cross-volume result, where it falls to 0.475.

### Control 2 — can the graph reach the pair at all?

`G1` carries `part_of` and `adjacent_to` and nothing else, and both are within-volume relations.
The projection is therefore a **disjoint union of 1 246 components, one per volume, with no path
of any length between items of different volumes**. The graph holds no information about such a
pair, so the 0.649 it scores on them is not information: it is PPMI-SVD placing structurally
similar but unconnected components near one another. Embedding a disconnected graph produces
coordinates for every component, and nothing forbids two islands from landing close together.

Emblematica's `pure` projection links volumes through shared creator, place and date nodes, so
there the question is meaningful — every volume lies in one component. The answer there is
**0.505**, chance to three decimals.

**Where the graph can carry cross-volume information it carries none; where it appears to, it
structurally cannot.** Publication gate 2 is unsupported on both corpora, and the reachability
check is now written into every hard-pair artifact so the distinction cannot be lost again.

H3 remains wrong-signed throughout, for the separate reason given under *What the ground truth
can and cannot express*: Iconclass cannot express it.

### What the graph is good at, stated positively

The same graph separates **same-author from different-author volume couples at AUC 0.963** under
the `pure` projection, measured on 589 couples of each kind:

| graph variant | same-author score | other | AUC |
| --- | ---: | ---: | ---: |
| `book` (volume membership; the graph that failed) | +0.0752 | +0.0752 | 0.457 |
| `full` | +0.2132 | +0.0830 | 0.620 |
| `pure` (projected attributes) | +0.3461 | +0.0168 | **0.963** |

So the graph is an excellent bibliographic instrument and a null iconographic one. The
aggregate fusion gain is real because plates of one volume, and volumes of one author, do share
iconography often enough to move nDCG. It is not evidence that the graph recognises a motif.

### The two superseded readings, retained

**First reading — "H2 firmly refuted, because hard positives are bibliographically unrelated."**
Computed on pairs mined without a format control. Holding libraries scan differently:
Wolfenbüttel serves full pages with a colour chart and ruler in frame at median aspect ratio
1.52, Glasgow and Illinois serve cropped picturae at 0.81 and 0.98. "Visually distant" collapsed
into "different holding library" — 247 of 250 CLIP-mined hard positives crossed collections
against 0 of 250 hard negatives, and 243 against 0 on Iconclass. Pairs drawn from different
libraries are necessarily from different books and authors, so their bibliographic unrelatedness
followed from the sampling. The reasoning was circular and is withdrawn.

**Second reading — "weakly above chance in three conditions of four."** Computed with controls
drawn uniformly inside the caliper. Hard positives sit in the low tail of the similarity
distribution, so a symmetric window offers far more candidates above the target than below, and
the control came out reliably *more* similar to the query than the positive was. The visual arm,
which the design requires at chance, read 0.395-0.468. Balancing the draw fixes it, and raises
the Iconclass figures — which is what made the two controls above necessary.

## What each modality actually sees

Mean overlap of the top 10, on the Emblematica test split, every table restricted to the same
candidate pool. The reference figure is the overlap between two visual encoders: that is how
much two honest views of the same evidence agree.

| pair | overlap@10 |
| --- | ---: |
| **fused V+G ↔ its own visual arm** | **0.811** |
| visual DINOv2 ↔ visual CLIP | 0.228 |
| fused V+G ↔ visual CLIP | 0.261 |
| fused V+G ↔ graph | 0.121 |
| graph ↔ visual CLIP | 0.100 |
| graph ↔ visual DINOv2 | 0.081 |
| fused V+G ↔ text | 0.048 |
| text ↔ visual CLIP | 0.043 |
| text ↔ visual DINOv2 | 0.042 |
| graph ↔ text | 0.042 |

Two things follow. **The fused system is four fifths its visual arm** — which agrees with the
rank-change analysis, measured by a different route, that 1.89 of 10 results change on
Emblematica. And **the three modalities are close to mutually independent**: the graph shares a
twelfth of its neighbourhood with a visual encoder, and the text a twenty-fourth with either.

This refines the V+G+T result rather than contradicting it. The text is *not* redundant in what
it sees; it is redundant in what it contributes **to this relevance definition** once the graph
is present. Mottoes bring a genuinely different neighbourhood that graded Iconclass relevance
does not reward.

## Cross-volume retrieval, and what can be claimed from it

Of the 28 040 top-10 slots the fused system fills on the Emblematica test corpus, **346 point
outside the query's own volume** — 1.2%. Each is classified by the evidence joining the two
volumes, using the attributes the graph itself carries.

| class | n | share | what it is worth |
| --- | ---: | ---: | --- |
| `réseau` — shared place or decade | 124 | 35.8% | restates a catalogue fact |
| `inexpliqué` — nothing shared at all | 118 | 34.1% | a failure, or a resemblance the annotation omits |
| `atelier` — shared creator | 87 | 25.1% | restates a catalogue fact |
| **`circulation`** — no shared attribute, shared notation | **17** | **4.9%** | **can suggest a hypothesis** |

Only the last class is doing work no catalogue does. Seventeen retrievals across the whole test
corpus join two volumes that share no creator, place or decade, through an Iconclass concept.
The clearest sets a German emblem book against a French one on `46C215` (anchor) and `54D5`:
*Sine his periculum* beside *Prince procurant la saulveté de ses subjectz*. That is a hypothesis
a historian could take up, not a finding, and the distinction is the deliverable.

## Concept alignment to an external vocabulary

The Iconclass SKOS export shipped with the benchmark carries only `notation`, `broader` and
`narrower` — no `exactMatch`, no `sameAs` — so an alignment has to come from outside. Wikidata's
property P1256 holds an Iconclass notation, and one SPARQL query is the whole mapping: **4 125
distinct notations**.

| corpus | assignments covered | items with at least one mapped notation |
| --- | ---: | ---: |
| Iconclass AI | 37.5% | 68.3% |
| Emblematica | 28.2% | 83.0% |

Enough to name a concept in readable words for the demonstrator and for error analysis; nowhere
near enough to serve as ground truth or to re-derive relevance.

## Phase 6 — the learned model loses to the transparent one

Projection heads over frozen encoders, trained with symmetric InfoNCE and in-batch negatives on
the 11 628 Iconclass images carrying both modalities, three seeds by two capacities, checkpoint
selected on validation loss. Test metrics against the transparent baselines, paired over the
same queries:

| method | nDCG@10 | MAP | vs late fusion, nDCG | vs late fusion, MAP |
| --- | ---: | ---: | --- | --- |
| visual only | 0.5639 | 0.2220 | — | — |
| **late fusion** | **0.6026** | 0.2411 | — | — |
| learned, linear heads | 0.5975 | **0.2643** | −0.0051, p = 0.0010, d = −0.081 | **+0.0231, p = 0.0001, d = 0.259** |
| learned, MLP heads | 0.5892 | 0.2496 | −0.0134, p = 0.0001, d = −0.177 | +0.0085, p = 0.0001, d = 0.114 |

The learned space **trades graded relevance for exact-label precision**: it loses the primary
endpoint by a negligible margin and wins MAP by a small-to-moderate one. Across three seeds the
spread is 0.0006 of nDCG, so this is not seed noise. The capacity ablation runs the wrong way —
the MLP is worse than the linear head on both metrics — and the collapse diagnostics say why:
effective rank falls from 20.5 of 128 dimensions for the linear head to 15.3 for the MLP.

**Training never generalises past the first epoch.** Train loss falls monotonically from 4.57 to
1.94 while validation loss rises monotonically from 7.79, so early stopping selects epoch 1 in
all six runs. The objective asks a projection to match an item's visual vector to its own graph
vector, and that graph vector is essentially the item's *volume identity*. The split is
group-aware, so no validation volume appears in training, and predicting an unseen volume from
pixels is exactly the thing that cannot transfer. The model is behaving correctly; the task is
ill-posed for this graph.

**On hard positives it reproduces the artifact rather than escaping it.** The aligned space
scores 0.701 against matched controls and 0.653 on cross-volume pairs — but it is a function of
`G1`, which holds no path between volumes, so the cross-volume figure is the same embedding
artifact documented above, inherited. There is no interpretable hard-case behaviour to report.

**Verdict against the phase-6 exit criterion.** The criterion asks the learned model to beat the
strongest simple fusion on at least one central preregistered metric *and* retain interpretable
behaviour on hard pairs. The first half is met on MAP; the second is not. The roadmap's stop
condition therefore applies: **the transparent method is preferred**, and the MAP result is
recorded as a genuine but narrow advantage rather than a reason to build further.

## Two annotation artifacts found along the way

Both had the same signature — one label shared by thousands of unrelated objects, silently
inflating relevance — and both were caught by inspecting data rather than by a failing test.

**Bracketed Iconclass keys.** `normalize_notation` stripped all parenthesised keys, so notation
86 (proverbs, emblems, mottoes) collapsed 5 972 distinct mottoes onto one node and any two
emblems scored graded relevance 1.0. The hard-pair benchmark was entirely compromised: **1 of
250** mined hard positives shared an exact notation with its partner, against **186 of 250**
after the fix. Protocol v0.5 keeps both definitions and requires every nDCG claim to be reported
under each.

**A cataloguer's template read as data.** Emblematica records awaiting indexing carry a
commented-out XML template containing `<skos:notation>[notation]</skos:notation>`. Parsed
without stripping comments, it gave every unindexed emblem the same fake label. Corrected
coverage is 82.0%, not the 90.5% first measured.

## What the ground truth can and cannot express

H2 and H3 are not symmetric hypotheses under Iconclass, and the asymmetry is a property of the
vocabulary rather than of the method.

Iconclass is predominantly **denotative**: it records what is depicted. Division 5, *Abstract
Ideas and Concepts*, supplies the connotative layer and accounts for 17.1% of assignments in the
Emblematica corpus, so two emblems that look nothing alike can genuinely meet on a shared
concept. Mined hard positives do exactly that — `56F2` (personifications of Love), `57B1`
(Praise, Approval), `54E1` (Cooperation) recur among the pairs, joining a crowned cipher to a
sermon on divine love, or a county's coat of arms to an unrelated device.

**H2 is therefore testable.** 193 of 250 mined hard positives share an exact notation with their
partner.

**H3 is not, and never was.** It asks whether the graph can push apart images that look alike but
mean different things — a lion standing for fortitude beside a lion standing for defeat. Both
carry `25F23`, predatory animals, so Iconclass scores them as semantically *close*, and the
mining rule for a hard negative (semantic similarity ≤ 0.2) can never select them. What the miner
returns instead is pairs that share a **page layout**: 1 of 140 hard negatives shares any
notation at all, and a representative pair sets `86(ZORNIGER NARR)` with `31D14`, `41D28` against
`11U1`, `11U21` at visual similarity 0.84 — an angry fool and a religious subject, alike only in
being scanned book pages.

So the consistently wrong-signed H3 figures reported above measure the graph's relation to
scanning format, not to iconographic contradiction. Testing H3 as stated would require an
annotation that separates a motif from its allegorical reading.

**The emblem's own interpretive verse was the obvious candidate, and it is not available.** The
subscriptio states what the picture means, which is precisely the layer Iconclass omits. Counting
it across all 31 041 SPINE records (`scripts/audit_emblem_text.py`):

| field | records carrying it | median length |
| --- | ---: | ---: |
| motto | 30 402 | 38 characters |
| pictura description | 5 178 | 60 characters |
| **subscriptio** | **455** | 99 characters |
| Iconclass annotation | 25 463 | — |
| **subscriptio *and* Iconclass on the same record** | **1** | — |

454 of the 455 transcribed subscriptiones carry no Iconclass annotation at all — 422 of them from
Wolfenbüttel, 32 from Illinois. The verse and the ground truth therefore cannot be joined: there
is one record on which both exist. Whatever the literary value of those 455 transcriptions, they
cannot supply graded relevance for a hard-negative test, so this route is closed by the corpus
rather than by the method. **Expert judgement (phase 10) is the only remaining route to H3.**

Mottoes do not substitute. They are short, formulaic, and at most 11 of 250 hard-positive pairs
share a single content word.

## Limits on these results

- The Iconclass graph conditions are evaluated on the 1 714-image sub-corpus where `G1` exists
  (19.3% of that test split), not corpus-wide.
- Emblematica evaluation excludes 80 test emblems whose picturae return HTTP 404 — catalogued
  but not served, 79 from one Glasgow volume. Recorded in `exclusions.json`.
- Emblematica creator matching is by surname, not authority identifier, and will merge
  distinct people who share one.
- Recall@K is reported in the artifacts but is not informative on either benchmark: the median
  Iconclass query has 176 exactly relevant candidates, so Recall@10 is bounded near 0.25 by
  construction. Precision-oriented metrics are the ones to read.
- The hard-pair benchmark is mined with thresholds calibrated on one encoder, so pair classes
  are encoder-relative by construction; a visual AUC near 0 or 1 on such pairs is a tautology,
  not a baseline. The matched control above exists to make the comparison fair, and the residual
  visual AUC of 0.34-0.49 shows the matching is good but not exact.
- Phase 6 (learned joint alignment) was not run on real corpora. Its inputs would be the same
  graph shown here to hold no path at all between the items of a cross-volume hard positive, so
  no projection head can recover what is not present.
- The text modality is evaluated on Emblematica only. Iconclass AI carries no transcribed text.
- `G1` results rest on an embedding of a disconnected graph. Cross-volume similarities it
  produces are artifacts of the method and should not be read as weak evidence of anything; the
  `graph_reachability` block of each hard-pair artifact records this.
