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

**H1 holds, replicated on two independent corpora. The transparent fusion gain replicates in
all six encoder-corpus combinations. H2 is not supported: once the hard-pair benchmark is
controlled for digitisation format, the graph sits weakly above chance in three conditions of
four and below it in the fourth. H3 is consistently wrong-signed.**

A first version of this document reported H2 as firmly refuted with a mechanism. That was
computed on confounded pairs and has been corrected in place — see *H2 and H3* below. The
correction is retained rather than deleted so the error stays legible.

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

## H2 and H3 — not supported, after one confound was removed

Three explanations were eliminated in turn.

**Not the corpus.** Emblematica supplies creator, place, date and work for 99.6% of volumes.

**Not the graph construction.** The first graph built on Emblematica reproduced the earlier
failure — 100% of an emblem's top-5 neighbours came from its own volume — because within-book
edges outnumbered bibliographic ones 29 to 1 and book nodes are hubs of ~96 emblems. Projecting
book attributes onto emblems and dropping within-book edges fixes it: measured on 589
same-author against 589 different-author volume couples,

| graph variant | same-author score | other | AUC |
| --- | ---: | ---: | ---: |
| `book` (volume membership; the graph that failed) | +0.0752 | +0.0752 | **0.457** |
| `full` | +0.2132 | +0.0830 | 0.620 |
| `pure` (projected attributes) | +0.3461 | +0.0168 | **0.963** |

**Correction — this section was wrong, and the corrected result is different.**

The paragraph that follows was written on hard pairs mined without a format control, and that
invalidated it. Holding libraries scan differently: Wolfenbüttel serves full pages with a colour
chart and ruler in frame (median aspect ratio 1.52), Glasgow and Illinois serve cropped picturae
(0.81 and 0.98). "Visually distant" therefore collapsed into "different holding library" — of
250 hard positives mined with CLIP on Emblematica, **247 crossed collections while 0 of 250 hard
negatives did**; on Iconclass AI the same pattern held at 243 and 0. Concluding from those pairs
that hard positives share no bibliographic attribute was circular: pairs drawn from different
libraries are necessarily from different books, authors and places.

Re-mined with `--stratum-key collection`, so both items of every pair come from one library and
one scanning format, and compared against a control matched on visual similarity (±0.02, same
collection, no shared notation) so that the two classes are equally distant to the eye:

| corpus | encoder | n | graph AUC on H2 | 95% CI | reading |
| --- | --- | ---: | ---: | --- | --- |
| Emblematica | DINOv2 | 175 | 0.449 | [0.387, 0.510] | inconclusive |
| Emblematica | CLIP | 140 | 0.633 | [0.567, 0.697] | above chance |
| Iconclass AI | DINOv2 | 250 | 0.691 | [0.646, 0.736] | above chance |
| Iconclass AI | CLIP | 249 | 0.582 | [0.532, 0.633] | above chance |

The matched control brings the *visual* AUC to 0.34-0.49, near chance, which is what makes the
comparison fair; before matching it ranged from 0.00 to 0.79 depending on the direction of the
imbalance.

**The corrected finding is weaker than a refutation and weaker than support.** Three of four
conditions put the graph above chance on hard positives, by 0.08 to 0.19 of AUC; the fourth is
inconclusive and sits below 0.5. H3 remains consistently wrong-signed (0.10-0.43): the graph
rewards hard negatives, which are disproportionately same-volume pairs. So the graph carries a
weak and encoder-dependent signal on hard positives, not the null previously reported, and
nothing that supports publication gate 2.

**The superseded reasoning follows, retained so the error is legible.**

**The actual reason: hard positives are bibliographically unrelated.** Of 250 hard positives
mined with DINOv2, **243 share no bibliographic attribute at all** and 2 share an author; of 250
mined with CLIP, **none share an author**, and the 66 sharing anything share only a *decade*.
That weak attribute is what produces CLIP's apparent H2 AUC of 0.759 against DINOv2's 0.521 —
chronological coincidence, not iconographic knowledge. Hard *negatives* meanwhile do share
attributes (36 and 20 shared authors), pushing the score the wrong way.

Mottoes do not bridge it either: at most 11 of 250 hard-positive pairs share one content word.

**Interpretation.** Two emblems depicting the same subject in visually unlike ways are
systematically drawn from unrelated books — different authors, places and centuries.
Iconographic recurrence does not follow bibliographic lineage. This is what emblem scholarship
describes as the circulation of motifs, and it means a context-only graph is structurally
incapable of recovering hard positives however well it is constructed. Publication gate 2 is
unsupported on two independent corpora.

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
  graph shown here to hold no edge between hard positives.
