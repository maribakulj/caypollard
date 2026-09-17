# The visual record: an intermediate language for cross-medium iconography

A structured, inspectable description of a picture, designed so that a printed emblem and a
painted panel that share a motif can be compared, and so that the support they arrived on cannot
be written down.

Every design choice below is a measurement, not a preference. The referee throughout is the
cross-medium test: 2 723 cropped emblem prints pooled with 1 864 museum objects, 4 587 items
joined by shared Iconclass concepts, asking two things of a representation — can a probe still
name which corpus an object came from, and where does an object's best cross-medium
iconographic partner rank.

## Why pixels will not do

| representation | medium probe | median rank of partner | neighbours from the other corpus |
| --- | ---: | ---: | ---: |
| DINOv2 on the image | 97.4% | 821 of 4 587 | 2.6% |
| DINOv2 on a Sobel rendering | 95.1% | — | — |
| DINOv2 on a coarse shape map | 93.9% | 656 | 6.5% |
| DINOv2 on a binarised silhouette | 87.5% | 419 | 13.5% |
| **shape vocabulary, 512 signs** | **74.4%** | **176** | **28.5%** |
| human Iconclass transcription | 91.3% | 92 | 7.6% |
| *majority baseline* | *59.4%* | | |

A visual encoder names the corpus 97.4% of the time and places the partner at the middle of the
pool. It has learned the material, not the motif. Re-rendering does not fix it and can make it
worse: a Sobel map preserves stroke structure, and stroke structure *is* the support — an
engraving is made of lines and a painting is not. What helps is destroying line structure
(silhouette) and then naming what survives.

The last row matters as a target. Driving the probe to 59.4% would mean having destroyed real
iconographic differences between holdings, because Wolfenbüttel's German devotional emblems and
the Rijksmuseum's paintings genuinely depict different things. The floor is content, not chance,
and the shape vocabulary already sits below the human transcription on this measure.

## The three channels

### 1. Shape — the lexicon

Regions are segmented by a watershed over a smoothed gradient, classical rather than learned: a
learned segmenter is trained on photographs and inherits their statistics, which is the confound
being escaped. 41 804 regions over 4 587 images, median 10 per emblem and 6 per museum object.

Each region is described only by properties that survive a change of support — area, elongation,
solidity, four log moment invariants, and tone reduced to *darker or lighter than the page*,
which is the only photometric fact a monochrome scan preserves. No colour, no texture, no
resolution. Signs are then **found by clustering** those descriptors on a sample balanced across
corpora, so that the larger corpus cannot define the vocabulary and leave the smaller one
described in a foreign language.

Vocabulary size is a free parameter with respect to the confound and not with respect to the
motif:

| signs | medium probe | median rank | neighbours from the other corpus |
| ---: | ---: | ---: | ---: |
| 64 | 74.7% | 248 | 27.9% |
| 128 | 74.1% | 237 | 28.8% |
| 256 | 75.3% | 196 | 28.1% |
| 512 | 74.4% | **176** | 28.5% |
| 1 024 | 74.5% | 175 | 28.4% |

Enlarging the lexicon costs nothing in leakage and buys precision until about 512, where it
saturates.

**18 of 256 signs are used more than 90% by one corpus.** Those are signs for a support rather
than a motif, and the count is the vocabulary's honesty check; it should be reported whenever the
vocabulary is rebuilt.

### 2. Relations — the grammar

Seven coarse relations between regions, computed from geometry the segmenter already records:
*above, below, left, right, contains, inside, touches*. Each feature is an ordered pair of signs
plus the relation. A print and a painting of one subject will not agree on exact positions; they
will agree that one element sits above another.

At 64 signs, adding relations improves every column — probe 74.7% to 73.4%, median rank 248 to
222, top-10 share 5.3% to 6.2%, cross-medium neighbours 27.9% to 29.8%.

**A grammar needs a small lexicon.** At 256 signs the pair space holds 459 000 features, only 86
of which occur thirty times or more, and 3 975 of 4 587 images end up with no surviving feature.
At 64 signs, 4 014 features survive and 143 images are emptied. This is Neurath's reduction
argument arriving from the other direction: signs must repeat before their combinations can.

### 3. Palette — recorded, never matched on

Eleven basic colour terms assigned in a perceptual space after grey-world balancing.

This channel exists to be consulted and must carry **weight zero in matching across media**,
which is a measurement rather than a caution. Within one collection, one format and seventeen
books — so that only the scanning session and the artwork vary — a raw colour histogram names the
volume 61.3% of the time against a 31.0% baseline. Canonical terms bring that to 53.2% and
grey-world balancing to 39.7%: 71% of the excess removed, and a third of it still there. Seven
per cent of digitisations in both corpora are effectively monochrome, so any weighting of colour
must also degrade gracefully to its absence.

Giving it weight confirms the prediction directly:

| record | medium probe | median rank | neighbours from the other corpus |
| --- | ---: | ---: | ---: |
| shape alone | 74.4% | 176 | 28.5% |
| shape + relations | 75.0% | 180 | 28.4% |
| **shape + palette** | **80.0%** | **235** | **22.3%** |
| all three | 80.1% | 230 | 22.5% |

Colour degrades every column. Separate channels are what make that expressible: a single fused
vector could not keep the information without also suffering it.

## What the record does and does not do

It **crosses the medium**. 28.5% of a shape record's ten nearest neighbours come from the other
corpus, against 2.6% for a visual encoder, where the other corpus is 41% of the pool. The
representation is approaching not seeing the difference between a print and a painting.

It finds **formal similarity, not iconographic identity**. Median rank 176 against the human
transcription's 92 says the record reaches many cross-medium neighbours and does not yet know
which is right. That is what a shape channel with no grounding in meaning should do, and it marks
where the work continues: the signs are found by clustering and named by number. Attaching them
to a vocabulary a historian uses — Iconclass primitives, or a reduced set of them — is the step
that would turn formal recurrence into iconographic evidence.

And it will not interpret. Impett and Süsstrunk, clustering Warburg's Bilderatlas on relative
limb angles, recovered pose clusters corresponding to Pathosformeln *and* found that
morphologically similar poses can represent wildly different emotions. A transcription language
that captures form will find form recurring. What the recurrence means is not in the record, and
should not be claimed from it.

## Reproducing

```
scripts/segment_shapes.py            regions from images
scripts/build_shape_vocabulary.py    signs by clustering, with the corpus-skew check
scripts/build_shape_relations.py     the seven relations over sign pairs
scripts/build_visual_record.py       weighted concatenation of channels
scripts/test_cross_medium.py         the referee
scripts/probe_confound.py            the probe, linear and kNN
```
