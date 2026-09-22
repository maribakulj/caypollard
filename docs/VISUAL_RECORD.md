# The visual record: an intermediate language for cross-medium iconography

A structured, inspectable description of a picture, designed so that objects made and
photographed in different ways can be compared iconographically, and so that the support they
arrived on cannot be written down.

Every design choice here is a measurement. The corpus is **21 128 objects** — 2 723 cropped
emblem prints and 18 405 museum works from eight institutions, reached through Wikidata and
Wikimedia Commons without an API key — with zero group and zero checksum leakage and a retrieval
pool of 18 840. Figures from earlier, narrower corpora are superseded and live in the commit
history; several of the conclusions drawn from them were wrong, and the error was always the
same shape: a base too narrow for the claim made on it.

## The finding that organises everything else

**There is no single best record.** Each question wants a different channel, and mixing the
channels is right for some questions and actively wrong for others.

| question | best configuration | median rank | pool |
| --- | --- | ---: | ---: |
| cross-corpus partner (prints ↔ museum works) | shapes + 2× composition + ½ repetition | 820 | 18 840 |
| cross-object-kind partner (flat ↔ volume ↔ vessel) | **shapes alone** | 1 008 | 12 944 |
| a gesture (division 3) | **pose alone** | 65 | 1 729 |
| a scene (division 1) | **composition** | 1 429 | 12 479 |
| an object (division 2) | shapes, with or without the rest | 655 | 18 840 |

Two of these were surprises that reversed an earlier design. Adding pose to the record degrades
nearly every division, and adding composition to cross-object-kind retrieval nearly doubles the
rank. The channels are **a set to select from, not a sum to compute**, and a system built on this
record should choose its configuration from the question rather than carry one vector.

## Why pixels will not do

A DINOv2 embedding names which corpus an object came from 97.4% of the time and places an
object's best cross-medium partner in the middle of the pool. It has learned the material, not
the motif. Re-rendering does not fix it and can make it worse: a Sobel map preserves stroke
structure, and stroke structure *is* the support — an engraving is made of lines and a painting
is not. What helps is destroying line structure and then naming what survives.

| representation | medium probe | cross-corpus neighbours |
| --- | ---: | ---: |
| DINOv2 on the image | 97.4% | 2.6% |
| DINOv2 on a Sobel rendering | 95.1% | — |
| DINOv2 on a binarised silhouette | 87.5% | 13.5% |
| **the record, final corpus** | **85.2%** | **21.8%** |
| *majority baseline, final corpus* | *86.1%* | *14.5% proportional* |

The probe target is the majority baseline, not chance. Driving it to chance would mean having
destroyed real iconographic differences between holdings, which are content and not support.

## The channels

### Shape — the lexicon

Regions are segmented by a watershed over a smoothed gradient, classical rather than learned: a
learned segmenter is trained on photographs and inherits their statistics, which is the confound
being escaped. Each region is described only by properties that survive a change of support —
area, elongation, solidity, enclosed holes, scale against the picture's own median part, four log
moment invariants, a twenty-four bin radial contour signature, and tone reduced to *darker or
lighter than the page*, the only photometric fact a monochrome scan preserves. No colour, no
texture, no resolution.

Signs are found by clustering those descriptors on the same number of regions drawn from each
corpus, so the larger one cannot define the vocabulary. Regions are then grouped into composite
signs by transitive closure of a proximity test that is relative rather than absolute — absolute
box overlap is useless, since a picture's dark and light regions interpenetrate and the closure
swallows the whole plate.

**This is the object channel.** It reaches division 2 at rank 655 and religious scenes at 10 370,
which is chance.

### Composition — the scene channel

A coarse grid of ink density taken as departure from the page median in either direction, so a
light figure on a dark ground and a dark one on light paper read alike; left-right and top-bottom
mirror agreement; the vertical and horizontal mass profiles; and the spread, elongation and count
of the region centroid cloud. Nothing local.

**This is the scene channel**, and it reverses the shape channel exactly: division 1 from 6 403
to 1 429, Bible scenes from 3 532 to 835, mythology from 1 943 to 863. A cat has a distinctive
silhouette; an Annunciation and a Nativity share figures and architecture whose local shape
statistics are generic, and what two pictures of one episode share is where things are.

It is also the channel that **fails on vessels**, at 2 036 against the shape channel's 1 008,
because a vessel's form dictates its own layout: the decoration of a vase is arranged by the vase
and not by the subject.

### Repetition — multiplicity as a typed quantity

An Isotype says three by drawing the sign three times, and a histogram records that as a
magnitude, so one figure and five become nearby points on one axis instead of two arrangements.
The channel types it twice: a profile of how many signs occur once, twice, a few times or many,
and a per-sign multiplicity, because three columns and three lions are different pictures.

It is the strongest single channel on objects and **hurts religious scenes at every weight
tried**. An episode does not fix its count, so counting misleads exactly where the subject leaves
the number free.

### Pose — relative limb angles, after Impett

Angles taken against the figure's own torso axis, so a leaning figure is described by what its
limbs do relative to its body rather than to the picture's edge, with a presence flag for each
absent limb rather than a zero angle that would read as a direction.

The stop condition was measured before the channel was trusted: a keypoint detector trained on
photographs would make this a medium detector if it read paintings and not engravings. It finds
a usable figure in 28.2% of engravings and 29.3% of museum works — evenhanded, and merely
conservative on art. **Coverage of 28% is the real limit**: the channel speaks for a minority of
queries however good it is on them.

On the 1 729 items carrying both a pose and a record, it reaches division 3 at rank 65 against
the record's 114, with a *lower* medium probe, 60.6% against 62.9%. It wins wherever there are
human figures and loses on religious scenes.

### Palette — recorded, never matched on

Eleven basic colour terms assigned in a perceptual space after grey-world balancing. This channel
must carry **weight zero in matching**, which is a measurement and not a caution. Within one
collection, one format and seventeen books — so that only the scanning session and the artwork
vary — a raw colour histogram names the volume 61.3% of the time against a 31.0% baseline;
canonical terms bring that to 53.2% and grey-world balancing to 39.7%. Seven per cent of
digitisations are effectively monochrome, so any use of colour must degrade gracefully to its
absence.

Giving it weight degrades every column: the medium probe rises from 74.4% to 80.0%, the partner's
rank worsens from 176 to 235, and cross-medium neighbours fall from 28.5% to 22.3%.

## Does a vase find an engraving?

Yes, and weakly. Object kinds are grouped into families that share a viewing condition rather
than a technique — a flat surface seen frontally, a thing in the round lit and shadowed, a curved
vessel whose own form hides part of its decoration — and a pair counts when two objects share a
non-hub notation across families.

The shape channel places such a partner at **median rank 1 008 of 12 944 against a chance of
6 472**, with 8.6% of queries reaching one in their top hundred and 14.1% of neighbours drawn
from another family. Six times better than chance and far out of reach.

For scale, crossing corpora reaches 820. **Crossing a viewing condition is harder than crossing a
workshop**, which is what this benchmark called cross-medium throughout.

## Division 5 is a cataloguing asymmetry, not a limit of the method

The emblems carry 3 986 abstract assignments over 614 distinct notations. The 18 405 museum works
carry **one distinct notation**, `56DD1`, fear. There is nothing to match against, and a figure
that appears to improve as the corpus grows is that one concept and not abstraction.

The Wikidata bridge is not the blocker, which had to be checked before blaming cataloguing
practice: the P1256 alignment carries 166 abstract notations including existence, similarity and
ambivalence. Wikidata editors record what a picture literally shows — a painting of a woman is
tagged woman, never Patience — while Iconclass specialists annotating emblems record what it
means, because that is what an emblem is.

Inside the emblem corpus alone, with partners required from a different book so that binding
cannot supply the answer, an abstract-concept partner sits at **median rank 117 of 2 629 against
a chance of 1 314** — eleven times better than chance, and better than religious scenes at 226.
The record finds two pictures sharing an abstract idea perfectly well where the annotation
exists.

## Four routes to a name, and what each cost

**Clustering the patches.** A sign's associated notations share 0.075 of their resolved ancestry
against 0.037 for random sets from the same pool, p = 0.0001, d = 0.61. Significantly related,
and a thirteenth of an ancestry apart. A sign is a tendency, not a name.

**Grouping the patches.** Grouping took the medium probe from 75.3% to 61.4% and left nameability
untouched: composites sit at 0.064 against 0.054, p = 0.30. A clump of mid-sized dark patches is
drapery in one picture and foliage in another. Geometry joins them; meaning does not.

**Supervising at picture level.** This works, at 2.3 times the prior on emblems, and only once
the shared vocabulary is wide enough to measure it. It was declared a failure on twelve
notations.

**Supervising at region level.** Fails under eight framings. The pooled control is what makes
the failure interpretable — it beats the narrowing on museums, so a picture's notation is better
predicted by all its regions together than by the one a narrowing procedure elects.
**Iconographic identity is not localised in a single shape**; it lives in the distribution of
shapes across a picture.

That reading suggested a repair: describe each region by its neighbourhood — how many parts sit
above, below, beside it, whether anything encloses it or it encloses anything — which moves the
relation channel down from the picture to the region. It was run both ways so the contribution of
context would be measured rather than assumed, which is the mistake this project has made more
than once by crediting a gain to whatever was added last.

| 30 notations, hits@1 | narrowing | pooled | + context | prior | record |
|---|---|---|---|---|---|
| emblems (783 pictures) | 0.087 | 0.061 | 0.078 | 0.257 | **0.324** |
| museums (1 464 pictures) | 0.066 | 0.093 | 0.103 | 0.147 | **0.531** |

Context is worth about a point to the pooled model (+1.7 on emblems, +1.0 on museums) and costs
the narrowing about one; neither approaches the prior. The repair fails.

The last column is what turns a failure into a result. It is the same notations, the same
pictures, the same split and the same classifier, with the picture described by its whole-image
record instead of by its regions — and it clears the prior on both corpora, by 6.8 points on the
emblems and by a factor of four on the museums. So the protocol is not asking an unanswerable
question, and the region result is not an artefact of the evaluation.

Running that control across the medium-invariant renderings, rather than only on the plain
record, shows that the failure has nothing to do with regions.

| emblems, 30 notations, 783 pictures | hits@1 | 95% interval | paired gain over random |
|---|---|---|---|
| record, plain | 0.324 | [0.292, 0.356] | +0.230 [+0.193, +0.268] |
| record, greyscale and squared | 0.321 | [0.287, 0.354] | +0.226 [+0.189, +0.262] |
| record, library direction projected out | 0.292 | [0.261, 0.324] | +0.198 [+0.161, +0.235] |
| *constant predictor (prior)* | *0.257* | — | — |
| record, Sobel edges | 0.211 | [0.183, 0.240] | +0.116 [+0.082, +0.151] |
| record, coarse-grid shape | 0.161 | [0.135, 0.186] | +0.066 [+0.034, +0.098] |
| picture redrawn from its signs | 0.130 | — | +0.036 [+0.005, +0.065] |
| record, ink mass | 0.121 | — | +0.027 [−0.003, +0.057] |
| record, silhouette | 0.101 | [0.080, 0.123] | **+0.006 [−0.023, +0.036]** |
| *holding collection alone, one-hot* | *0.089* | — | *−0.005 [−0.033, +0.023]* |
| *same classifier on random vectors* | *0.095* | *[0.073, 0.116]* | — |
| regions, best of eight framings | 0.087 | — | — |

Two floors are needed to read this, and they are not the same floor. The constant predictor is
what a representation must beat to be *useful*; random vectors under the identical one-vs-rest
fit are what it must beat to contain *anything*. Intervals are 2 000 bootstrap resamples of the
783 test pictures, and the last column resamples the *paired* per-picture difference, which is
the comparison that matters when two rows are scored on the same pictures.

On the emblems the ladder reads in one line: the more of the support a rendering destroys, the
less iconography survives, and the silhouette — the abstract black-and-white drawing this record
was asked for — is the one row whose gain over random vectors straddles zero. The holding
collection alone, given to the same classifier as a one-hot, is also at the floor, so nothing in
this table is provenance.

That looked like an answer about shape. It is not, and the museums corpus is what says so. The
same renderings, the same classifier, the same two floors, on 869 paintings, sculptures,
vessels and garments:

| museums, 30 notations, 869 objects | hits@1 | paired gain over random |
|---|---|---|
| record, plain | 0.530 | +0.497 [+0.461, +0.532] |
| record, greyscale and squared | 0.530 | +0.497 [+0.463, +0.534] |
| record, Sobel edges | 0.443 | +0.410 [+0.374, +0.445] |
| record, coarse-grid shape | 0.367 | +0.334 [+0.299, +0.367] |
| **record, silhouette** | **0.298** | **+0.265 [+0.232, +0.297]** |
| **record, ink mass** | **0.251** | **+0.217 [+0.186, +0.249]** |
| *holding collection alone, one-hot* | *0.182* | *+0.148 [+0.121, +0.177]* |
| **picture redrawn from its signs** | **0.146** | **+0.113 [+0.086, +0.139]** |
| *constant predictor (prior)* | *0.135* | — |
| regions, pooled | 0.102 | — |
| regions, narrowing | 0.057 | — |
| *same classifier on random vectors* | *0.033* | — |

**It does not replicate, and it fails in the direction that overturns the emblems reading.** On
museum objects the silhouette is not at the floor: it more than doubles the prior, and it beats
the holding collection by +0.116 [+0.078, +0.154], so it is not provenance either. A black-and-
white shape, with every trace of tone, texture and support thrown away, recovers half of what the
plain record recovers over noise.

Looking at the two renderings suggested the emblems row was disqualified. The silhouette mode is
a median threshold of a coarse-gridded greyscale, not a figure-ground cut: on a museum photograph,
whose ground is a uniform grey, it approximates the object's outline and the garment is plainly a
garment; on an engraving it thresholds hatching, and a Glasgow test pictura comes out as noise.
The same code produces a representation on one corpus and static on the other.

So a rendering was built to do the job properly on engravings. `mass` reverses the order of
operations: Otsu on the full-resolution greyscale, since ink against paper is genuinely bimodal
before any averaging; then the binary mask is downsampled so each cell holds a local ink fraction;
then the fraction is thresholded. On the pictura that had come out as noise, a standing figure is
legible.

**It changes nothing.** On the emblems it reaches 0.121, a gain of +0.027 [−0.003, +0.057] over
random whose interval still contains zero, and it is not distinguishable from the silhouette it
replaced (+0.020 [−0.009, +0.050]). On the museums it reaches 0.251, clearing the prior and the
collection control (+0.069 [+0.033, +0.105]) like the silhouette before it. Two renderings built
on different principles agree with each other on each corpus and disagree across corpora.

**The difference is the corpus, not the rendering.** A binary shape carries iconography on museum
objects and does not carry it on emblem picturae. The likely reason is visible in what the two
corpora index: a museum notation largely tracks object type, and an outline that says *garment*
is most of the way to it, while an emblem notation describes a narrative scene assembled from
small line-drawn figures, of which a 48×48 binary reduction keeps nothing that tells one scene
from another. That also makes the two tables less alike than they look — the emblems task has a
dominant label (prior 0.257, random floor 0.095) and the museums task does not (0.135 and 0.033),
so the columns should be read down, never across.

What is robust across both corpora is the region result. The regions fail everywhere: 0.087 on the
emblems, 0.102 pooled on the museums, both under their priors — including on the corpus where a
binary shape reaches 0.298. The two representations are built from the same picture and keep the
same kind of information — no colour, no texture, no support — and on the museums one works while
the other does not, by twenty points. The difference between them is not shape. It is that the
silhouette stays a continuous raster and the region bag is **discretised into typed parts**:
watershed boundaries, then per-part area, elongation, solidity, tone, Hu moments, a radial
signature, holes, scale ratios and context counts.

**So the loss is in the discretisation, not in the reduction to shape.** That is a direct answer to
the question this record was built to ask. An intermediate visual language made of discrete named
signs — an isotype — throws away what a plain black-and-white outline keeps, on the one corpus
where a plain outline is worth keeping.

Two diagnoses still fit that, and they call for opposite repairs: either the vocabulary throws the
picture away, or it keeps it and the loss happens afterwards, when a bag of parts is summarised by
a mean and a max and the layout goes with it. Deciding between them needs no new classifier, only
a pencil. Each region is **drawn back into an image from its description alone** — contour profile,
size, place, tone — and the reconstruction is put on the same ladder as the renderings it imitates.

It lands at 0.146 on the museums: significantly above noise, significantly *below* the silhouette
it is meant to reproduce (−0.152 [−0.186, −0.117]), and below even the holding collection
(−0.036 [−0.071, −0.001]). That is the same place the region bag reaches on its own (0.102
pooled), by a route with no drawing in it at all. **The two agree, so the vocabulary is where the
picture is lost.** Summarising the bag was never the problem; the descriptors were already empty
of what the silhouette had.

The reconstruction is a lower bound, since the drawing adds losses of its own — a contour profile
is fitted to its bounding box, tone is two values, an overlapping part is painted opaque — and it
should be read together with the region bag, not alone. What makes it worth having anyway is that
it is the first artefact here a historian can contest by looking rather than by reading a table.
The *robe à la française* comes back as three abstract masses; the Glasgow pictura as four. One
look says the vocabulary does not see a garment.

One suspect in that vocabulary was nameable and fixable: the radial signature is rotated to start
at its longest radius, so **orientation was discarded on purpose** — a tilted anchor being an
anchor — and a scene is not orientation-free the way a single motif is. The bin each profile
started from is now kept, and the reconstruction rolls it back. Drawn that way the *robe à la
française* visibly improves: its conical mass and its train land roughly where they belong.

It buys nothing. The oriented reconstruction reaches 0.167 on the museums against 0.146 without
orientation — +0.021 [−0.012, +0.053], an interval containing zero — and remains 0.131
[0.096, 0.166] below the silhouette and below even the holding collection. On the emblems the two
are identical to the third decimal. **The language is not missing a field; it is too coarse.**

Where the coarseness sits is measurable and worth stating plainly: the cap is 24 signs per
picture, but the median picture yields **four**, because a region must cover four thousandths of
the frame to be kept at all. A vocabulary of four parts per picture is being asked to carry what a
48×48 binary raster carries, which is 2 304 cells.

One limit on how far this may be read. The discretisation finding rests on a single contrast — the
museums corpus, where raster shape beats the region bag by twenty points — because the emblems
corpus gives both representations the floor and a contrast between two floors is not a contrast. A
third corpus of narrative scenes with a shape rendering that works on it would decide whether the
discretisation loses scenes as well as objects.

One row must not be added to either ladder. The corpora carry `isotype-*` channels that look like
the obvious candidates for "the proposed visual language", but their method field reads *iconclass
truncated to n characters*: they are built from the target. Scoring them here would measure
nothing but leakage.

The record's own advantage is not the support confound: projecting the holding-library direction
out of it costs three points on the emblems and still clears the prior. What a medium-invariant
rendering removes on purpose — texture, hatching, tonal modelling — is a real part of what names
iconography on engravings, and much less of it on museum objects.

Four faults in that experiment were found and fixed without rescuing it: ranking notations by raw
margins across independently fitted classifiers was wrong, and switching to probabilities changed
nothing because the sigmoid is monotone; taking the most frequent notations made the prior
unbeatable by construction, since on the emblems one label covered three quarters of the test set;
and the museums ceiling covers 869 of the 1 464 test pictures, so its prior is recomputed on
exactly those, which is why it reads 0.135 rather than 0.147. The museums corpus has no rendered
variants, so the ladder is emblems-only; its cropped subset is Illinois and Glasgow, HAB being
the full-page collection excluded throughout.

## What this will not do

Impett and Süsstrunk, clustering Warburg's Bilderatlas on relative limb angles, recovered pose
groups corresponding to Pathosformeln *and* found that morphologically similar poses can
represent wildly different emotions. A transcription language that captures form will find form
recurring. What the recurrence means is not in the record, and should not be claimed from it.

## Reproducing

```
scripts/fetch_wikidata_collection.py   any collection, from Wikidata and Commons
scripts/build_museum_benchmark.py      merge collections into one frozen benchmark
scripts/segment_shapes.py              regions from images
scripts/build_shape_vocabulary.py      signs by clustering, balanced across corpora
scripts/build_shape_groups.py          clumps of touching regions as composite signs
scripts/build_shape_relations.py       seven coarse spatial relations over sign pairs
scripts/build_composition_channel.py   global layout: mass grid, symmetry, profiles
scripts/build_repetition_channel.py    multiplicity as a typed quantity
scripts/build_pose_channel.py          relative limb angles, after Impett
scripts/build_visual_record.py         weighted concatenation of channels
scripts/name_shape_signs.py            a sign's form, associations, and their coherence
scripts/learn_region_labels.py         multiple-instance naming of regions, with the
                                       context and whole-image-record controls
scripts/test_cross_medium.py           the corpus referee
scripts/test_cross_object_type.py      the object-kind referee
scripts/test_by_subject_kind.py        the breakdown by Iconclass division
scripts/probe_confound.py              the probe, linear and kNN
```
