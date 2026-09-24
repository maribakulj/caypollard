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

| question | best configuration | result | floor | pool |
| --- | --- | ---: | ---: | ---: |
| cross-corpus partner (prints ↔ museum works) | shapes + 2× composition + ½ repetition | rank 820 | 9 420 | 18 840 |
| cross-object-kind partner (flat ↔ volume ↔ vessel) | **shapes alone** | rank 1 008 | 6 472 | 12 944 |
| a gesture (division 3) | **pose alone** | rank 65 | 865 | 1 729 |
| a scene (division 1) | **composition** | rank 1 429 | 6 240 | 12 479 |
| an object (division 2) | **the record, all three channels** | rank 655 | 9 420 | 18 840 |
| what a museum object is catalogued as | **named nodes, a bag** | hits@1 0.342 | 0.022 | 1 075 |
| what an emblem is catalogued as | *nothing here works* | hits@1 0.104 | 0.095 | 783 |

The floor column was missing until it was noticed that this table breaks the rule the rest of the
record keeps — a rank means nothing without the chance it beats. The two hits@1 rows were also
sitting in a column headed *median rank*, where a reader scanning down would have compared 0.342
with 820; the column now names its unit per row.

Adding the floors also produced a false alarm worth recording, because the trap is still in the
code. `test_cross_medium.py` reports two rows, and it calls them **"pixels (DINOv2)"** and
**"isotype (signes)"**. The first is not pixels: it is whatever table was passed on the command
line — the record under test. The second is not a record at all: it is a one-hot of the Iconclass
notations themselves, the ground truth, which is why it reads 0.91–0.98 unchanged across files
evaluating completely different records. Read literally, those labels say the record probes worse
than pixels everywhere, and this document was briefly "corrected" to match them before the script
was read. The labels are fixed at the source; the figures here were right.

A sixth channel was added late and belongs in the same table rather than above it. **Named nodes**
— each part given a word from a closed list of 116 ordinary nouns, by a model that can only emit
members of that list — beat every mute representation at saying what a museum object is catalogued
as, and tie the mute shapes at finding a partner across object kinds, and do nothing at all on
emblems. One more channel, chosen by one more question.

Two of these were surprises that reversed an earlier design. Adding pose to the record degrades
nearly every division, and adding composition to cross-object-kind retrieval nearly doubles the
rank. The channels are **a set to select from, not a sum to compute**, and a system built on this
record should choose its configuration from the question rather than carry one vector.

## Why pixels will not do

A DINOv2 embedding names which corpus an object came from 97.4% of the time and places an
object's best cross-medium partner in the middle of the pool. It has learned the material, not
the motif.

This document used to add that re-rendering *does not fix it and can make it worse*, on the
ground that a Sobel map preserves stroke structure and stroke structure is the support. The
figure supporting that sentence, 95.1%, could not be traced to any artifact, so the four
renderings were measured together on one pool of 12 479 with one floor:

| rendering | names the corpus | over its floor | cross-corpus neighbours | median rank of the partner |
|---|---:|---:|---:|---:|
| pixels | 97.4% | +19.2 | 1.9% | 1 303 |
| grey, squared | 96.8% | +18.6 | 2.0% | 1 153 |
| **Sobel edges** | 94.2% | +16.1 | 4.8% | **600** |
| silhouette | 89.3% | +11.1 | 10.7% | 987 |

*Floors: 78.2% for the probe, 38.1% for the neighbours.*

**The sentence was wrong.** Every rendering lowers the probe, monotonically, and Sobel lowers it
further than plain greyscale rather than less; it also more than doubles the cross-corpus
neighbours and gives the best partner rank of the four. Destroying line structure hides the
support best — the silhouette is five points lower on the probe than Sobel — but it does not
retrieve best, and the claim that edge detection *makes things worse* is contradicted on this
corpus. The likely reconciliation is the corpus rather than the method: the earlier reading was
formed when the museum side was paintings only, where a gradient does amplify hatching against
brushwork; against sculpture, dress and vessels it does not.

What helps most on the probe is still destroying line structure. What helps most at finding the
partner is not.

Each row carries its own floors, because the first three are measured on a pool of 4 587 and the
record on the final 18 840, and the floors differ by more than the figures do.

| representation | pool | medium probe | its floor | over floor | cross-corpus neighbours | its floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 on the image | 4 587 | 97.4% | 59.4% | **+38.0** | 2.6% | 47.2% |
| DINOv2 on a Sobel rendering | 4 587 | 95.1% | 59.4% | +35.7 | — | — |
| DINOv2 on a binarised silhouette | 4 587 | 87.5% | 59.4% | +28.1 | 13.5% | 47.2% |
| **the record, final corpus** | 18 840 | **85.2%** | **86.1%** | **−0.9** | **21.8%** | **28.4%** |

The probe target is the majority baseline, not chance. Driving it to chance would mean having
destroyed real iconographic differences between holdings, which are content and not support. On
that measure the record does what it was built to do: it sits *at* its floor, a shade under, where
a pixel embedding sits thirty-eight points above.

The neighbour column is the one that had to be withdrawn. It read *14.5% proportional*, and the
record's 21.8% was printed as clearing it. That floor was the share a *museum* item alone would
see; the figure is averaged over every item that has a cross-corpus partner, four fifths of which
are museum items and one fifth emblems, and computed over that set the floor is **28.4%**. So the
record draws cross-corpus neighbours **three quarters as often as chance**, not more often — and
a pixel embedding draws them one eighteenth as often. The improvement is real and large, fourteen
fold in that ratio, and it does not reach chance. The floor is now computed by the benchmark
itself and written into every artifact, so it cannot be asserted again.

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

**This is the object channel.** Alone it reaches division 2 at rank 1 194 of 18 840 against a
chance of 9 420, and religious scenes at 10 370 — which is not merely chance but a shade worse
than it. (The 655 this document reported for division 2 belongs to the three-channel record, not
to shapes alone; the two rows sit side by side in the same artifact and one number was taken from
each. Adding composition and repetition halves the rank here, so "shapes, with or without the
rest" was wrong in both its number and its claim.)

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

**So the loss is in the discretisation, not in the reduction to shape** — but only half of what
first looked like discretisation was, the other half being the way the parts were summarised. An
intermediate visual language made of discrete signs, read structurally, recovers three fifths of
what a plain black-and-white outline carries; the missing two fifths are the discretisation
itself, and they do not close with a wider vocabulary.

Two diagnoses still fit that, and they call for opposite repairs: either the vocabulary throws the
picture away, or it keeps it and the loss happens afterwards, when a bag of parts is summarised by
a mean and a max and the layout goes with it. Deciding between them needs no new classifier, only
a pencil. Each region is **drawn back into an image from its description alone** — contour profile,
size, place, tone — and the reconstruction is put on the same ladder as the renderings it imitates.

It lands at 0.146 on the museums: significantly above noise, significantly *below* the silhouette
it is meant to reproduce (−0.152 [−0.186, −0.117]), and below even the holding collection
(−0.036 [−0.071, −0.001]). That is the same place the region bag reaches on its own (0.102
pooled), by a route with no drawing in it at all. At this vocabulary size the two agree, and the
vocabulary bounds them both.

They stop agreeing once the vocabulary widens, which is worth stating because the obvious reading
of the paragraph above is wrong. Run on the thirty-two-sign segmentation — same 1 075 pictures,
same notations, same classifier — the redrawn picture reaches 0.194 while the bag of those very
same signs reaches 0.088 pooled. Drawing the signs and looking at the drawing extracts twice what
a mean and a max over their descriptors does, so summarising a bag that way is not free: it costs
nothing at four parts and half at thirty-two.

That was worth chasing, because a summary is cheap to change and a vocabulary is not. **A mean
and a max answer *what is in the picture* and discard *where*.** Replacing them with the same
descriptors pooled per cell of a coarse grid, plus an occupancy count, costs one pass and nothing
else:

| how the bag of thirty-two signs is read | hits@1 | share of the raster's gain |
|---|---:|---:|
| mean and max, as used throughout | 0.088 | 27% |
| relation-conditioned means (the grammar) | 0.101 | 32% |
| **pooled on a 3×3 grid (where the parts are)** | **0.173** | **61%** |
| both | 0.168 | 59% |
| *the same signs drawn, and the drawing encoded* | *0.194* | *69%* |
| *continuous 48×48 silhouette of the same pictures* | *0.270* | *100%* |
| *holding collection alone* | *0.145* | *50%* |
| *random vectors* | *0.022* | *0%* |

**The readout was half the problem.** Reading the same signs structurally more than doubles them,
takes them from a quarter of the raster's gain to three fifths, and lands
−0.021 [−0.050, +0.007] from the drawing — that is, the grid *is* the generous reading, obtained
symbolically and without a rendering step. So the earlier sentence putting the whole loss in the
vocabulary was too strong, and is corrected here rather than left standing.

Two things the grid does not do, both measured paired on the same 1 075 pictures. It stays
**0.097 [0.067, 0.126] below the continuous silhouette**, which is where the remaining loss
genuinely is discretisation. And it still does **not** beat knowing which museum holds the object:
+0.028 [−0.003, +0.058], an interval containing zero. The verdicts below therefore stand, at a
readout that no longer understates the record.

The grammar is the part that does not survive. Relation-conditioned means read 0.158 unscaled and
0.101 once the features are standardised, so that gain was an accident of feature scale rather
than a finding; adding them to the grid costs it five points. *Where* the parts are pays; *how
they stand to each other* does not, by this route or by the picture-level channel.

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

Where the coarseness sits is measurable, and the first place it was looked for was the wrong one.
The cap is 24 signs per picture and the median picture yields four, but lowering the area floor
from four thousandths to one moved that median only from four to five: `min_area` was never the
constraint. The constraint is the blur — seven pixels wide on a 448 side — and the single global
threshold around the page tone, neither of which was reachable from the command line. Both are now
arguments, and with them the vocabulary widens on demand.

That turns the question into a curve. Same corpus, same 869 objects, same classifier, with the
picture redrawn from vocabularies of increasing size:

| museums, redrawn from signs | signs per picture | hits@1 | paired gain over random | share of the raster's gain |
|---|---|---|---|---|
| *random vectors* | — | *0.033* | — | — |
| redrawn | 4 | 0.146 | +0.113 [+0.087, +0.139] | 43% |
| redrawn, oriented | 4 | 0.167 | +0.133 [+0.106, +0.161] | 50% |
| redrawn | 13 | 0.198 | +0.165 [+0.137, +0.193] | 62% |
| redrawn | 32 | 0.214 | +0.181 [+0.152, +0.212] | 68% |
| redrawn | 75 | 0.189 | +0.155 [+0.125, +0.184] | 59% |
| *holding collection alone* | — | *0.182* | *+0.148 [+0.121, +0.177]* | *56%* |
| **silhouette, 48×48 raster** | **2 304 cells** | **0.298** | **+0.265 [+0.232, +0.297]** | **100%** |

**The curve rises, stops rising, and then turns down.** Four signs to thirteen is worth +0.052
[+0.022, +0.083]; thirteen to thirty-two is worth +0.016 [−0.014, +0.047], an interval containing
zero; and seventy-five signs score −0.025 [−0.058, +0.007] *below* thirty-two. The peak sits near
thirty-two parts, and the raster keeps a gap of +0.109 [+0.072, +0.146] over the largest
vocabulary tried. So the shortfall is not a capacity problem that more parts would close: the
discrete language saturates at about two thirds of what the raster of the same picture carries,
and pouring more parts into it makes it slightly worse — past a point the extra signs are
fragments of one object rather than objects, and a picture described by seventy-five splinters is
harder to recognise than one described by thirty-two parts.

The sharper verdict is in the provenance column. At **no** vocabulary size does the redrawn
picture beat simply knowing which museum holds the object: +0.016 [−0.018, +0.052] at thirteen
signs, +0.032 [−0.005, +0.070] at thirty-two, both straddling zero. The raster silhouette beats it
by +0.116 [+0.078, +0.154]. **An isotype-style record, as built here, is not yet worth more than a
provenance field**, and the continuous shape it transcribes is.

One methodological result deserves separating from that, because it cost a wrong conclusion
earlier tonight and would cost others later. The reconstructions improve *visibly* as orientation
is restored and as the vocabulary widens — at thirty-two signs the *robe à la française* has its
conical skirt, its train and the break of its bodice — and the score follows only part of the way,
not at all in the case of orientation. **A redrawing that looks more like the picture is not
evidence that it carries more of it.**

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

## Naming the nodes, which is what the mute vocabulary was missing

Every measurement above is of a vocabulary whose signs are **numbers**. A cluster is "sign 312",
and the saturation those signs hit — three fifths of the raster's gain, no further with more of
them — is a fact about mute signs. It says nothing about signs that carry a name, because until
now none did: every naming route had failed, and a sign remained a tendency.

So the names were fetched from outside instead of learned from inside. Each part is scored
against a **closed vocabulary** and takes the best member of it; there is no free text to
generate, so no author's style and no page's format can re-enter, and no output can fall outside
the vocabulary. The record is then nodes by place — a coarse grid, and in each cell the names of
the parts centred there — which is the form the pooling comparison selected.

Two vocabularies were tried, and which one wins is the finding.

*Iconclass's own labels*, every notation of at most four characters that Wikidata names: 472
terms. *Ordinary nouns*, a closed list of 116 words for what pictures in these corpora contain —
dress, tree, column, angel — carrying **no notation at all**.

| museums, 30 notations, 1 075 pictures | hits@1 |
|---|---:|
| **named nodes, ordinary nouns, by place** | **0.318** |
| named nodes, Iconclass labels, by place | 0.285 |
| *continuous 48×48 silhouette* | *0.270* |
| *constant predictor (prior)* | *0.261* |
| mute 32-sign grid | 0.173 |
| *holding collection alone* | *0.145* |
| *the same places, names shuffled* | *0.101* |
| *random vectors* | *0.022* |

Paired on those pictures, the ordinary-noun record sits **+0.218 [+0.185, +0.248] over shuffled
names**, so it is the names and not the arrangement; **+0.173 [+0.140, +0.207] over provenance**,
which nothing symbolic in this record had managed; and **+0.048 [+0.017, +0.080] over the
continuous silhouette**, which it therefore beats rather than reaches. **The saturation was a
property of mute signs.**

The obvious objection is vocabulary alignment: 11 of the 30 target notations sit literally in the
Iconclass node vocabulary and 27 of 30 have a prefix there, so for many targets that namer can
emit the answer's own code, and shuffling does not control for it — shuffling destroys naming and
alignment together. The ordinary nouns are the control that separates them, since not one of the
116 is a notation. They **win** (+0.033 [−0.001, +0.066] over the aligned vocabulary), so the gain
is naming and not alignment. The same measurement makes a second point for anyone building one of
these: **the discipline's own labels are the wrong prompt**. Given `fable`, `domesticated animal`
and `mandrake` as its only choices, a 3B vision-language model answers `mandrake` five times for a
photograph of a dress — at 32 terms, at 134 and at 465, so the failure is the kind of word and not
the number of them. Given ordinary nouns it answers `dress`. The catalogue term is the right name
for a finding and the wrong name for a prompt; the mapping to Iconclass belongs after the
transcription, not inside it.

It does not win by smuggling the support back in, which was the real risk, since the namer knows
perfectly well what a photograph is. Probed for the holding institution against a 47.3% floor:

| representation | institution named |
|---|---:|
| pixels (DINOv2) | 61.8% |
| named nodes, ordinary nouns | 54.4% |
| named nodes, Iconclass labels | 54.2% |
| continuous silhouette | 54.0% |
| *the same places, names shuffled* | *45.5%* |

**The named record gives away exactly as much provenance as the silhouette and carries more
iconography.** That is the property this whole record was built for, and it is the first
representation here to hold both ends of it at once.

Three limits, none of them small. This is measured on **museum objects only**, whose notations
track object type closely — the corpus where a plain outline already worked, and not the one where
it failed. The transcription is produced by a model trained on photographs: the *product* is
symbolic and probes clean, but its *production* is not medium-blind, and a corpus of engravings
could behave differently. And the vocabulary is a designed artefact, written for these corpora;
it is closed and inspectable, which is the point, but it is not neutral.

### What a name is worth, and to whom

The names come from one instrument, so a second one was given the identical closed vocabulary —
every segmented part scored against it in one case, the picture read whole in the other — and the
two compared on 5 583 museum pictures. Set overlap was the first thing computed and the wrong
thing to report: it counts *man* against *woman* as the same failure as *man* against *vase*, it
punishes the namer that returns more names, and it credits agreement that comes from both namers
saying whatever the corpus says most often.

| | agreement | floor |
|---|---:|---:|
| set overlap | 0.104 | 0.022 |
| one-to-one semantic matching | 0.772 [0.770, 0.775] | 0.713 [0.711, 0.715] |

The second row's *absolute* value is a trap and is recorded here so that nobody reads it as
agreement: in a sentence encoder every ordinary noun sits near every other, so the floor is 0.713
and only the excess of 0.059 is information. Read that way the two measures agree, and set
overlap was overstating the disagreement rather than discovering it.

Reliability is not a property of the namers but of **each word**, and the spread is the finding.
Confirmed by the other namer against how often that namer says the word at all:

| word | proposed | confirmed | base rate | lift |
|---|---:|---:|---:|---:|
| ship | 47 | 0.53 | 0.016 | 34 |
| fruit | 82 | 0.42 | 0.020 | 21 |
| clock | 61 | 0.21 | 0.005 | 44 |
| … | | | | |
| chain | 209 | **0.00** | 0.001 | 0 |
| drum | 246 | **0.00** | 0.000 | — |
| glove | 127 | **0.00** | 0.000 | 0 |

A quarter of the vocabulary is trustworthy and part of it is noise a namer emits in the hundreds
and the other never confirms once. **The vocabulary can be pruned by measurement rather than by
taste**, which is the practical consequence.

A single name carries a great deal about the catalogue — a picture called *ship* is 38 times more
likely to carry its notation than a picture at large, *key* 24, *mountain* 23. And the question
the agreement measurement existed to answer, asked directly rather than by proxy: **do names both
namers gave lift more than names only one gave?**

| | mean lift on the catalogue |
|---|---:|
| names both namers gave (n = 4 807) | **2.551 [2.486, 2.618]** |
| names only one gave (n = 51 014) | 1.747 [1.735, 1.761] |

Disjoint intervals, 46% apart. **Agreement does mark reliability.** So an intersection record is
worth building, against a cost that is now a number rather than a worry: it keeps 8.6% of the
assignments, and whether that trade is good is the measurement that follows.

### Five things that should have helped and did not, and the one that did

The named record was built on a chain of reasonable beliefs, and taking them apart one at a time
changed the design more than adding anything to it. Each row below is the same 1 075 museum
pictures, the same classifier, the same floors.

**A better namer makes a worse record.** The second namer enumerates distinct things where the
first repeats itself, and its names are the ones a person would give — man, woman, angel, crown.
Its record reaches 0.264 against the first namer's 0.318, −0.054 [−0.087, −0.021]. Per-name
quality and record quality are not the same quantity.

**Agreement marks reliability and the intersection still loses.** Names both namers gave lift 46%
more on the catalogue, which is why the intersection was worth building; it keeps 8.6% of the
assignments and scores 0.253, −0.044 [−0.088, −0.003] against the first namer alone. The
sparseness eats the reliability, exactly the trade that was called a measurement rather than an
intuition.

**Cutting the unreliable third of the vocabulary changes nothing.** Reliability measured on the
training split alone — never on the pictures the result is read from — keeps 81 words of 116 and
drops the ones one namer proposes in the hundreds and the other confirms never. The record moves
by −0.005 [−0.028, +0.019]. The vocabulary can be cut by 30% for free, and buying anything with
the cut is not on offer.

**More nodes is slightly worse.** Three nodes a picture 0.330, six 0.327, twelve 0.318. And a
namer's advantage is not its node count: the first namer at three nodes beats the second at 3.1
by 0.066.

**Position adds nothing once the nodes are named.** This is the one that revises the architecture.

| | with position | without | what position adds |
|---|---:|---:|---:|
| first namer, 3 nodes | 0.330 | **0.342** | −0.012 [−0.038, +0.014] |
| second namer | 0.264 | 0.225 | +0.039 [+0.009, +0.069] |

The record is *better* as a bag. Position helps the second namer only because it returns so few
names that a cell is the only further thing it has to say. With position removed from both, the
gap between the namers is +0.116 [+0.083, +0.149] — **it was the names all along**.

Set beside the mute signs, where pooling on a grid *doubled* the score from 0.088 to 0.173, that
is a single sentence: **position was standing in for identity.** A vocabulary that cannot say what
a part is needs to say where it is; once the parts have names, where they are stops carrying
anything. The "nodes by place" design was right for signs and wrong for names, and the measurement
rather than the design decides which.

So the record that survives all five is the simplest one on the list — **a bag of named nodes, no
position, three nodes a picture, 116 ordinary words**:

| museums, 30 notations, 1 075 pictures | hits@1 |
|---|---:|
| *pixels, the ceiling* | *0.567* |
| **named nodes, a bag** | **0.342** |
| named nodes, by place | 0.318 |
| *continuous 48×48 silhouette* | *0.270* |
| *constant predictor (prior)* | *0.261* |
| *holding collection alone* | *0.145* |
| *random vectors* | *0.022* |

Paired: **+0.073 [+0.039, +0.107] over the silhouette it transcribes**, **+0.197 [+0.166, +0.231]
over provenance**, and −0.225 [−0.261, −0.189] against the pixels it will not reach. It recovers
59% of what a photographic embedding recovers over noise, out of 116 readable words, no texture
and no support.

One bug is recorded because catching it was luck rather than method. Collapsing the grid to a
single cell first *dropped* every node whose cell was not zero instead of moving it there, so
"without position" was silently "only the parts in the top-left corner". The numbers looked
plausible; what gave it away was that the paired comparisons ran on 58 to 484 pictures where the
others ran on 1 075. **Check the n before reading the effect.**

### Does naming help the question the record was built for?

Predicting what a picture is catalogued as is one question. Finding the same motif on an object
made and seen differently is the other, and it is the one this record exists for. Naming answers
them differently, which is the same lesson the channels taught and is worth having twice.

**Across corpora — prints against museum objects — the named record retrieves five times better
and gives the game away.** On 12 424 objects, 7 037 of which have a partner in the other corpus:

| | pixels | named nodes |
|---|---:|---:|
| median rank of the best partner | 501 | **104** |
| partner in the top 10 | 5.2% | **14.1%** |
| partner in the top 50 | 16.8% | **33.5%** |
| *names the corpus, linear probe* | *82.6%* | *96.9%* |
| *neighbours drawn from the other corpus* | *22.6%* | *5.9%* |

Against a majority baseline of 78.1%, pixels sit 4.5 points above and the named record 18.8. The
mute record sat *at* its floor — 85.2% against 86.1% — so **on the criterion this whole record was
built for, naming is by far the worst representation measured.** The likely reason is not support
but content that happens to align with the split: emblems hold swords, crowns and banners, museums
hold vases, textiles and frames, and a namer that works must separate them. The probe cannot tell
that apart from support, which is a limit of the probe.

**The benchmark that removes the confound removes the advantage too.** Crossing object families
inside the museum corpus alone — a flat surface, a thing in the round, a curved vessel — there is
no corpus to read off. Same pool of 6 384, chance at 3 192, 2 983 queries:

| | median rank | partner in top 100 | neighbours from another family |
|---|---:|---:|---:|
| named nodes | 749 | **24.7%** | 14.3% |
| mute shapes | **743** | 19.6% | **21.1%** |
| continuous silhouette | 1 820 | 15.0% | 11.3% |
| pixels | 3 609 | 9.0% | 2.5% |

**A tie**, inside one per cent. Naming ranks the true partner higher when the list is read deep;
the mute vocabulary volunteers more cross-family neighbours without being asked. Both are four
times better than chance and four to five times better than pixels, which on this benchmark is
*worse than chance*.

So naming is a channel and not a replacement. It wins where the question is *what is catalogued
here* — +0.073 over the silhouette at notation prediction, +0.197 over provenance — and ties where
the question is *find me this motif on a different kind of object*. That is the finding that
organises this whole record, arrived at a third time by a third route.

### Where naming stops

The named record was measured on museum objects. The emblems are the corpus where mute shape
failed outright — silhouette at its floor, region bag under its prior — so they are the only place
naming could do better than tie. It does not.

| emblems, 30 notations, 780 pictures | hits@1 |
|---|---:|
| *pixels* | *0.324* |
| *constant predictor (prior)* | *0.257* |
| named nodes, a bag | 0.104 |
| *continuous silhouette* | *0.101* |
| *random vectors* | *0.095* |
| *holding collection alone* | *0.089* |
| *the same names shuffled* | *0.061* |

Against the silhouette, +0.004 [−0.026, +0.033]: nothing. It clears only its own shuffled control,
+0.042 [+0.017, +0.069], so the names carry *something* and not enough to leave the floor. **Naming
does not rescue the emblems, and the whole named result is bounded to museum objects.**

The reason is in the transcription and is countable rather than a story. On the emblems the five
commonest words cover **50%** of all assignments — *man* alone 21%, *horse* 12% — against 38% on
the museums. An emblem pictura is very often a man, a horse and a tree, so a vocabulary of object
nouns describes them all alike; and an emblem's notation records what the scene *means*, which is
the layer a list of things cannot reach. The same vocabulary that discriminates a garment from a
vase cannot discriminate one allegory from another.

One earlier tension resolves here too. Across corpora the named record named the corpus 96.9%
against a 78.1% baseline, which looked like the support coming back. Probed *within* the museum
corpus, across its eight institutions, it reads 55.2% against a 47.3% baseline — +7.8 points where
the silhouette is at +6.7 and pixels at +14.5 — and its neighbourhoods do not cluster by
institution at all, 47.25% against a 47.3% floor. So the cross-corpus figure was **content aligned
with the split**, not support: what emblems and museum objects depict genuinely differs, and a
namer that works has to say so.

## Peeling the layers: one regime at a time

Every benchmark above bundles. A Greek vase painted with dancers is four things at once — a
*subject* (people dancing), an *object kind* (a curved vessel whose own form arranges its
decoration), a *period*, and a *museum photograph* with that institution's ground and lighting —
and asking whether a representation "crosses the medium" asks about the bundle. Worse, corpora
confound them by construction: a corpus of eighteenth-century paintings against one of medieval
portraits will cluster by corpus whatever the representation does, because its *content* really
does differ.

A regime names one attribute as the target and **requires the others to differ**, so a partner can
only be found for the reason the regime is about. The subject regime wants the same notation on a
different kind of object, from a different collection, of a different century: the vase with
dancers finding a modern painting of dancers rather than a vase with athletes. One pool of 12 349
museum objects, four regimes, four representations.

The floor is computed per query rather than taken as half the pool, because the regimes are not
equally hard: under a random ordering the expected rank of the best of *k* targets in a pool of
*n* is (n+1)/(k+1), which differs by three orders of magnitude between the tightest regime and the
loosest. The ratio to that floor is what can be compared across regimes.

| regime | named nodes | pixels | mute shapes | silhouette | floor |
|---|---:|---:|---:|---:|---:|
| **subject** — other kind, other collection, other century | **×1.55** | ×1.27 | ×1.11 | ×0.87 | 834 |
| **object kind**, rare kinds, no subject shared | ×0.70 | **×2.26** | ×1.50 | ×1.93 | 487 |
| **century**, no subject shared, other kind | **×1.14** | ×0.39 | ×1.03 | ×1.04 | 74 |
| **collection**, no subject shared, other kind | ×1.11 | ×0.33 | **×1.27** | ×0.95 | 104 |

**The ordering inverts.** Named nodes are first on subject and last on object kind; pixels are
first on object kind and last on both century and collection. There is no best representation
here, and saying one beats another is a category error — they answer different questions, and the
question is chosen by what the searcher wants to hold constant.

Three readings follow, and the third is the one that changes what this record is for.

**The named record earns its place on the question it was built for.** Finding the same subject
across a different kind of object, a different collection *and* a different century is the hardest
regime — a floor of 834 in a pool of 12 349 — and it is the one where naming wins, reaching the
partner within the first hundred for 27.8% of queries against 21.7% for pixels. Every earlier
benchmark that called this a tie was bundling object kind into the question.

**Pixels are keyed to the object and blind to everything else.** They are more than twice the
floor at recognising a rare kind of thing and *below* the floor at century and collection. A
photographic embedding knows what sort of object it is looking at and nothing about when it was
made or who holds it — which is the opposite of what the medium confound suggested, and follows
from it: pixels cluster by object and subject, so asked for a same-collection partner of a
different kind about a different subject, they return exactly what the regime excludes.

**And the silhouette inverts against the named nodes.** It is second on object kind and last on
subject, they are first on subject and last on object kind. The two are not competing
representations of one thing; they are the outline and the name of it, and a system holding both
can choose which one the question wants.

## The grammar, twice asked and twice refused

An isotype is a closed vocabulary *and* a grammar — a lion beneath a crown is not a lion wearing
one — so the relation between signs was built as its own channel: seven coarse relations
(above, below, left, right, contains, inside, touching) over ordered pairs of signs, computed
from geometry the segmenter already records.

On the first corpus of 4 587 objects it helped, moving the best cross-medium partner from median
rank 92 to 84 and lifting the share reached within fifty from 35.8% to 41.0%. It was then never
carried over when the corpus grew to 21 128 — dropped at a change of base rather than after a
failure, which left it the one channel never retested. It has now been retested, paired on the
20 146 objects it covers:

| division | shapes | shapes + relations | difference | queries |
|---|---:|---:|---:|---:|
| 1 · religion (scenes) | 2 589 | 2 613 | −24 | 1 174 |
| 2 · nature (objects) | 920 | 924 | −4 | 2 954 |
| 3 · body, action | 3 679 | 3 650 | +29 | 1 782 |
| 4 · society | 1 280 | 1 279 | +1 | 9 957 |
| 9 · classical mythology | 1 225 | 1 206 | +19 | 664 |

**Nothing.** On the largest division the median moves by one rank in a pool of twenty thousand,
and the two visible swings sit on 182 and 51 queries. The narrow-corpus gain does not replicate,
and it joins the list of conclusions this project drew from a base too thin to carry them.

A second, harder obstacle showed up in building it, and it is a property of the design rather
than of this corpus. **The relation vocabulary is the square of the sign vocabulary.** At the 256
signs the record actually uses, there are 458 752 possible (sign, relation, sign) features; 247 631
occur, and only **635** occur the thirty times needed to be a pattern rather than an accident, so
**14 651 of 21 128 pictures come out with no relation feature at all**. The channel only exists at
64 signs, where 20 146 pictures keep one — and a grammar that requires a coarser lexicon than the
lexicon channel wants is not a free addition to it.

The same question was put a third way, on the annotation rather than the image: hide one notation
per item, represent the item by what remains, let its neighbours vote it back. A lexicon of bare
notations reaches hits@10 0.244, the same lexicon with Iconclass's own modifiers 0.254, and the
lexicon **plus co-occurring pairs — the grammar — 0.236**, against a frequency prior of 0.282 that
none of them beats. Adding the grammar made it worse there too.

Three constructions, three refusals. What that does not settle is whether a grammar read
*structurally* would do better than a grammar read as more features in a bag, which is the
distinction the pooling comparison below was built to test.

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
