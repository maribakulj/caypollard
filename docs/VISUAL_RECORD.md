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

**Supervising at region level.** Fails under six framings: multiple-instance narrowing reaches
hits@1 0.087 on emblems and 0.053 on museums against priors of 0.257 and 0.263, and a pooled
control reaches 0.061 and 0.096. The pooled control is what makes the failure interpretable — it
beats the narrowing on museums, so a picture's notation is better predicted by all its regions
together than by the one a narrowing procedure elects. **Iconographic identity is not localised
in a single shape**; it lives in the distribution of shapes across a picture.

Two faults in that experiment were found and fixed without rescuing it: ranking notations by raw
margins across independently fitted classifiers was wrong, and switching to probabilities changed
nothing because the sigmoid is monotone; and taking the most frequent notations made the prior
unbeatable by construction, since on the emblems one label covered three quarters of the test set.

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
scripts/learn_region_labels.py         multiple-instance naming of regions
scripts/test_cross_medium.py           the corpus referee
scripts/test_cross_object_type.py      the object-kind referee
scripts/test_by_subject_kind.py        the breakdown by Iconclass division
scripts/probe_confound.py              the probe, linear and kNN
```
