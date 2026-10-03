# An intermediate visual language for cross-medium iconography

A curated orientation map, not a systematic review. It answers a question the project's
[review protocol](../LITERATURE_REVIEW_PROTOCOL.md) did not originally pose, and which the
measurements in [`RESULTS.md`](../RESULTS.md) forced onto it: **what should an image be transcribed
into, so that a printed emblem and a painted panel sharing a motif become comparable?**

## Why the question is now the central one

Pooling 5 889 emblem prints with 1 864 museum objects — overwhelmingly paintings, with drawings,
watercolours, etchings, pastels and reliefs behind them — gives 7 753 items joined by 326 shared
Iconclass concepts, and 5 127 of them have at least one cross-corpus iconographic partner.

Measured on that pool, a DINOv2 embedding names which corpus an object came from **98.1% of the
time** and places an object's best cross-medium partner at **median rank 3 285 of 7 753** — the
middle of the pool, which is chance. Only 1.9% of an object's ten nearest neighbours come from
the other corpus, where the other corpus is a quarter of the pool. The encoder has learned the
material, not the motif.

The same pool, represented by the human Iconclass transcription alone, puts that partner at
median rank 113. The geometry a symbolic representation produces is therefore usable and the
geometry pixels produce is not, so the bottleneck is transcription rather than ranking. (That
second figure is circular as a retrieval result — partners are *defined* by a shared sign — and
is quoted only as a statement about the target geometry.)

## What the language has to satisfy

| requirement | why | how it is checked |
| --- | --- | --- |
| closed vocabulary | a lexicon with no word for "hatching" cannot leak the support | the probe |
| compositional | a lion beneath a crown is not a lion wearing one | held-out prediction |
| medium-blind | the point of the exercise | probe to ~92%, the content floor, **not** to chance |
| machine-producible | 7 753 objects is already past hand transcription | coverage and error rate |
| human-inspectable | an art historian must be able to contest a transcription | by construction |

The third row is the one most easily got wrong. Driving the probe to the 76% majority would mean
having destroyed real iconographic differences between holdings: Wolfenbüttel's German
devotional emblems and Glasgow's Alciato tradition genuinely depict different things. The floor
is content, not chance.

## Five strands, and what each supplies

### 1. The historical picture languages

**Isotype** — Otto Neurath with Marie Reidemeister and Gerd Arntz, Vienna, from 1925. Arntz
designed some 4 000 pictograms. Neurath described it as a *language-like technique*: repeatable
typed units, quantity expressed by repetition, rigorous consistency of graphic element. Its
grammar is thin — it was built for statistics, not for describing an arbitrary picture — but its
discipline of **reduction to types** is exactly right, and our own measurement says reduction is
free: truncating Iconclass from 4 034 signs to 9 changes the medium leakage by eight points and
nothing else.

**Blissymbolics** — Charles K. Bliss, *Semantography*, mid-century. About a hundred primitives
that **combine** into compound concepts, rather than a catalogue of finished signs. This is the
compositional architecture Isotype lacks, and it is much closer to what a transcription language
needs: not a list of everything depictable, but a small set of elements plus rules for building.

*Supplies:* the reduction ethos, and the proof that a hundred primitives can compose to cover a
domain. *Lacks:* any way to produce a transcription from an image, and art-historical coverage.

### 2. The discipline's own vocabularies

**Iconclass** is the only candidate with real coverage and institutional adoption, and it already
carries more structure than this project has been using: 16.6% of the notations in the
Emblematica corpus bear a `(+N)` modifier key, which the pipeline stripped from phase 1 until
this week. Keeping them measurably helps — held-out sign recovery rises from 0.195 to 0.219
hits@10 — and costs nothing.

What it lacks is spatial relations. It records *that* a lion and a crown are depicted, never that
one is above the other. It is also, as established at length, **denotative**: two lions of
opposite meaning share `25F23`, so the vocabulary cannot by itself express a difference of
interpretation.

*Supplies:* a vetted lexicon of ~40 000 typed signs, a hierarchy, a modifier system, a Wikidata
alignment of 4 125 notations for multilingual labels. *Lacks:* the grammar, and any connotative
layer beyond division 5.

### 3. Scene graphs — the formalism that supplies the grammar

The computational form of "lexicon plus grammar" already exists and is standard: a **scene
graph** is objects as nodes, attributes on nodes, and relations as edges — actions, spatial
prepositions, comparatives. Two large surveys cover generation and application
([arXiv:2201.00443](https://arxiv.org/pdf/2201.00443),
[arXiv:2104.01111](https://arxiv.org/pdf/2104.01111)).

The obvious construction for this project is therefore **Iconclass primitives as the node
vocabulary and a small closed relation set as the edges** — above, holds, wears, faces, contains,
flanks. Nodes come from the discipline, edges come from the formalism, and both are finite, so
neither can encode a paper texture.

*Supplies:* the shape of the representation, and a mature literature on producing it. *Lacks:*
any art-historical grounding in its usual relation vocabularies, which are built for photographs
of everyday scenes.

### 4. The evidence that composition is what survives a change of medium

The problem has a name and a literature. **The cross-depiction problem** — recognising objects
regardless of whether they are photographed, painted or drawn — was posed by Cai, Wu, Corradi and
Hall ([arXiv:1505.00110](https://arxiv.org/pdf/1505.00110); *Computational Visual Media*,
[10.1007/s41095-015-0017-1](https://doi.org/10.1007/s41095-015-0017-1)). Their conclusions read
as a direct instruction:

1. appearance-based recognition systems **over-fit to one depiction**;
2. models that **explicitly encode spatial relations between parts are more robust**;
3. recognition and non-photorealistic synthesis are related tasks.

The first is what our 98.1% probe measured. The second is an independent empirical argument for
the compositional requirement, and it is stronger evidence than our own crude test produced —
adding co-occurring pairs of signs as features did not help, but the corpus records no spatial
relation, so "these two signs appear together" was all the composition available.
[SwiDeN](https://arxiv.org/pdf/1607.08764) follows with a depiction-invariant architecture that
switches on depictive style.

*Supplies:* the name of the problem, and the finding that structure transfers where appearance
does not.

### 5. The art-historical precedent — Warburg, and its warning

Aby Warburg's *Pathosformel* and the *Mnemosyne Bilderatlas* are the original project of tracking
a formula across media and centuries, which is precisely the goal here.

Its computational realisation is **Impett and Süsstrunk, *Pose and Pathosformel in Aby Warburg's
Bilderatlas*** (ECCV Workshops 2016,
[10.1007/978-3-319-46604-0_61](https://doi.org/10.1007/978-3-319-46604-0_61)). They crowdsourced
2D human pose across a third of the Bilderatlas panels and clustered on **relative limb angles
alone** — a deliberately medium-blind, compositional, low-dimensional code that discards
everything about the support. They recovered pose clusters corresponding to Pathosformeln.

And they found that **morphologically similar poses can represent wildly different emotions.**

That is this project's H3, reached by the closest precedent from the other direction. A
transcription language that captures form will find the form recurring and will not, on its own,
tell you what the recurrence means. It is a discovery instrument, not an interpretation.

*Supplies:* a working example of transcription-by-relation on real art-historical material, and
the sharpest available warning about what it will and will not deliver.

### 6. Learned vocabularies, noted and set aside

A discrete codebook can be learned rather than designed — vector quantisation gives a finite set
of visual "words". But a codebook learned on pixels will spend its capacity on texture, which is
the support, so it would need the medium probe as an adversarial training signal. Speculative
here, and it forfeits inspectability, which is the requirement an art historian will care about
most.

## What this suggests building

Nodes from Iconclass, reduced — the measurement says reduction is free, so choose the level an
art historian can argue with rather than the finest available. Edges from a small closed relation
set. Produced by a vision-language model under **constrained decoding**, so its output is a
structure in that vocabulary rather than free text, which would reintroduce the author's style
and the page's format. Validated by the probe against the ~92% content floor, by held-out sign
recovery against the frequency prior, and by human inspection of the transcriptions themselves.

Three risks, stated in advance. Impett's: same form, different meaning. Ours: in emblematics,
motifs combine too freely for co-occurrence to predict much, which may be a property of the genre
— a more conventional iconography, religious painting for instance, would test this. And the
ordinary one: transcription errors compound through a pipeline, so the transcription must be
inspectable at every stage, which is an argument for a small vocabulary over a rich one.
