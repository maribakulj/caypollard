# Expert evaluation protocol (phase 10)

## What this measures, and what it does not

Every number in [`RESULTS.md`](RESULTS.md) is an agreement between a ranking and an annotation.
None of them says whether a retrieval is *useful to a scholar*. Phase 10 exists because those are
different questions, and because a project that has already found two of its own confounds should
not assume the third.

The evaluation therefore answers one question — **does a metric improvement correspond to a
judgement improvement?** — and is designed so that a "no" is as reportable as a "yes".

## Five dimensions, never collapsed

The roadmap forbids reducing these to a single "relevance" score, and the project's own results
show why: a retrieval can be iconographically apt and historically meaningless, or contextually
apt and iconographically empty, and a single number cannot tell those apart.

| dimension | question put to the rater |
| --- | --- |
| `ressemblance_visuelle` | Do the two images resemble one another formally? |
| `parente_iconographique` | Do they represent the same subject or motif? |
| `pertinence_historique` | Is there a historical or contextual link between them? |
| `utilite_pour_la_recherche` | Would this pairing be useful in a piece of research? |
| `confiance` | How confident are you in the answers above? |

Scale: **1** not at all · **2** slightly · **3** moderately · **4** considerably · **5** entirely ·
**?** cannot judge.

`?` is a real answer and must stay available. An expert on emblem books asked about a Munich
incunabulum should decline, and a protocol that forces a number there manufactures data. The
analysis excludes those cells and counts them.

## Sampling

`scripts/build_expert_evaluation.py` pools the top 5 of each method per query, and stratifies
queries into **terciles of how much the methods disagree**, measured as the size of the pooled set
relative to *k*.

Fixed thresholds were tried first and put 2 456 of 2 804 queries into a single band, because a
graph ranking and a visual ranking share about a twelfth of their neighbourhood and so almost
never agree by an absolute standard. Terciles make the sample span this corpus's own range rather
than an imported one. The default package is 8 queries per band — 24 queries, 255 pairs.

Sampling only the disagreements would measure the method where it is most conspicuous rather than
where it is most typical, which is why the agreement band is included even though it is the least
interesting to read.

## Blinding

The rater receives `package.jsonl` and `rating-sheet.csv`. Neither carries a method name, a rank,
or the pooling order: candidates are shuffled with a fixed seed. The mapping from pair back to
method and rank lives in `KEY-do-not-open.jsonl`, which only the analysis reads.

This matters more than it sounds. An expert shown three columns headed "visual", "graph" and
"multimodal" is no longer judging retrievals; they are judging a hypothesis, and usually the one
they can see the author hopes for.

## Procedure

1. Build the package: `scripts/build_expert_evaluation.py`.
2. Send each rater `package.jsonl`, `rating-sheet.csv`, and this document. Ask them to fill the
   `rater` column with a stable pseudonym.
3. **At least two raters must rate every pair**, or agreement is not estimable. A pair rated once
   is reported as skipped, not averaged in.
4. Return the filled sheets and run `scripts/analyse_expert_evaluation.py`.

**Effort.** 255 pairs across 5 dimensions is roughly two hours per rater at a steady pace. If that
is too much to ask, reduce `--queries-per-band` rather than dropping a dimension or a rater: the
design degrades gracefully in sample size and not at all in the other two directions.

## Analysis, fixed in advance

- **Agreement**, per dimension, by Krippendorff's alpha with an *ordinal* difference function.
  Percentage agreement is not chance-corrected, and unweighted kappa counts a 1-versus-5
  disagreement the same as 1-versus-2. Alpha also tolerates the `?` gaps.
- **The blind comparison**: mean rating per method per dimension. A pair returned by several
  methods counts for each, which is the correct reading of a pooled design.
- **Human against automatic**: Kendall's tau-b between each dimension's mean rating and the graded
  Iconclass relevance the metric assigned to the same pair. Tau-b rather than a score correlation,
  because the two live on different scales and ordinal ratings tie constantly.
- **Disagreements as items, not as variance.** Pairs where raters differ by 2 points or more are
  returned whole and worst-first. The roadmap asks for them to be analysed, and a standard
  deviation analyses nothing.
- **By band**, so that the answer can depend on how much the methods disagreed to begin with.

## What would count as what

- **Alpha below about 0.4 on a dimension**: that dimension is not being measured reliably, and no
  comparison resting on it should be reported. This is a finding about the protocol, and it is the
  most likely single outcome for `pertinence_historique`.
- **Fused rated above both single modalities, on `utilite_pour_la_recherche`**: the metric gain
  corresponds to something a scholar wants. This is the result the project hopes for and has no
  evidence for yet.
- **Fused rated no higher, with tau-b near zero**: the metric and the judgement are measuring
  different things, and the project's central claim is about the metric only. Given that graph
  fusion is four fifths its visual arm and fails every hard-case test, this is the outcome to
  expect, and it must be reported as plainly as the other.

## Status

Everything above is built and tested: the package builder, the agreement module (`krippendorff_alpha`,
`kendall_tau_b`, `disagreements`), and the analysis script, exercised end to end on synthetic
judgements so that no expert's time is spent discovering a bug.

**What is missing is the experts.** Recruiting them, and whatever consent and acknowledgement
their institutions require, is a decision outside this repository.
