# Research Plan

## 1. Problem statement

Cultural-heritage image retrieval increasingly relies on high-dimensional visual or vision-language embeddings. These representations are powerful, but the neighbourhood they create is not equivalent to an art-historical, iconographic, or curatorial notion of relatedness.

A knowledge graph offers a different representation: explicit relations among objects, concepts, creators, publications, places, dates, vocabularies, and collection records. The research problem is therefore not simply how to combine two vector types. It is how to determine whether graph-structured knowledge changes retrieval in ways that are **measurable, interpretable, and useful for heritage research**.

## 2. Primary research question

> Can knowledge-graph structure improve the retrieval of iconographically related cultural-heritage images beyond what visual embeddings alone recover?

## 3. Secondary questions

### RQ1 — Representation

What kinds of similarity are encoded by visual, textual, and graph embeddings, and where do their neighbourhoods disagree?

### RQ2 — Fusion

Do transparent combinations such as weighted late fusion or graph reranking improve retrieval before a learned joint representation is introduced?

### RQ3 — Difficult cases

Does graph information help specifically with:

- visually dissimilar but iconographically related images;
- visually similar but iconographically unrelated images?

### RQ4 — Generalisation

Do gains survive work-/edition-/collection-aware splits and cross-collection transfer?

### RQ5 — Interpretability

Can graph paths and shared concepts explain why a multimodal method changes a ranking?

### RQ6 — Humanities usefulness

Do automatic improvements correspond to relationships that heritage researchers judge useful, plausible, or worthy of further investigation?

## 4. Hypotheses

### H1 — Complementary signal

Graph neighbourhoods overlap only partially with visual neighbourhoods and contain independent semantic information.

### H2 — Hard-positive benefit

KG-aware methods improve ranking for pairs that share iconographic meaning despite substantial visual difference.

### H3 — Hard-negative suppression

KG-aware methods reduce false positives caused by formal similarity without semantic/iconographic relevance.

### H4 — Cross-collection generalisation

At least part of the KG contribution persists when evaluation moves to a collection not used to fit the fusion model.

### H5 — Complexity must earn its place

A learned joint model is justified only if it outperforms or provides qualitatively distinct value beyond weighted late fusion and reranking.

## 5. Conceptual definitions

### Visual similarity

Proximity under an image encoder. This may reflect composition, shape, colour, object presence, style, pose, texture, or semantic associations learned from training data.

### Iconographic similarity

Relatedness of represented subjects, motifs, allegories, narratives, symbols, or concepts, operationalised initially through expert Iconclass structure and later through historically richer metadata.

### Contextual/historical similarity

Relatedness induced by graph relations such as shared work, edition, author, printer, place, date, collection, source, or documented concept.

### Multimodal similarity

A ranking produced from more than one representational source. It must not be assumed to be more historically valid merely because it uses more modalities.

## 6. Why Iconclass first

Iconclass provides a controlled setting because images are linked to expert-assigned hierarchical concepts. This enables graded relevance rather than flat binary labels and makes it possible to construct disagreement cases between visual and semantic similarity.

The first benchmark should use Iconclass to answer a methodological question, not to claim that Iconclass itself is a complete model of meaning.

## 7. Why Emblematica second

Emblems are structurally advantageous because image, motto, text, bibliographic object, and iconographic description are partially separable. This permits experiments on the interaction between formal resemblance, iconography, language, and publication context.

The Emblematica stage changes the project from a representation benchmark into a historical case study.

## 8. Why Rijksmuseum as transfer target

A large museum collection offers a different institutional context, object distribution, cataloguing practice, and graph structure. Cross-collection evaluation is essential to distinguish generalisable multimodal behaviour from collection-specific metadata memorisation.

## 9. Why WJoconde is secondary

WJoconde is valuable for comparing the approach with current multimodal cultural-heritage KG research, especially KG completion/enrichment. It should not replace the primary retrieval benchmark because the project's main question runs in the opposite direction: whether KG structure improves image discovery.

## 10. Experimental families

### 10.1 Vision-only

Candidate encoders:

- DINOv2;
- CLIP/OpenCLIP;
- SigLIP.

Each must record exact model/revision and image preprocessing.

### 10.2 Graph-only

Candidate methods:

- Node2Vec;
- RDF2Vec;
- ComplEx;
- RotatE.

The graph projection used for each method must be explicitly documented because changing edge semantics changes the experiment.

### 10.3 Text-only

Text baselines should use available descriptions/mottoes/captions without leaking test labels into the input.

### 10.4 Fusion

#### Weighted late fusion

A transparent score combination. It is the mandatory baseline.

#### Graph reranking

Visual retrieval generates candidates; graph information reranks them.

#### Learned shared projection

Frozen source encoders feed small projection heads trained under a contrastive objective. This is preferred before end-to-end fine-tuning because it isolates the value of alignment from model capacity.

## 11. Experimental design

### 11.1 Data partitions

Evaluate multiple split strategies:

- random image split only as a diagnostic lower bar;
- grouped by work/book;
- grouped by edition;
- grouped by creator where relevant;
- grouped by collection;
- temporal split where metadata supports it;
- cross-collection transfer.

### 11.2 Duplicate control

Where possible, use perceptual/embedding-based duplicate discovery to prevent near-identical plates from straddling train and test.

### 11.3 Hard-pair construction

Construct four classes from orthogonal visual and iconographic similarity estimates. Freeze the evaluation subset before tuning fusion parameters.

### 11.4 Ablation

At minimum compare V, G, T, V+T, V+G, G+T, V+G+T.

### 11.5 Hyperparameter discipline

Hyperparameters are tuned on validation data only. Test performance is reported once per frozen protocol or clearly labelled exploratory if protocol changes.

## 12. Relevance model

Flat exact-label matching is insufficient for hierarchical iconography.

A graded relevance score should consider the relation between two concept sets. Candidate approaches include:

- shortest-path distance in the hierarchy;
- depth of lowest common ancestor;
- information-content-weighted similarity;
- set-to-set maximum/average matching for multi-label images.

The metric itself must be analysed because a poorly chosen semantic distance can predetermine the result.

## 13. Metrics

### Automatic

- Recall@K;
- MRR;
- mAP;
- nDCG@K;
- hard-positive recall;
- hard-negative error rate;
- top-K ranking overlap between modalities.

### Representation diagnostics

- hubness;
- neighbourhood stability;
- graph degree correlation;
- performance by concept depth;
- performance by label frequency;
- performance by collection/source.

### Human/expert evaluation

Separate ratings for:

- visual resemblance;
- iconographic relatedness;
- historical/contextual relevance;
- research usefulness;
- confidence.

## 14. Statistical analysis

Use paired evaluation over queries whenever possible.

Recommended analyses:

- bootstrap confidence intervals;
- paired permutation/bootstrap tests;
- effect sizes;
- seed variance for learned models;
- subgroup confidence intervals;
- multiple-comparison correction where many model variants are tested.

## 15. Interpretation strategy

The project should expose *why* graph information changed a result. For a retrieved candidate, record where possible:

- visual similarity score;
- graph similarity score;
- fused score;
- shared concepts;
- shortest interpretable graph path;
- metadata responsible for contextual relatedness.

This allows retrieval examples to become evidence rather than decoration.

## 16. Threats to validity

### Circular ground truth

Using Iconclass to train graph representations and then evaluating only on Iconclass structure risks circularity.

**Response:** treat direct taxonomy-derived image vectors as oracle controls only; for headline KG conditions, remove test-image target Iconclass edges or use context-only graph projections, then use transfer evaluation and expert judgement. See `GRAPH_PROJECTIONS.md` and `protocol-v0.2.md`.

### Collection bias

Metadata conventions may be institution-specific.

**Response:** cross-collection transfer and source-stratified evaluation.

### Duplicate leakage

Reprints or reused plates can inflate performance.

**Response:** group-aware splits and near-duplicate analysis.

### Text leakage

Descriptions may explicitly contain target concepts.

**Response:** document input fields and run text-free ablations.

### Graph degree bias

High-degree concepts may dominate similarity.

**Response:** analyse performance by degree and compare graph models.

### Model pretraining leakage

Foundation models may have seen publicly available heritage images.

**Response:** avoid claims of unseen-image purity; emphasise comparative retrieval and use difficult group/cross-collection splits.

### Epistemic overreach

Embedding proximity cannot prove historical influence, transmission, or intention.

**Response:** phrase outputs as candidate relationships for investigation unless independent evidence exists.

## 17. FAIR and reproducibility strategy

The research object is a bundle of:

- source code;
- environment specification;
- manifests;
- data schemas;
- split definitions;
- exact model identifiers;
- configurations;
- metrics;
- executed notebooks;
- checksums;
- release metadata.

Large source images should normally be reconstructed from stable identifiers or IIIF services, subject to source rights. Large derived embeddings should be stored in an appropriate versioned repository rather than Git.

## 18. Minimum publishable result

A credible paper does **not** require a new neural architecture. A strong result could be:

1. a carefully designed heritage retrieval benchmark;
2. explicit visual/iconographic disagreement sets;
3. strong visual and graph baselines;
4. evidence that graph reranking or fusion improves specific difficult cases;
5. cross-collection validation;
6. expert evaluation;
7. an analysis of when graph structure helps and when it does not.

A well-supported negative result can also be publishable if it demonstrates that currently available KG representations fail to add value under proper controls.

## 19. Potential paper framing

Working framing:

> **Beyond Visual Similarity: Evaluating Graph-Structured Iconographic Knowledge for Cross-Collection Heritage Image Retrieval**

Possible contributions:

1. a controlled benchmark for visual/iconographic disagreement;
2. reproducible comparison of vision, KG, text, and fusion baselines;
3. graph-aware retrieval with interpretable ranking changes;
4. Emblematica case study;
5. cross-collection museum evaluation.

## 20. Long-term extension: architectural ornament

Architectural ornament is better treated as a **discovery corpus** after the method has been validated. In weakly labelled ornament collections, the system can propose visually and semantically motivated correspondences to prints, emblems, paintings, or decorative arts.

Such correspondences are hypotheses for historical research, not automatic evidence of influence.
