# Literature review protocol

**Version:** 0.1
**Frozen:** 2026-09-14
**Purpose:** define a reproducible search/screening procedure before making originality claims.

This protocol is intentionally separate from [`LITERATURE.md`](LITERATURE.md). The latter is
a curated orientation map; this file specifies how a publication-grade review should be
constructed.

## 1. Review questions

- **LRQ1:** How have images been incorporated into knowledge-graph embeddings or multimodal
  knowledge graphs?
- **LRQ2:** How have graph/ontology representations been used to improve image retrieval,
  classification, or representation learning?
- **LRQ3:** Which of these methods have been evaluated on art, museum, archive, library, or
  other cultural-heritage collections?
- **LRQ4:** How is semantic/iconographic relevance defined and evaluated?
- **LRQ5:** Which studies explicitly compare visual similarity with structured semantic or
  contextual similarity?
- **LRQ6:** Which cultural-heritage systems combine vector retrieval with symbolic KG
  reasoning, and which instead learn a shared embedding space?

## 2. Sources

At minimum, search:

- ACM Digital Library;
- IEEE Xplore;
- SpringerLink;
- Scopus or Web of Science where access permits;
- arXiv for recent/pre-publication work;
- Google Scholar as a broad supplementary discovery source;
- citation chaining from included papers.

Repository/code searches on GitHub and dataset repositories such as Zenodo are supplementary
and should not replace scholarly-database screening.

## 3. Search concepts

The final queries should be adapted to each database syntax while preserving four concept
families.

### A. Graph representation

`"knowledge graph" OR "graph embedding" OR "knowledge graph embedding" OR ontology OR RDF`

### B. Visual / multimodal representation

`image OR visual OR vision OR multimodal OR "vision-language" OR CLIP OR embedding`

### C. Retrieval / representation task

`retrieval OR similarity OR "nearest neighbour" OR representation OR completion OR ranking`

### D. Heritage domain

`"cultural heritage" OR museum OR archive OR library OR art OR artwork OR iconography OR Iconclass`

A broad query combines `(A) AND (B) AND (C)`, while the heritage-focused query adds `(D)`.
Because requiring `D` may miss foundational MMKG work, both searches are retained.

## 4. Time window

- Foundational search: 2010–2026.
- Recent update search: 2024–2026, rerun immediately before submission.

Older work can be included through backward citation chaining when conceptually necessary.

## 5. Inclusion criteria

Include a work when it satisfies at least one of the following:

1. jointly represents or fuses image/visual information and KG/graph structure;
2. uses graph/ontology information to alter image retrieval or visual representation;
3. evaluates multimodal retrieval/reasoning over a cultural-heritage KG;
4. provides a dataset/benchmark directly relevant to art-historical or iconographic
   multimodal evaluation;
5. operationalises art-historical image similarity in a way that informs the research
   question.

## 6. Exclusion criteria

Exclude from the core evidence table:

- papers using "graph" only to mean a neural computation graph;
- generic image-text retrieval with no graph/structured-knowledge relevance, except selected
  domain baselines such as SemArt;
- heritage papers with no image or multimodal component;
- purely descriptive project pages without enough methodological detail;
- duplicate preprint/published versions (retain the final peer-reviewed version and link the
  preprint/code where useful).

## 7. Screening procedure

1. Deduplicate results by DOI/title.
2. Title/abstract screening against the criteria above.
3. Full-text screening.
4. Backward citation chaining for foundational methods.
5. Forward citation chaining for close precedents.
6. Search the names of included datasets/systems separately to locate updates and code.
7. Record an exclusion reason at full-text stage.

For a formal review, a second reviewer should independently screen a sample and disagreement
rates should be reported. If the review is single-author, state that limitation explicitly
rather than conjuring inter-rater reliability from the void.

## 8. Extraction schema

For each included work record:

- bibliographic identifier / DOI;
- year and venue;
- domain and corpus;
- number of images/entities/triples where reported;
- visual representation;
- graph representation;
- text representation;
- fusion stage (early / late / reranking / joint alignment / symbolic reasoning);
- task (retrieval / classification / KGC / enrichment / other);
- supervision / ground truth;
- split strategy and duplicate controls;
- evaluation metrics;
- external validation;
- human/expert evaluation;
- code/data availability;
- main result;
- limitation relevant to this repository;
- relation to our claimed contribution.

## 9. Reproducible artifacts

Before publication, commit or archive:

- exact database/query strings;
- query dates;
- exported result metadata where licences permit;
- deduplication script;
- screening table with decisions;
- extraction table;
- PRISMA-style counts;
- final `references.bib`.

Raw vendor exports that cannot legally be redistributed should be represented by checksums,
query metadata, and transformation scripts instead.

## 10. Originality rule

The repository may describe a contribution as "novel" only after the review has specifically
searched for the relevant combination of **task + representation + evaluation**. Absence of a
paper from a casual search is not evidence of priority, despite the internet's longstanding
attempt to make it so.
