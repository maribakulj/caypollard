# Literature map

This document positions the repository against work on multimodal knowledge graphs,
computational art history, iconographic representation, and cultural-heritage retrieval.
It is a **curated research map**, not yet a systematic review. Before a paper submission,
the search must be rerun with a documented protocol, databases, query strings, screening
criteria, and an archived result set.

The working bibliography is also available as [`references.bib`](../references.bib).

## 1. What is already established

The broad idea "combine images and graph representations" is not novel. The literature
already establishes at least four relevant lines of work:

1. **multimodal knowledge-graph representation/completion**, where visual features are
   incorporated into entity representations;
2. **art-specific multimodal representation**, often aligning images with text or metadata;
3. **knowledge-aware visual modelling**, where graph embeddings enrich image tasks;
4. **cultural-heritage hybrid retrieval**, where vector search and symbolic reasoning are
   combined in operational or near-operational systems.

The repository therefore does **not** claim novelty for image + KG fusion itself. Its target
is narrower: controlled evaluation of whether graph-structured iconographic knowledge
changes visual neighbourhoods in useful ways, especially under visual/iconographic
**disagreement** and cross-collection transfer.

## 2. Multimodal knowledge-graph representation

### Xie et al. (2017), IKRL

Ruobing Xie, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. **Image-embodied Knowledge
Representation Learning.** *IJCAI 2017*, pp. 3140–3146.
DOI: [10.24963/ijcai.2017/438](https://doi.org/10.24963/ijcai.2017/438).

IKRL is an important early baseline: it learns knowledge representations jointly from KG
triples and associated entity images, then evaluates them on knowledge-graph completion and
triple classification. It demonstrates that visual information can improve graph-oriented
representation tasks.

**Relation to this repository:** foundational precedent for multimodal KGE, but the task here
is reversed in emphasis. We ask whether graph structure improves **heritage image retrieval
and semantic neighbourhoods**, rather than primarily whether images improve KGC.

### Chen et al. (2022), MKGFormer

Xiang Chen et al. **Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge
Graph Completion.** *SIGIR 2022*, pp. 904–915.
DOI: [10.1145/3477495.3531992](https://doi.org/10.1145/3477495.3531992).

MKGFormer integrates textual, visual, and structural information using a hybrid transformer
and multi-level fusion for several multimodal KG tasks.

**Relation to this repository:** a useful advanced architecture reference. It is intentionally
*not* the starting baseline here. A learned architecture is only justified after transparent
late-fusion and reranking baselines demonstrate complementary graph signal.


## 2a. Graph-embedding methods used as baselines

The repository's graph baselines are intentionally conventional. DeepWalk (Perozzi et al.,
2014) established truncated random walks plus language-model-style representation learning;
node2vec (Grover & Leskovec, 2016) introduced biased walks controlled by return/in-out
parameters; RDF2Vec (Ristoski & Paulheim, 2016) adapted walk/language-model ideas to RDF
graphs; ComplEx (Trouillon et al., 2016) introduced complex-valued factorisation capable of
representing asymmetric relations; and RotatE (Sun et al., 2019) models relations as rotations
in complex space.

Caypollard uses two lightweight CI controls, `node2vec-style-ppmi-svd` and
`rdf2vec-style-ppmi-svd`. They factorise positive-PMI co-occurrence matrices from deterministic
walk corpora instead of claiming to reproduce the original Word2Vec optimisation. Canonical
relation-aware ComplEx/RotatE runs are delegated to the optional PyKEEN backend and versioned
separately. This naming distinction is deliberate: methodological convenience should not
quietly become bibliographic fiction.

## 3. Multimodal representation for art

### Garcia & Vogiatzis (2018), SemArt

Noa Garcia and George Vogiatzis. **How to Read Paintings: Semantic Art Understanding with
Multi-Modal Retrieval.** *ECCV Workshops 2018*.
Open-access paper: [CVF](https://openaccess.thecvf.com/content_eccv_2018_workshops/w13/html/Garcia_How_to_Read_Paintings_Semantic_Art_Understanding_with_Multi-Modal_Retrieval_ECCVW_2018_paper.html).
Preprint: [arXiv:1810.09617](https://arxiv.org/abs/1810.09617).

SemArt pairs paintings with attributes and catalogue-like textual comments and introduces a
cross-modal Text2Art retrieval task in a common semantic space.

**Relation to this repository:** establishes art-domain multimodal retrieval as a meaningful
evaluation problem. It is particularly relevant to later text ablations (`T`, `V+T`,
`G+T`, `V+G+T`).

### Castellano et al. (2022), ArtGraph

Giovanna Castellano, Vincenzo Digeno, Giovanni Sansaro, and Gennaro Vessio.
**Leveraging Knowledge Graphs and Deep Learning for automatic art analysis.**
*Knowledge-Based Systems* 248 (2022), 108859.
DOI: [10.1016/j.knosys.2022.108859](https://doi.org/10.1016/j.knosys.2022.108859).
Dataset: [ArtGraph on Zenodo](https://zenodo.org/records/8172374).

ArtGraph builds an art KG from WikiArt and DBpedia and injects graph embeddings as
contextual knowledge into deep visual models for artwork attribute prediction.

**Relation to this repository:** one of the closest precedents for combining visual and graph
representations in art. The distinction is task and evaluation: our central object is
**retrieval geometry and iconographic relatedness**, not artwork attribute classification.

## 4. Iconclass, hierarchy, and iconographic representation

### Labusch & Neudecker (2023)

Kai Labusch and Clemens Neudecker. **Gauging the Limitations of Natural Language Supervised
Text-Image Metrics Learning by Iconclass Visual Concepts.** *HIP 2023*, pp. 19–24.
DOI: [10.1145/3604951.3605516](https://doi.org/10.1145/3604951.3605516).

The paper fine-tunes contrastive text-image similarity models using Iconclass concepts and
explicitly asks whether learned image metrics capture iconographic meaning.

**Relation to this repository:** extremely close methodological precedent. It motivates the
use of expert Iconclass annotations as a controlled semantic benchmark while also warning
that natural-language supervision and iconographic structure must not be conflated.

### Springstein et al. (2024), Visual Narratives

Matthias Springstein et al. **Visual Narratives: Large-Scale Hierarchical Classification of
Art-Historical Images.** *WACV 2024*.
DOI: [10.1109/WACV57701.2024.00705](https://doi.org/10.1109/WACV57701.2024.00705).
Open-access paper: [CVF](https://openaccess.thecvf.com/content/WACV2024/html/Springstein_Visual_Narratives_Large-Scale_Hierarchical_Classification_of_Art-Historical_Images_WACV_2024_paper.html).

The work introduces a large art-historical image dataset with more than 20,000 Iconclass
concepts and treats iconographic prediction explicitly as **hierarchical multi-label
classification**.

**Relation to this repository:** reinforces two design choices: Iconclass must be treated as a
hierarchy rather than a flat tag set, and evaluation should account for graded semantic
proximity. Our task remains retrieval rather than classification.

## 5. Computational art-historical comparison

### Impett & Süsstrunk (2016), Warburg / Pathosformel

Leonardo Impett and Sabine Süsstrunk. **Pose and Pathosformel in Aby Warburg's Bilderatlas.**
*Computer Vision – ECCV 2016 Workshops*, LNCS 9913, pp. 888–902.
DOI: [10.1007/978-3-319-46604-0_61](https://doi.org/10.1007/978-3-319-46604-0_61).

The study operationalises a specific Warburgian relation through human pose, using annotated
2D body configurations and clustering to explore recurring expressive formulas.

**Relation to this repository:** conceptual precedent for formalising an art-historical mode
of comparison. It is **not** a KG-embedding precedent. Our proposed question is broader:
whether structured iconographic knowledge can alter computational image neighbourhoods
beyond surface/formal resemblance.

## 6. Cultural-heritage multimodal knowledge systems, 2026

The 2026 literature makes it especially important not to oversell novelty. Several systems
now combine KGs, images, vector retrieval, and language models in cultural heritage.

### Blanco et al. (2026), ArtKB

Giacomo Blanco et al. **ArtKB: A Multimodal Art Knowledge Base for Cultural Heritage.**
In *The Semantic Web – ESWC 2026*, LNCS 16550, pp. 58–75.
DOI: [10.1007/978-3-032-25159-6_4](https://doi.org/10.1007/978-3-032-25159-6_4).

ArtKB combines semantic graph infrastructure, visual/vector representations, and digital
asset storage to support hybrid retrieval, enrichment, and text-to-graph workflows.

**Relation to this repository:** infrastructure precedent. Our intended contribution is not a
new storage architecture; it is an experimentally controlled claim about representation and
retrieval quality.

### Duan et al. (2026), knowledge-enhanced cultural-heritage retrieval

Xuemin Duan, Federico D'Asaro, Ruben Peeters, Giacomo Blanco, Tommaso Monopoli,
Giuseppe Rizzo, and Anastasia Dimou. **Knowledge-Enhanced Multimodal Retrieval over
Cultural Heritage Knowledge Graphs.** In *The Semantic Web – ESWC 2026*, pp. 488–506.
DOI: [10.1007/978-3-032-25159-6_26](https://doi.org/10.1007/978-3-032-25159-6_26).
Code: [REEVALUATE/knowledge_enhanced_multimodal_retrieval](https://github.com/REEVALUATE/knowledge_enhanced_multimodal_retrieval).

The system combines a domain-adapted CLIP retriever with LLM-based Text2SPARQL and a
fusion stage, and evaluates the system quantitatively and with heritage professionals.

**Relation to this repository:** probably the closest current retrieval-system precedent.
Its symbolic component performs query/reasoning and result fusion; our central experiment
asks whether **embedded graph structure itself changes image similarity/neighbourhoods** and
whether that effect survives leakage-controlled, cross-collection evaluation.

### Zhang et al. (2026), WJoconde

Yang Zhang, Nada Mimouni, Jean-Claude Moissinac, and Fayçal Hamdi.
**Multimodal Cultural Heritage Knowledge Graph Extension with Language and Vision Models.**
[arXiv:2605.17669](https://arxiv.org/abs/2605.17669).

The paper introduces WJoconde, a multimodal French cultural-heritage KG integrating text and
images, plus variants and a KGC benchmark, and uses LLM/VLM extraction to extend the graph.

**Relation to this repository:** valuable secondary benchmark and French heritage reference.
Its main direction is **multimodal KG extension/completion**; ours is graph-informed heritage
retrieval.

### Zhang et al. (2026), Beyond Images

Pengyu Zhang, Klim Zaporojets, Jie Liu, Jia-Hong Huang, and Paul Groth.
**Are a Thousand Words Better Than a Single Picture? Beyond Images — A Framework for
Multi-modal Knowledge Graph Dataset Enrichment.** In *The Semantic Web – ESWC 2026*,
pp. 82–101.
DOI: [10.1007/978-3-032-25156-5_5](https://doi.org/10.1007/978-3-032-25156-5_5).

The work studies multimodal KG enrichment and, importantly, shows that converting ambiguous
visual material into textual descriptions can outperform naïve visual use in some KGC
settings.

**Relation to this repository:** strong warning against assuming that an additional modality
necessarily adds useful information. This is why the protocol requires explicit modality
ablations and transparent baselines.

## 7. The gap this repository tests

Taken together, the literature already supports all of the following statements:

- image information can improve KG representations;
- visual and textual art representations can be aligned for retrieval;
- graph embeddings can enrich visual art models;
- Iconclass can supervise and evaluate iconographic image representations;
- cultural-heritage systems can combine vector retrieval and symbolic KG reasoning.

The remaining research space is therefore **not** "multimodal cultural heritage" in the
abstract. The repository tests the narrower hypothesis:

> **Graph-structured iconographic knowledge provides complementary information to visual
> representations, improving retrieval specifically when formal visual similarity and
> iconographic similarity diverge.**

The strongest version of the claim additionally requires:

1. leakage-controlled splits rather than random image splits;
2. hierarchical, multi-label relevance rather than flat exact-match accuracy alone;
3. hard positives and hard negatives selected independently of the tested fusion model;
4. visual-only, graph-only, text-only, and fused ablations;
5. external transfer from the controlled Iconclass benchmark to historically richer
   collections such as Emblematica and Rijksmuseum;
6. expert evaluation for historical usefulness, because taxonomy distance is not identical
   to art-historical significance.

## 8. Claims the project should avoid

Unless a later systematic review demonstrates otherwise, do **not** claim:

- first multimodal knowledge graph for cultural heritage;
- first use of graph embeddings with art images;
- first knowledge-enhanced multimodal cultural-heritage retrieval system;
- first computational use of Iconclass for image representation;
- that embedding similarity establishes historical influence.

A defensible contribution is instead a **reproducible benchmark, evaluation protocol, and
retrieval method for testing graph-conditioned iconographic similarity across heritage
collections**.
