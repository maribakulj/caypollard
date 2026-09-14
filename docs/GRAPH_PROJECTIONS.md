# Graph projections and target-leakage policy

The project must distinguish **graph topology used as experimental input** from
**expert structure used as evaluation ground truth**. Collapsing those roles
would make KG performance partly tautological.

## Projection G0 — Iconclass taxonomy control

**Nodes:** Iconclass concepts.  
**Edges:** `skos:broader` / parent-child hierarchy.  
**Image representation:** mean pooling of the embeddings of an image's assigned
Iconclass concepts.

This projection is implemented as a pipeline sanity check. Because the image
representation directly contains the same Iconclass assignments used to define
hierarchical relevance, G0 is an **oracle/control condition**. It may verify that
graph information can flow through the code and establish an upper structural
reference, but it cannot support the main claim that a KG independently improves
iconographic retrieval.

## Projection G1 — Context-only heritage graph

The main graph condition should represent historical/contextual relations while
excluding the target iconographic labels for evaluated images.

Candidate entity types:

- image / pictura;
- emblem or artwork;
- book / edition;
- creator / author / engraver;
- printer / publisher;
- place;
- collection / holding institution;
- controlled object types or non-target classifications.

Candidate relations:

- `depictsImageOf` / `hasPictura`;
- `partOf`;
- `createdBy`;
- `printedBy`;
- `publishedAt`;
- `heldBy`;
- `sameEditionAs` or other provenance-safe equivalence links where justified.

**Excluded from the main G1 input:** direct test-image → target Iconclass edges,
free-text captions/descriptions, generated captions, and any feature derived from
the evaluation labels.

Text remains a separate modality (`T`) rather than being smuggled into `G`.

## Projection G2 — Masked-label graph

A complementary experiment may use Iconclass elsewhere in the graph while
masking target edges for evaluated images.

For every test image:

1. remove its direct target Iconclass edges before fitting or message passing;
2. keep graph information available from training/validation objects according
   to the declared transductive or inductive regime;
3. evaluate retrieval against the hidden expert labels;
4. record exactly which edges were masked in a versioned artifact.

This tests whether graph structure can reconstruct useful iconographic
neighbourhoods without simply reading the answer attached to each test object.

## Projection G3 — Rich historical graph

After Emblematica ingestion, G3 combines contextual entities and relations with
controlled semantic links under an explicit ablation matrix:

| Variant | Context relations | Iconclass elsewhere | Test target edges | Text |
| --- | :---: | :---: | :---: | :---: |
| G1-context | yes | no | no | no |
| G2-masked | yes | yes | **masked** | no |
| G3-full-control | yes | yes | yes | no |
| T | no | no | no | yes |

`G3-full-control` is reported only as an oracle/upper-reference condition, never
as the main evidence for graph benefit.

## Dates and literals

KGE models operate on entity-relation triples, so dates should not be converted
into thousands of arbitrary day entities without a reason. Initial experiments
should compare:

- raw date literals retained outside KGE;
- decade or period nodes only where historically meaningful;
- explicit temporal models only if temporal reasoning becomes a research
  question rather than an implementation ornament.

## Evaluation consequences

The headline multimodal comparison should therefore be:

`V` vs `G1-context` / `G2-masked` vs `V+G1` / `V+G2`.

`G0-taxonomy` and `G3-full-control` are sanity/oracle conditions. Improvements
from those conditions cannot by themselves demonstrate that structured
historical knowledge adds independent information to visual retrieval.
