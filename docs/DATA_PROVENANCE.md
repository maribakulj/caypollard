# Data Provenance and Rights

## Policy

This repository should not become an untraceable redistribution point for heritage images. For each source collection, preserve provenance and rights information at object level where feasible.

## Manifest requirements

A canonical manifest row should contain as many of these fields as the source permits:

| Field | Description |
| --- | --- |
| `object_id` | stable source identifier |
| `source_institution` | institution responsible for source record |
| `source_record_url` | canonical metadata record |
| `iiif_manifest` | IIIF Presentation manifest when available |
| `iiif_image` | IIIF Image API service/derivative when available |
| `image_url` | fallback image URL |
| `rights` | rights/licence URI or statement |
| `retrieved_at` | retrieval timestamp |
| `sha256` | checksum of local derivative if downloaded |
| `concept_ids` | Iconclass or other controlled vocabulary identifiers |
| `work_id` | work/book/object group for split control |
| `edition_id` | edition/reproduction group where available |
| `collection_id` | source collection partition |
| `split` | frozen experimental split |

## Redistribution

Code is MIT licensed. External data are **not automatically MIT licensed**. Each dataset retains its own terms. Before a release:

1. document source licences;
2. determine whether image derivatives may be redistributed;
3. prefer manifests and retrieval scripts where redistribution is uncertain;
4. never strip institutional attribution or rights metadata;
5. record transformations applied to source assets.

## IIIF

Where IIIF is available, prefer durable IIIF identifiers and explicit requested sizes/regions over ad hoc scraped image URLs. The exact IIIF request used for an experiment should be reproducible.

## Derived data

Embeddings may themselves raise licensing and policy questions depending on source material and repository rules. Treat derived vectors as versioned research artifacts with documented source coverage and avoid assuming they are unrestricted merely because they are numeric.
