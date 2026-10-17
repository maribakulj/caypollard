---
license: cc-by-nc-4.0
task_categories:
- zero-shot-object-detection
language:
- en
tags:
- art
- visual-grounding
- cross-modal-retrieval
- cultural-heritage
- computational-art-analysis
pretty_name: RichArt
size_categories:
- 1K<n<10K
---

# RichArt

RichArt is a benchmark dataset of **7,087 paintings** with **20,882 region-level annotations** pairing bounding boxes with rich symbolic narratives extracted from museum catalogs. It is introduced in *"Fine-Grained Cross-Modal Retrieval in Art via Region-Level Grounding of Symbolic Narratives"* (ICMR 2026).

## Dataset Structure

```
├── annotations/
│   ├── train.json   
│   ├── val.json     
│   └── test.json    
└── data/
    ├── train/ (80% — 5,669 paintings)
    ├── val/   (10% —   709 paintings)
    └── test/  (10% —   709 paintings)
```

## Statistics

#### General Aspects

| Metric | Value |
|---|---|
| Paintings | 7,087 |
| Artists | 2,288 |
| Annotated objects | 20,882 |
| Unique object labels | 6,233 |
| Century range | 14th – 21st |

**Source Distribution:**
- 76.5% Web Gallery of Art 
- 12.9% Metropolitan Museum of Art
- 10.6% WikiArt

#### Annotation Granularity

**Object Density:** Median of 2 objects per painting (max 8).

**Narrative Depth:** Region descriptions average 35 words, while painting-level descriptions range from 26 to 8,573 words (Mean: 212, Median: 123).

#### Taxonomy & Style

**Categorization:** 81.89% of paintings are classified into 7 coarse types (e.g., religious, portrait, landscape) and 25 fine-grained subcategories.

**Artistic Style:** 40.77% of the corpus is tagged with one of 33 styles (e.g., Baroque, Realism, Tenebrism).

## Visual Examples

#### Structured Painting Metadata
 
![Example entry showing the painting metadata schema: title, artist, year, coarse type, style, source, and description](painting_example.png)

Example of a painting and its metadata (including text highlights of a referring expression and attached description)
 
#### Region-Level Symbolic Annotation
 
![Region-level annotation example showing bounding boxes with symbolic narratives on a Dalí painting](annotation_example.png)
 
Region-level annotation example showing bounding boxes with symbolic narratives on a Dalí painting

#### Diversity of Annotated Object Regions
 
![Grid of diverse annotated object crops from across the RichArt dataset](image_objects_examples.png)
 
Grid of diverse annotated object crops from across the RichArt dataset 

## Annotation Format

The annotations are provided in JSON format. Each entry represents a single painting and contains high-level metadata alongside region-level "symbolic narratives" grounded in the image.

#### Schema Definition

| Field | Type | Description |
| :--- | :--- | :--- |
| **`id`** | int | Unique identifier for the painting. |
| **`title`** | string | The official title of the artwork. |
| **`artist`** | string | The name of the artist. |
| **`year`** | int | The year of creation. |
| **`coarse_type`** | string \| null | High-level thematic category (e.g., `religious`, `portrait`). |
| **`first_fine_grained_type`** | string \| null | Primary specific subcategory within the coarse type. |
| **`second_fine_grained_type`** | string \| null | Secondary subcategory, if applicable. |
| **`first_style`** | string \| null | Primary artistic style (e.g., `baroque`, `early renaissance`). |
| **`second_style`** | string \| null | Secondary artistic style, if applicable. |
| **`description`** | string | The comprehensive narrative text for the painting from the source museum catalog. |
| **`source`** | string | The origin of the data (e.g., `met`, `wga`, or `wikiart`). |
| **`height`** | int | Height of the digital image in pixels. |
| **`width`** | int | Width of the digital image in pixels. |
| **`objects`** | dict | **The core annotation object.** Map of symbolic entity names to their groundings. |
| **`objects[key].description`** | string | The specific narrative extracted from the painting description tied to the identified region. |
| **`objects[key].bounding_boxes`** | list | A list of groundings: `[score, [x1, y1, x2, y2]]`. |

#### Generic Example

The following example illustrates a typical entry. Note that a single symbolic entity can be linked to multiple bounding boxes if that element appears in different parts of the painting.

```json
{
    "id": 123,
    "title": "Example Artwork Title",
    "artist": "Artist Name",
    "year": 1500,
    "coarse_type": "religious",
    "first_fine_grained_type": "altarpiece",
    "second_fine_grained_type": null,
    "first_style": "renaissance",
    "second_style": "manierism",
    "description": "A comprehensive description of the entire painting, including historical context...",
    "source": "met",
    "height": 1200,
    "width": 900,
    "objects": {
        "symbolic_entity_name": {
            "description": "A specific narrative excerpt describing this particular element.",
            "bounding_boxes": [
                [
                    0.85, 
                    [100.5, 200.0, 300.2, 450.7]
                ],
                [
                    0.72, 
                    [500.0, 210.5, 650.3, 400.1]
                ]
            ]
        }
    }
}
```

#### Key Considerations

**ID Value:** Some paintings have IDs with values larger than 7,087 because the IDs from the initial set of 12,078 paintings, which have been filtered, have been kept. 

**Symbolic Narratives:** Unlike standard object detection (literal labels like "person"), RichArt uses descriptive phrases (e.g., "the nervous behavior of the hands") to enable fine-grained retrieval.

**Coordinates:** Bounding boxes use absolute pixel format: [x_min, y_min, x_max, y_max].

**Confidence Scores:** The first value in the bounding box array represents an alignment score from the grounding process provided by GroundingDINO.

## Methodology & Creation

Detailed information regarding the data collection pipeline, annotation methodology, and region grounding verification can be found in the associated research paper and in the [corresponding repository](https://github.com/Bindila-Bogdan/Fine-Grained-Cross-Modal-Retrieval-in-Art).