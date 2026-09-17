# Rijksmuseum transfer set — data card

A third corpus, ingested for phase 8 to test whether anything the project measured survives
outside the cataloguing environment it was developed in. It is deliberately the least comfortable
of the three.

## Provenance

| | |
| --- | --- |
| Source of records | [Wikidata](https://www.wikidata.org), works with `collection` (P195) = Rijksmuseum (Q190804) |
| Source of images | [Wikimedia Commons](https://commons.wikimedia.org), via `Special:FilePath` |
| Subject terms | Wikidata `depicts` (P180), mapped to Iconclass through `Iconclass notation` (P1256) |
| Retrieved | 17 September 2026 |
| Builder | `scripts/fetch_rijksmuseum.py`, `scripts/build_rijksmuseum_benchmark.py` |
| Licence | Record text CC0 (Wikidata). Image licences vary by file and are not redistributed here; only URLs and derived hashes are versioned. |

**No API key is used.** The Rijksmuseum's own API requires registration, which would put a manual
step inside an automated pipeline. Wikidata and Commons carry the same works openly.

## Size and shape

| | count |
| --- | ---: |
| Rijksmuseum works with a Commons image | 6 051 |
| … of those, carrying at least one `depicts` statement | 3 727 |
| … of those, depicting something with an Iconclass notation | **1 864** |
| Images downloaded successfully | 1 864 (no failures) |
| Distinct Iconclass notations | 548 |
| Iconclass assignments | 3 455 |
| Mean notations per item | **1.85** |
| Notations occurring once | 249 |
| Distinct creator groups | 1 051 |
| Largest single creator group | 41 works |

For comparison, Emblematica carries 10.68 notations per item and Iconclass AI 4.46. **This corpus
is an order of magnitude thinner in annotation**, which is the first thing any result on it has
to be read against.

## Splits and leakage

Groups are the union of creator, exact image bytes, and confirmed near duplicate, closed
transitively — the creator standing in for what the volume is on the other two corpora, since an
artist's works resemble one another the way plates of one book do. A work with no recorded
creator becomes its own group rather than joining an "unknown artist" bucket that would merge
unrelated hands into one leakage unit.

| split | items |
| --- | ---: |
| train | 1 258 |
| validation | 297 |
| test | 309 |

Ratios are 0.7 / 0.15 / 0.15 rather than the 0.8 / 0.1 / 0.1 used on the larger benchmarks,
because 10% of 1 864 leaves too few test queries to measure anything.

**Zero group leakage and zero checksum leakage.** Near-duplicate detection, the same two-stage
perceptual hash used on Iconclass AI, finds 2 multi-image groups covering 4 images.

Despite the small candidate pool, **all 309 test queries have at least one graded-relevant
candidate** inside the test split, and 277 have an exactly relevant one, so evaluation needs no
widening of the pool.

## What the graph carries

Two projections are written, so the construction that failed on Emblematica is compared rather
than assumed away:

| variant | triples | shape |
| --- | ---: | --- |
| `creator` | 1 907 | items linked to a creator node — the hub shape |
| `pure` | 11 112 | creator, material, genre, object type and decade attached to each item directly |

Predicate counts in `pure`: `made_of` 3 565, `instance_of` 1 990, `created_by` 1 907, `made_in`
1 856, `genre` 1 794. Every item carries at least one attribute.

**`depicts` appears in neither projection.** Those statements are the relevance ground truth, and
putting them in the graph would be target leakage of the most direct kind.

## Caveats a reader must carry

- **Subject terms are crowd-assigned.** Wikidata `depicts` statements are written by volunteer
  editors, not Iconclass specialists. They are sparser, shallower and less consistent than either
  of the two curated corpora, and the Iconclass notation attached to a Wikidata entity describes
  the *entity*, not the work's iconographic programme.
- **The alignment is partial by construction.** Only 4 125 Iconclass notations have a Wikidata
  entity carrying P1256, so a work depicting something outside that set is invisible to this
  corpus even if the Rijksmuseum catalogued it richly.
- **Selection is not random.** A work enters only if an editor added a `depicts` statement *and*
  the depicted entity happens to be aligned to Iconclass. Both steps favour recognisable,
  frequently catalogued subjects; the most common notations are `25G3` (trees, 500 works) and
  `31D15` (old age, 435).
- **Images are thumbnails**, negotiated by `Special:FilePath` at roughly 960 pixels, not archival
  masters. Every encoder resizes to 224 pixels, so this costs nothing in representation, but the
  corpus cannot be reused for work that needs full resolution.
- **The corpus is 90% paintings.** Object types, by Wikidata `instance of`: painting 1 685,
  drawing 57, watercolour 51, print 18, pastel 18, etching 17, chalcography 14, triptych 13,
  engraving 12, relief 10. Only two type strata are large enough to measure separately, so the
  "by object type" breakdown phase 8 asks for is really paintings against everything else.
- **This is a transfer probe, not a benchmark of the Rijksmuseum collection.** 1 864 of its
  ~700 000 objects, chosen by the accidents above.
