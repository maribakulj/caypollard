# Emblematica Online data card

## Why a second corpus

The Iconclass AI Test Set is documented by its own `index.html` as "sampled from the Arkyves
database" — a Brill subscription resource. Its filenames therefore carry Arkyves-internal
collection codes with no public key back to any library catalogue. Book-level provenance could
be recovered for 16.7% of images by reading shelfmarks out of filenames, but the relations that
matter for H2 could not: bibliographic enrichment from 505 Munich IIIF manifests reached
**0 of 250** hard positives, under both relevance policies and both mining encoders.

That was an identification failure, not an absence of relations. Hard positives in that corpus
are emblem-book pairs drawn across St Andrews, UIUC, Wolfenbüttel, Mnemosyne and Utrecht, and
those books do share creators — they simply could not be resolved.

Emblematica Online removes the obstacle instead of working around it: one open corpus carrying
images, Iconclass notations, emblem texts, and a resolvable book identifier for the same object.

## Source and access

- API base: `http://emblematica.library.illinois.edu` (HTTP only; the HTTPS host does not respond)
- Emblem search: `/api/Emblem/Search` — **ignores the `Take` parameter and returns a fixed page
  of 18**, so a full index walk is roughly 1 850 requests
- Emblem detail: `/api/Emblem?id=<emblemID>` — gives `bookID`, `CollectionCode`, `SpineXML`
- Book search: `/api/Book/Search` — 1 407 volumes with title, authors, place, date
- Emblem record XML: `http://emblemimages.library.illinois.edu/<bookID>/emblematica/emblem<NNNNNN>.xml`

The book identifier is embedded in the pictura URL returned by the search endpoint, so it can be
read there instead of costing one detail request per emblem.

## Conditions of use, and what they require of us

The project's stated conditions (`help/conditions`) permit exactly the use made here:

> Items included in the Emblematica Online Digital Virtual Collection may be used freely for
> non-publication, non-commercial applications such as personal research, classwork, or other
> private use.

Two obligations follow, and both bind this repository.

**Acknowledgement is required, per contributing library.** The stated form is "Courtesy of the
Emblematica Online Digital Collection and the ___ Library of the ___". Because the corpus spans
Wolfenbüttel, Illinois, Glasgow, Utrecht, Duke and Getty, every derived artifact retains the
`collection` field so the right institution can be named rather than a generic credit.

**Publication is not covered by the blanket permission.** Items "may be subject to additional
conditions of use, according to the policies of the institutions holding the physical items".
Reproducing picturae in a paper, or redistributing the image cache, therefore requires checking
each holding institution separately. Derived vectors and metrics are unaffected; images are not.

## Request rate

Neither Emblematica host serves a `robots.txt`, and no stated rate limit, crawl delay, or
automated-access policy could be found. That silence is not permission, so ingestion is
deliberately bounded rather than tuned for speed: at most three concurrent connections with a
per-connection delay, which keeps the load at or below the parallelism a single ordinary browser
opens against one host. Requests carry a descriptive User-Agent, and every stage caches to disk
so a re-run costs the server nothing.

## Scale

| | count |
| --- | ---: |
| emblems indexed | 33 233 |
| volumes in the catalogue | 1 407 |
| volumes flagged as carrying emblems | 368 |

By holding library: HAB 13 683, Illinois 11 516, Glasgow 5 594, Utrecht 2 192, Duke 132, Getty 84.
By period: 17th century 18 267, 18th century 7 277, 16th century 7 135, 19th century 294.

## What each emblem carries

- **Iconclass notations**, with bracketed text keys intact (`25F23(BEAR)`), which is the form
  protocol v0.5 requires;
- **motto**, transcribed, usually Latin;
- **subscriptio**, where transcribed;
- **figDesc**, a cataloguer's prose description of the pictura (`"A bear breathing smoke."`);
- **pictura image** URL.

Motto and subscriptio give this project its first real text modality — three of the seven
conditions in the model matrix (`T`, `V+T`, `G+T`) had no data source before.

## Known limitations

### Records are not in one dialect

Contributing libraries do not share a house style. Glasgow writes unprefixed elements under a
default namespace with `skos:notation`; Wolfenbüttel writes `emblem:`-prefixed elements with
`tei:p`. The parser matches prefix-agnostically rather than assuming one form — a parser written
against a single sample silently returns empty notations for the other library.

### Iconclass coverage varies by collection

Coverage is measured at ingestion rather than assumed. On an early partial ingest, Illinois was
labelled 294/294 and Wolfenbüttel 584/985. Emblems without notations cannot enter a
relevance-based evaluation and are excluded from the manifest by default, so the usable corpus is
smaller than the indexed one and the audit reports both figures.

### Creator matching is by surname, not authority identifier

The book catalogue supplies no GND or VIAF identifier and no printer. Creators are therefore
matched on slugged surnames, which will merge distinct people who share one. This is weaker than
the identifier matching available for the Munich volumes and is accepted deliberately: the
alternative is no creator edge at all, which removes the only relation capable of linking two
emblem books by one author — exactly what the hard-positive question needs.

### Work titles are a crude stand-in

No normalised work title is exposed, so `instance_of` uses the first three words of the title.
Two editions catalogued with different opening words will not link. Measured on emblem books,
`instance_of` connects 79 of 368 volumes against 272 for `created_by`.

## Why this corpus can answer H2

74% of emblem books (272 of 368) share a creator with another emblem book, and 69% share a place
of publication. Alciati appears across 26 volumes, de Bry 14, Harsdörffer 8. Cross-volume paths
therefore exist in the graph before any modelling, which was not true of the first corpus at all.

## Graph variants, and why the obvious one fails

The first graph built here reproduced the failure it was meant to fix: 100% of an emblem's
top-5 graph neighbours came from its own volume, exactly as in the Iconclass corpus.

The cause is structural, not a coverage gap. The obvious graph holds 25 463 `part_of` and
25 199 `adjacent_to` edges against 1 765 bibliographic ones — 29 to 1 — and a book node is a
hub with roughly 96 emblem children. A walk that steps up to the book almost always steps back
down to a sibling; reaching another volume requires `book → person → book`, and person nodes
carry far lower degree. The contextual relations are present and unreachable.

Two changes make them legible. Book attributes are **projected onto the emblems themselves** —
an entailment of `part_of` plus the book's relation, not a new claim — and the within-book
edges are dropped. Measured on 589 same-author and 589 different-author book couples, sampled
three emblem pairs each:

| variant | edges | same-author score | other | AUC |
| --- | --- | ---: | ---: | ---: |
| `book` (`part_of` + `adjacent_to`; the Iconclass-style graph) | 50 662 | +0.0752 | +0.0752 | **0.457** |
| `full` (everything) | 52 427 | +0.2132 | +0.0830 | 0.620 |
| `pure` (projected attributes, no within-book edges) | 152 244 | +0.3461 | +0.0168 | **0.963** |

The `book` variant is at chance, which is the Iconclass result reproduced on a corpus that does
contain the relations — proof that the earlier failure was about graph construction as much as
about the source. The `pure` variant separates the same pairs at 0.963.

One property of `pure` must be stated rather than discovered later: emblems of one volume share
every attribute, so they are structurally identical in it and their mutual similarity ties at
the top. Its top-5 neighbour list is therefore still dominated by same-volume siblings; the
signal lives in the score, not in the ranking of tied siblings. That is acceptable for fusion,
where a visual score breaks the tie, and it is why neighbour composition is the wrong statistic
for judging this graph.

## Generated artifacts

`scripts/fetch_emblematica.py` caches the index, one XML per emblem, and picturae for labelled
emblems only. `scripts/build_emblematica_benchmark.py` then produces `manifest.jsonl`,
`context_triples.tsv` and `audit.json` using the same field names as the Iconclass pipeline, so
auditing, splitting, retrieval, fusion and hard-pair mining run unchanged across both corpora.
