# Viewer

A local page that shows the whole pool as thumbnails and, for any one picture,
the neighbours each representation returns. It is a way of looking at what the
benchmarks aggregate; it computes nothing and annotates nothing.

```
make viewer-build   # once, and again when a table changes (labels, thumbnails, neighbours)
make viewer         # serves http://127.0.0.1:8765/viewer/ from the repository root
```

What it reads, all under `data/derived/viewer/` and written by
`scripts/build_viewer.py`: one thumbnail per picture, `items.json` with the
attributes the regimes read (corpus, collection, kind, genre, century, creator,
material, notations, motto, named nodes), and per representation the hundred
nearest rows with their cosine similarity. Wikidata ids are labelled by
`scripts/fetch_wikidata_labels.py`, which keeps its answers in
`data/derived/wikidata-labels.json` and never refetches one.

In the page: filter the grid by corpus, collection, kind, genre or century, or
search any text; click a picture for its record; choose a representation and
how many neighbours; the chips restrict the list to a regime (same subject on
another kind of object, another collection, another century…) and each
neighbour says what it shares with the query. Clicking a neighbour makes it the
query; `←` goes back, `Échap` closes, the arrow keys move along the grid. The
picture's id is in the URL, so a link reopens the same record.

## Sous le capot

The button of that name (or the link in a picture's record) opens the demo engine. Drop
any image, or take the selected picture of the pool, and the page runs it through every
representation the experiments used, in the order they are built: the medium-invariant
renders, the strokes (contour drawing, strokes, sketch of forty, pictogram of twelve), the
DINOv2 embedding, the cut into regions and the 34-number description of each,
the shape signs and the relations between them, composition, repetition, the composite
signs and the record, palette and signal, the named nodes, the pose, the mix. Each step shows what it produced (images, tables, bars),
says which script and parameters it ran with, whether those are the pool's parameters, and,
for a picture of the pool, the cosine between the live vector and the frozen row: 1.0000 is
the proof that the demo computes what the benchmarks measured. The parameters can be edited
and the analysis rerun; a change that alters a vector's dimension is flagged as incomparable
to the pool.

The last step is the search: presets by regime (image, symbolique, sémantique, les deux,
surface, mix) or a weight per channel, exact cosine over the pool, one column of results
per active channel and one for the weighted mix. Clicking a result opens its record.

The named nodes are produced by Claude (Sonnet) through the Claude Code command line in
non-interactive mode, so the machine must have `claude` installed and signed in; the frozen
tables were named by a Mistral model with the same question, and the reproduction cosine on
that step shows how differently the two namers see. Relations between signs and pose are shown too,
with the measurements that argue against them on their card: the grammar is refused three
times in the record, and the pose detector finds a usable figure in 28% of pictures, so its
search table covers 1 903 pictures only.

Behind the page: `src/caypollard/demo/` (the pipeline, the pool search, the namer, the
explanations) and `scripts/fit_demo_models.py`, which refits the k-means vocabularies the
frozen tables were built with — their centres were never saved — and verifies each against
its table before writing it to `data/derived/demo/`.
