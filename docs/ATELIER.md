# The workshop (atelier)

caypollard is an experimental workshop: a place to plug new computer-vision tools into our
corpora, see what they find, and compare them on the same questions. It does not compete with
platforms such as AIKON; it borrows their bricks when useful and measures them.

## Three ideas

- **Bricks.** Every tool does one job: `figures` (picture → one box per figure), `pose`
  (picture + boxes → body points), `vector` (picture → one vector). A brick states its source,
  the size of its weights, and its settings (prompts, thresholds), fixed in
  `src/caypollard/atelier/bricks.py`. A variant with other settings is a new brick with a new
  name, so a setting chosen after seeing a result never hides inside an old name.
- **Benches.** Sets of pictures whose right answers are known in advance
  (`src/caypollard/atelier/benches.py`):
  - `copies`: eight scenes of Holbein's *Dance of Death* in five editions (1526 woodcuts,
    *Icones mortis* 1648, Bewick 1789, Mechel 1803, Hollar 1816 in colour), copies often
    mirrored. A hit is the same scene.
  - `sujets`: one subject across media, 12 emblems against 68 museum works, kept by looking at
    them (the subject must be the main scene). Each side searches the other; a hit shares the
    Iconclass subject.
- **One table.** Each run writes `results/atelier/<bench>/<combination>.json` and rebuilds
  [`results/atelier/TABLEAU.md`](../results/atelier/TABLEAU.md). Intermediate descriptions and
  distance matrices are cached under `data/derived/atelier/` (git-ignored).

## Commands

```sh
uv run python scripts/atelier.py briques          # what can be plugged in
uv run python scripts/atelier.py bancs            # what it is scored on
uv run python scripts/atelier.py pose copies --figures owlv2 --pose vitpose-humanart
uv run python scripts/atelier.py vecteur sujets --brique siglip
uv run python scripts/atelier.py somme sujets pose--opus--opus vector--dinov2
uv run python scripts/atelier.py tableau
```

Bricks over 1 GB in memory, and detectors in general, run through the machine's heavy-jobs
queue: `~/outils-seg/lourd.sh uv run python scripts/atelier.py …` (two at a time at most).

## Adding a brick

Write a class with `boxes(image, item_id, bench)`, `points(image, boxes, item_id, bench)` or
`encode(images)`, register it in `BRICKS` with its summary, source and weight size, and run it
on both benches. Look at a few of its outputs drawn on the pictures before reading its scores.

## Adding a bench

A folder under `data/derived/<name>/` with `images/<id>.jpg` and a `manifest.json` listing
`id`, the label that defines a right answer (`scene` or `subject`) and, for cross-group
questions, a `corpus`; then an entry in `BENCHES`. Say how the pictures were chosen, and what
was left out, in the bench description.

## Reading the numbers

The benches are small (34 and 64 queries): a difference of one or two queries is noise.
Per-label scores are kept in each JSON: a channel can win one subject and lose another (pose
finds every Crucifixion, misses the Visitation, whose meaning is a relation between figures).
