# Archive

These documents record a line of work that was abandoned: an intermediate "visual language"
meant to make an engraving and a painting comparable (shape signs, composition, repetition,
pose, palette, named nodes, relations between regions).

They are kept so that what was tried can be checked, not as results. When the decomposition was
finally inspected picture by picture, its units — regions cut by thresholding, relations
between them — did not correspond to what a person sees in the image. The measurements made on
those units therefore do not answer the question they were built for, and the claims these
documents make should not be cited as findings.

- [`VISUAL_RECORD.md`](VISUAL_RECORD.md) — the experiments and their measurements.
- [`VISUAL_LANGUAGE.md`](VISUAL_LANGUAGE.md) — the orientation map of picture languages that
  motivated them.
- [`SHAPE_SIGNS.md`](SHAPE_SIGNS.md) — the clustered shape vocabulary, sign by sign.

One factual error is known: `VISUAL_RECORD.md` says regions are cut by a watershed;
`scripts/segment_shapes.py` thresholds at two polarities and labels connected components.
