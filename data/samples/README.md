# Sample fixtures

`iconclass_testset_excerpt.json` reproduces the four-record `data.json` excerpt shown on the official Iconclass AI Test Set documentation page. It contains filenames and classification notations only, not image bytes.

`iconclass_notations_fixture.txt` is a compact structural fixture built from the documented path for `25G41` (`2 → 25 → 25G → 25G4 → 25G41`) and the children shown in the official Python example. It exists solely to exercise the parser and hierarchy metrics in CI. It is **not** a substitute for the full vocabulary.

Sources checked 2026-09-14:

- https://iconclass.org/testset/
- https://github.com/iconclass/data
