# Reproducibility

## Reproducibility target

A researcher should be able to reconstruct the same benchmark definition and rerun the principal experiments from documented source identifiers, configuration, and model versions.

## Required for each experiment

Record:

- git commit SHA;
- Python version;
- dependency lockfile (release gate; regenerate `uv.lock` before tagging);
- hardware/device;
- random seed(s);
- dataset manifest version/checksum;
- split version/checksum;
- model provider/name/revision;
- image preprocessing parameters;
- graph construction parameters;
- embedding normalisation;
- fusion hyperparameters;
- metric configuration.

## Notebook rule

Notebooks should be executable top-to-bottom in a clean environment. Hidden state is a bug.

Notebooks may:

- load data;
- call package functions;
- visualise results;
- explain methods;
- report metrics.

Notebooks should not contain the only copy of:

- dataset parsing logic;
- metric implementations;
- model wrappers;
- reusable training loops;
- core graph construction code.

Those belong in `src/` and should be tested.

## Determinism

Full numerical determinism is not always possible across hardware stacks. The repository should distinguish:

- deterministic data/split construction;
- seeded stochastic experiments;
- hardware-sensitive operations.

Report variance across seeds for learned models rather than pretending a single seed is a law of nature.

## Small sample

CI should operate on a small sample that validates schemas, retrieval code, metrics, and notebook execution without requiring full research data.

## Releases

A research release should freeze:

- manifests;
- split files;
- configs;
- metric definitions;
- model identifiers;
- key result tables;
- executed notebooks;
- software version.

Prefer a DOI-bearing archive for paper-associated releases. A release must not be tagged until the dependency lockfile can be regenerated and CI can reproduce it from network-accessible package indexes.

## Optional learned-alignment environment

The learned projection-head path is intentionally isolated behind the `alignment` extra. CI installs `dev` + `alignment` so the learned notebook and tests are executed rather than silently skipped. Full release environments should be reconstructed from the regenerated lockfile before tagging.
