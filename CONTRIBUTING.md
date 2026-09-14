# Contributing

This repository is currently research-first. Contributions should preserve experimental traceability.

## Before proposing a change

- state which research question or reproducibility problem it addresses;
- avoid mixing refactors with result-changing methodological changes;
- add or update tests for reusable logic;
- document changes to graph construction, data splits, metrics, or preprocessing;
- never commit third-party assets without checking redistribution rights.

## Experiment changes

A pull request that changes reported metrics should include:

- configuration difference;
- affected dataset/split version;
- before/after metrics;
- explanation of whether results are directly comparable.
