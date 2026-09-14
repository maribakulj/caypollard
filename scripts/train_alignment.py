#!/usr/bin/env python3
"""Train small visual/KG projection heads without touching the test split during fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caypollard.alignment import AlignmentConfig, embedding_collapse_diagnostics, train_joint_alignment
from caypollard.embeddings.store import load_embedding_table, save_embedding_table
from caypollard.provenance import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual_embeddings")
    parser.add_argument("graph_embeddings")
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    visual = load_embedding_table(args.visual_embeddings)
    graph = load_embedding_table(args.graph_embeddings)
    records = read_jsonl(args.manifest)
    common = set(visual.ids).intersection(graph.ids)
    train_ids = sorted(
        str(record["id"])
        for record in records
        if str(record.get("split")) == args.train_split and str(record["id"]) in common
    )
    validation_ids = sorted(
        str(record["id"])
        for record in records
        if str(record.get("split")) == args.validation_split and str(record["id"]) in common
    )
    if len(train_ids) < 2 or len(validation_ids) < 2:
        raise ValueError("at least two aligned train and validation items are required")

    config = AlignmentConfig(
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    run = train_joint_alignment(
        visual,
        graph,
        train_ids=train_ids,
        validation_ids=validation_ids,
        output_ids=sorted(common),
        config=config,
        device=args.device,
    )
    visual_table, graph_table, joint_table = run.as_tables()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("projected-visual.npz", visual_table),
        ("projected-graph.npz", graph_table),
        ("joint-mean.npz", joint_table),
    ):
        save_embedding_table(
            output_dir / name,
            ids=table.ids,
            vectors=table.vectors,
            metadata=table.metadata,
            normalize=False,
        )

    diagnostics = {
        "projected_visual": embedding_collapse_diagnostics(visual_table),
        "projected_graph": embedding_collapse_diagnostics(graph_table),
        "joint_mean": embedding_collapse_diagnostics(joint_table),
    }
    report = {
        "protocol": "protocol-v0.4",
        "metadata": run.metadata,
        "history": list(run.history),
        "collapse_diagnostics": diagnostics,
        "note": "No test labels or test retrieval metrics are used during fitting.",
    }
    (output_dir / "alignment-run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": run.metadata["best_epoch"],
                "best_validation_loss": run.metadata["best_validation_loss"],
                "output_id_count": run.metadata["output_id_count"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
