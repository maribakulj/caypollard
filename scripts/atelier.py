#!/usr/bin/env python3
"""The workshop's command line: list bricks and benches, run combinations, fuse, tabulate.

    uv run python scripts/atelier.py briques
    uv run python scripts/atelier.py bancs
    uv run python scripts/atelier.py pose copies --figures owlv2 --pose vitpose-humanart
    uv run python scripts/atelier.py vecteur sujets --brique dinov2
    uv run python scripts/atelier.py regions derivees --brique dinov2-patchs
    uv run python scripts/atelier.py relations sujets --figures opus --pose opus
    uv run python scripts/atelier.py somme sujets pose--opus--opus vector--dinov2
    uv run python scripts/atelier.py tableau

Heavy bricks (more than 1 GB in memory) are flagged in ``briques``: run them through
``~/outils-seg/lourd.sh``. Results gather in ``results/atelier/TABLEAU.md``.
"""

from __future__ import annotations

import argparse
import json

from caypollard.atelier.benches import BENCHES
from caypollard.atelier.bricks import BRICKS
from caypollard.atelier.run import (
    run_fusion,
    run_pose,
    run_regions,
    run_relations,
    run_scene,
    run_scene_vectors,
    run_vector,
    table,
)


def show(result: dict) -> None:
    print(result["combination"])
    for direction, s in result["scores"].items():
        c = result["chance"][direction]
        print(
            f"  {direction}: top1 {s['top1']} (chance {c['top1']}), mAP {s['mAP']} "
            f"(chance {c['mAP']}) — "
            + json.dumps(s.get("by_medium") or s.get("top1_by_label"), ensure_ascii=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("briques")
    sub.add_parser("bancs")
    sub.add_parser("tableau")
    p = sub.add_parser("pose")
    p.add_argument("banc", choices=sorted(BENCHES))
    p.add_argument("--figures", required=True)
    p.add_argument("--pose", required=True)
    p.add_argument("--joints", choices=["coco"], help="score only these joints")
    v = sub.add_parser("vecteur")
    v.add_argument("banc", choices=sorted(BENCHES))
    v.add_argument("--brique", required=True)
    q = sub.add_parser("relations")
    q.add_argument("banc", choices=sorted(BENCHES))
    q.add_argument("--figures", required=True)
    q.add_argument("--pose", required=True)
    q.add_argument("--postures-seules", action="store_true", help="the control without relations")
    g = sub.add_parser("scene")
    g.add_argument("banc", choices=sorted(BENCHES))
    g.add_argument("--brique", required=True)
    g.add_argument("--figures-seules", action="store_true", help="the control without relations")
    g.add_argument("--vecteurs", action="store_true", help="phrases through a text encoder")
    r = sub.add_parser("regions")
    r.add_argument("banc", choices=sorted(BENCHES))
    r.add_argument("--brique", required=True)
    s = sub.add_parser("somme")
    s.add_argument("banc", choices=sorted(BENCHES))
    s.add_argument("combinaisons", nargs="+")
    args = parser.parse_args()

    if args.command == "briques":
        for (kind, name), b in sorted(BRICKS.items()):
            heavy = "  [lourd]" if b.heavy else ""
            print(f"{kind:8} {name:18} {b.weights_mb:>5} Mo{heavy}  {b.summary}")
    elif args.command == "bancs":
        for b in BENCHES.values():
            print(f"{b.name:8} ({b.rule}) {b.description}")
    elif args.command == "tableau":
        print(table())
    elif args.command == "pose":
        show(run_pose(args.banc, args.figures, args.pose, args.joints))
    elif args.command == "vecteur":
        show(run_vector(args.banc, args.brique))
    elif args.command == "relations":
        show(run_relations(args.banc, args.figures, args.pose, not args.postures_seules))
    elif args.command == "scene":
        if args.vecteurs:
            show(run_scene_vectors(args.banc, args.brique))
        else:
            show(run_scene(args.banc, args.brique, not args.figures_seules))
    elif args.command == "regions":
        show(run_regions(args.banc, args.brique))
    else:
        show(run_fusion(args.banc, args.combinaisons))


if __name__ == "__main__":
    main()
