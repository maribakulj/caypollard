#!/usr/bin/env python3
"""A second, independent namer for the same closed vocabulary.

The record's nodes are named by scoring each part against a closed vocabulary,
which is fast enough for a whole corpus and has one weakness: a single instrument
decides, and nothing says whether it is right. This is the second opinion. A
local vision-language model is given the same vocabulary as a JSON-schema enum,
so its decoder *cannot* emit anything outside it, and asked what the picture
contains and where. Where the two namers agree, a name is worth more than either
alone; where they disagree, the picture is worth looking at.

Two things were measured before this was written, and both shaped it.

*The vocabulary has to be one a model can see.* Given Iconclass's own labels --
``fable``, ``domesticated animal``, ``mandrake`` -- a 3B model answers
``mandrake`` five times for a photograph of a dress, and it does so whether the
vocabulary holds 32 terms or 465, so the failure is the kind of word and not the
number of them. Given ordinary nouns it answers ``dress``. The catalogue term is
the right name for a finding and the wrong name for a prompt.

*The model latches rather than enumerates.* It fills the list with its first
answer repeated, which is why the request is for few nodes and the repetitions
are collapsed rather than counted.

This runs on a sample by design. At ten to sixty seconds a picture it is an
instrument for checking a corpus, not for transcribing one.
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

CELLS = (
    "haut-gauche", "haut-centre", "haut-droite",
    "milieu-gauche", "centre", "milieu-droite",
    "bas-gauche", "bas-centre", "bas-droite",
)
PROMPT = "List the different things depicted and where each sits in the picture."


def encoded(path: Path, side: int) -> str:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode()


def ask(host: str, model: str, image: str, schema: dict, predict: int, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": PROMPT,
                "images": [image],
                "stream": False,
                "format": schema,
                "options": {"num_predict": predict, "temperature": 0.0},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        body = json.loads(handle.read())
    return json.loads(body.get("response") or "{}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--labels", default="data/vocabularies/everyday-nouns.txt")
    parser.add_argument("--model", default="hf.co/mradermacher/churro-3B-GGUF:Q4_K_M")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--max-nodes", type=int, default=3)
    parser.add_argument("--side", type=int, default=448)
    parser.add_argument("--predict", type=int, default=90)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    words = [
        line.strip()
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    vocabulary = list(dict.fromkeys(words))
    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "maxItems": args.max_nodes,
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "enum": vocabulary},
                        "place": {"type": "string", "enum": list(CELLS)},
                    },
                    "required": ["concept", "place"],
                },
            }
        },
        "required": ["nodes"],
    }

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("image_path") and Path(row["image_path"]).is_file()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and output.is_file():
        done = {
            json.loads(line)["id"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    started = time.monotonic()
    failures = 0
    with output.open("a" if args.resume else "w", encoding="utf-8") as sink:
        for position, row in enumerate(rows, start=1):
            item = str(row["id"])
            if item in done:
                continue
            try:
                answer = ask(
                    args.host, args.model, encoded(Path(row["image_path"]), args.side),
                    schema, args.predict, args.timeout,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                failures += 1
                continue
            # The model repeats its first answer to fill the list, so a concept
            # is kept once per place rather than counted.
            seen: collections.OrderedDict[tuple[str, str], None] = collections.OrderedDict()
            for node in answer.get("nodes") or []:
                concept, place = node.get("concept"), node.get("place")
                if concept in vocabulary and place in CELLS:
                    seen[(concept, place)] = None
            sink.write(
                json.dumps(
                    {
                        "id": item,
                        "nodes": [
                            {"name": concept, "cell": CELLS.index(place)}
                            for concept, place in seen
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
            if position % 25 == 0:
                rate = position / max(time.monotonic() - started, 1e-6)
                print(
                    f"{position}/{len(rows)}  {rate * 60:.1f}/min  {failures} échecs",
                    flush=True,
                )
    print(json.dumps({"asked": len(rows), "failures": failures, "output": str(output)}))


if __name__ == "__main__":
    main()
