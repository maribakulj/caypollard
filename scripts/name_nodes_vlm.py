#!/usr/bin/env python3
"""A second, independent namer for the same closed vocabulary.

The record's nodes are named by scoring each part against a closed vocabulary,
which is fast enough for a whole corpus and has one weakness: a single
instrument decides, and nothing says whether it is right. This is the second
opinion, and it reads the picture whole rather than part by part. A
vision-language model is given the same vocabulary as a JSON-schema enum, so its
decoder cannot emit anything outside it, and asked what the picture contains and
where. Where the two namers agree, a name is worth more than either alone; where
they disagree, the picture is worth looking at.

Two measurements shaped this before it was written, and both are worth keeping.

*The vocabulary has to be one a model can see.* Given Iconclass's own labels --
``fable``, ``domesticated animal``, ``mandrake`` -- a model answers ``mandrake``
five times for a photograph of a dress, and it does so whether the vocabulary
holds 32 terms or 465, so the failure is the kind of word and not the number of
them. Given ordinary nouns it answers ``dress``. The catalogue term is the right
name for a finding and the wrong name for a prompt.

*Small local models latch rather than enumerate.* A 3B model run locally fills
the list with its first answer repeated, at ten to sixty seconds a picture. A
hosted model of moderate size returns distinct things in under a second, which
is the difference between an instrument for sampling a corpus and one for
transcribing it. Repetitions are collapsed regardless, since the guard costs
nothing and the failure is known.

Images leave the machine when this runs. That is a deliberate choice, made
explicitly, and the corpora here are published museum photographs and library
scans.
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import queue
import random
import threading
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
PROMPT = "List the distinct things depicted and where each sits in the picture."


def encoded(path: Path, side: int) -> str:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode()


def ask(url: str, headers: dict, body: dict, *, timeout: int, attempts: int = 5) -> dict:
    """One request, with backoff on the codes that mean *later*, not *no*."""
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                answer = json.loads(handle.read())
            return json.loads(answer["choices"][0]["message"]["content"] or "{}")
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            # Reporting the code matters: a transient 429 has been taken for a
            # block three times in this project.
            time.sleep(min(2**attempt, 30) + random.random())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 30) + random.random())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--labels", default="data/vocabularies/everyday-nouns.txt")
    parser.add_argument("--model", default="mistral-small-latest")
    parser.add_argument("--url", default="https://api.mistral.ai/v1/chat/completions")
    parser.add_argument("--api-key-env", default="MISTRAL_API_KEY")
    parser.add_argument("--sample", type=int, help="Transcribe this many, drawn at random")
    parser.add_argument("--max-nodes", type=int, default=6)
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"{args.api_key_env} is not set")

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
                    "additionalProperties": False,
                },
            }
        },
        "required": ["nodes"],
        "additionalProperties": False,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("image_path") and Path(row["image_path"]).is_file()]
    if args.sample:
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
    pending = [row for row in rows if str(row["id"]) not in done]

    work: queue.Queue = queue.Queue()
    for row in pending:
        work.put(row)
    results: queue.Queue = queue.Queue()
    failures = [0]
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                row = work.get_nowait()
            except queue.Empty:
                return
            try:
                body = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": "data:image/jpeg;base64,"
                                    + encoded(Path(row["image_path"]), args.side),
                                },
                            ],
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "nodes", "schema": schema, "strict": True},
                    },
                    "max_tokens": 300,
                    "temperature": 0,
                }
                answer = ask(args.url, headers, body, timeout=args.timeout)
            except Exception:
                with lock:
                    failures[0] += 1
                work.task_done()
                continue
            seen: collections.OrderedDict[tuple[str, str], None] = collections.OrderedDict()
            for node in answer.get("nodes") or []:
                concept, place = node.get("concept"), node.get("place")
                if concept in vocabulary and place in CELLS:
                    seen[(concept, place)] = None
            results.put(
                {
                    "id": str(row["id"]),
                    "nodes": [
                        {"name": concept, "cell": CELLS.index(place)}
                        for concept, place in seen
                    ],
                }
            )
            work.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    written = 0
    with output.open("a" if args.resume else "w", encoding="utf-8") as sink:
        while written + failures[0] < len(pending):
            try:
                record = results.get(timeout=args.timeout + 60)
            except queue.Empty:
                break
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            if written % 200 == 0:
                rate = written / max(time.monotonic() - started, 1e-6)
                left = (len(pending) - written) / rate if rate else float("nan")
                print(
                    f"{written}/{len(pending)}  {rate:.1f}/s  reste ~{left / 60:.0f} min  "
                    f"{failures[0]} échecs",
                    flush=True,
                )
    print(
        json.dumps(
            {"asked": len(pending), "written": written, "failures": failures[0],
             "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
