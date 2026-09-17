#!/usr/bin/env python3
"""Ingest the Emblematica Online emblem corpus: records first, images separately.

Two stages, deliberately split. ``--stage records`` walks the search endpoint for
emblem identifiers and fetches one SPINE XML per emblem; ``--stage images`` then
downloads picturae for emblems that actually carry Iconclass notations, which is
the only subset usable as a labelled benchmark. Splitting them means the image
download — by far the larger cost — is never paid for emblems that cannot be
evaluated.

Both stages cache to disk and skip what is already present, so an interrupted run
resumes by being re-run. Requests are serialised with a delay against what is a
university library server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from caypollard.datasets.emblematica import (
    book_id_from_pictura,
    iconclass_coverage,
    parse_emblem_xml,
)

BASE = "http://emblematica.library.illinois.edu"
XML_URL = "http://emblemimages.library.illinois.edu/{book}/emblematica/emblem{number}.xml"
USER_AGENT = "caypollard-research/0.1 (cultural heritage retrieval research)"


def _get(url: str, *, timeout: float, attempts: int = 3, backoff: float = 2.0) -> bytes:
    """Fetch one URL, retrying transient failures.

    A single dropped connection during a 333-page index walk would otherwise
    discard every page already retrieved, so transient errors are retried with a
    widening pause before the failure is allowed to propagate.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("unreachable")


def fetch_index(cache: Path, *, delay: float, timeout: float) -> list[dict]:
    """Page through the search endpoint once, appending each page to disk.

    The endpoint ignores ``Take`` and returns a fixed page of 18, so the walk is
    roughly 1 850 requests. Pages are appended as they arrive rather than written
    once at the end, so an interruption costs the current page instead of the
    whole walk, and a re-run resumes from the number of rows already stored.
    """
    index_path = cache / "index.jsonl"
    rows: list[dict] = []
    if index_path.is_file():
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]

    seen = {row.get("emblemID") for row in rows}
    handle = index_path.open("a", encoding="utf-8")
    try:
        skip = len(rows)
        total = None
        while True:
            payload = json.loads(
                _get(f"{BASE}/api/Emblem/Search?Skip={skip}&Take=100", timeout=timeout)
            )
            batch = payload.get("Emblems", [])
            total = payload.get("Total", total or 0)
            if not batch:
                break
            fresh = [row for row in batch if row.get("emblemID") not in seen]
            for row in fresh:
                seen.add(row.get("emblemID"))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            handle.flush()
            skip += len(batch)
            if skip % (18 * 50) < len(batch):
                print(f"  index {len(rows)}/{total}", flush=True)
            if skip >= total:
                break
            time.sleep(delay)
    finally:
        handle.close()
    print(f"index complet : {len(rows)} emblèmes", flush=True)
    return rows


def stage_records(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir)
    (cache / "xml").mkdir(parents=True, exist_ok=True)
    index = fetch_index(cache, delay=args.delay, timeout=args.timeout)

    pending = []
    for row in index:
        emblem_id = row.get("emblemID")
        book = book_id_from_pictura(row.get("Pictura"))
        if not emblem_id or not book:
            continue
        if (cache / "xml" / f"{emblem_id}.xml").is_file():
            continue
        pending.append((emblem_id, book))
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(pending)} enregistrements à récupérer "
          f"({args.workers} connexion(s))", flush=True)

    def fetch_one(item: tuple[str, str]) -> bool:
        emblem_id, book = item
        url = XML_URL.format(book=book, number=emblem_id.lstrip("E"))
        try:
            payload = _get(url, timeout=args.timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False
        (cache / "xml" / f"{emblem_id}.xml").write_bytes(payload)
        return True

    state = {"done": 0, "failures": 0, "streak": 0}
    lock = threading.Lock()
    outage = threading.Event()

    def wait_for_network() -> None:
        """Hold every worker until the host answers again.

        Without this a network outage is indistinguishable from 19 000 individual
        failures: the queue drains in minutes, every remaining item is marked
        failed, and the run exits "successfully" having fetched nothing. Pausing
        instead means a dropped connection — or a closed laptop lid — costs the
        time it lasts and nothing else.
        """
        if outage.is_set():
            outage.wait()
            return
        outage.clear()
        print("réseau interrompu — attente du rétablissement", flush=True)
        while True:
            time.sleep(15)
            try:
                _get(f"{BASE}/api/Emblem/Search?Skip=0&Take=1", timeout=15, attempts=1)
            except Exception:  # any failure here simply means still offline
                continue
            print("réseau rétabli — reprise", flush=True)
            outage.set()
            outage.clear()
            return

    def run(item: tuple[str, str]) -> None:
        if outage.is_set():
            outage.wait()
        ok = fetch_one(item)
        # Each worker paces itself, so the delay is per connection: with N
        # workers the host sees roughly N/delay requests per second.
        time.sleep(args.delay)
        with lock:
            state["done"] += 1
            state["failures"] += 0 if ok else 1
            state["streak"] = 0 if ok else state["streak"] + 1
            if state["done"] % 500 == 0 or state["done"] == len(pending):
                print(f"  {state['done']}/{len(pending)}  échecs={state['failures']}", flush=True)
            # A handful of scattered failures are individual bad records; a run of
            # them is the network, and burning the queue over it is the real loss.
            stalled = state["streak"] >= 20
        if stalled:
            wait_for_network()
            with lock:
                state["streak"] = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, pending))

    print(json.dumps({"stage": "records", "fetched": len(pending) - state["failures"],
                      "failures": state["failures"]}, indent=2))


def _load_records(cache: Path) -> list:
    by_id = {
        row["emblemID"]: row
        for row in (
            json.loads(line)
            for line in (cache / "index.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    records = []
    for path in sorted((cache / "xml").glob("E*.xml")):
        row = by_id.get(path.stem, {})
        records.append(
            parse_emblem_xml(
                path.read_text(encoding="utf-8", errors="replace"),
                emblem_id=path.stem,
                book_id=book_id_from_pictura(row.get("Pictura")),
                collection=row.get("CollectionCode"),
                pictura_url=row.get("Pictura"),
            )
        )
    return records


def stage_images(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir)
    images = cache / "images"
    images.mkdir(parents=True, exist_ok=True)
    # Only labelled emblems are worth the bytes: an emblem without notations
    # cannot enter a relevance-based evaluation either way.
    records = [r for r in _load_records(cache) if r.iconclass and r.pictura_url]
    pending = [r for r in records if not (images / f"{r.emblem_id}.jpg").is_file()]

    if args.priority_manifest:
        wanted = {split.strip() for split in args.priority_splits.split(",") if split.strip()}
        priority = {
            json.loads(line)["id"].split(":", 1)[1]
            for line in Path(args.priority_manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("split") in wanted
        }
        pending.sort(key=lambda record: record.emblem_id not in priority)
        print(f"priorité : {sum(1 for r in pending if r.emblem_id in priority)} "
              f"images de {sorted(wanted)} d'abord", flush=True)

    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(records)} emblèmes annotés, {len(pending)} images à télécharger", flush=True)
    failures = 0
    for position, record in enumerate(pending, start=1):
        try:
            (images / f"{record.emblem_id}.jpg").write_bytes(
                _get(record.pictura_url, timeout=args.timeout)
            )
        except (urllib.error.URLError, TimeoutError):
            failures += 1
        if position % 500 == 0 or position == len(pending):
            print(f"  {position}/{len(pending)}  échecs={failures}", flush=True)
        time.sleep(args.delay)
    print(json.dumps({"stage": "images", "fetched": len(pending) - failures,
                      "failures": failures}, indent=2))


def stage_report(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir)
    records = _load_records(cache)
    report = iconclass_coverage(records) | {
        "source": "Emblematica Online",
        "api_base": BASE,
        "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    (cache / "coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("records", "images", "report"), default="records")
    parser.add_argument("--cache-dir", default="data/raw/emblematica")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--priority-manifest",
        help="Benchmark manifest whose validation/test rows are downloaded first. "
             "Evaluation needs only those splits, so ordering by them yields a "
             "complete answer in a quarter of the time; train backfills after.",
    )
    parser.add_argument(
        "--priority-splits",
        default="validation,test",
        help="Comma-separated splits to prioritise when --priority-manifest is given.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent connections. Emblematica states no rate policy and serves no "
             "robots.txt, so this stays at or below the parallelism of one ordinary "
             "browser rather than treating silence as permission.",
    )
    args = parser.parse_args()

    {"records": stage_records, "images": stage_images, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
