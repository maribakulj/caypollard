#!/usr/bin/env python3
"""Build a transfer set from any Wikidata collection and Commons, with no API key.

Phase 8 asks whether the gains survive outside the cataloguing environment the
method was developed in. The Rijksmuseum API needs a registered key, which would
put a human step in the middle of an automated pipeline; Wikidata and Wikimedia
Commons carry the same works openly, and Wikidata's ``depicts`` (P180) statements
join to Iconclass through P1256 -- the alignment already fetched for phase 7.

The transfer is genuinely out of domain: a different institution, a different
cataloguing tradition, a different object mix, and subject terms assigned by
Wikidata editors rather than by Iconclass specialists. Relevance is computed by
the same hierarchical path as the two existing benchmarks, so a drop is a drop in
transfer rather than in method.

Leakage control groups by creator, which is this corpus's analogue of the volume:
an artist's works resemble one another the way plates of one book do.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SPARQL = "https://query.wikidata.org/sparql"
# The Rijksmuseum was the first collection ingested and remains the default, but
# nothing in the pipeline is specific to it: any collection whose works carry a
# Commons image and a depicts statement reaches Iconclass the same way.
DEFAULT_COLLECTION = "wd:Q190804"
USER_AGENT = "caypollard-research/0.6 (https://github.com/maribakulj/caypollard) python-urllib"

WORKS_QUERY = """
SELECT ?work ?image ?depicts WHERE {
  ?work wdt:P195 %s ; wdt:P18 ?image ; wdt:P180 ?depicts .
}
"""
# One query per property rather than five OPTIONAL clauses in one. The combined
# form returns 502 from the public endpoint: optionals multiply the intermediate
# result set, and the query planner times out well before the data is large.
ATTRIBUTE_QUERY = """
SELECT ?work ?value WHERE {
  ?work wdt:P195 %s ; wdt:P18 ?image ; wdt:%s ?value .
}
"""
ATTRIBUTE_PROPERTIES = {
    "creator": "P170",
    "material": "P186",
    "genre": "P136",
    "type": "P31",
    "inception": "P571",
}


def sparql(query: str, *, timeout: int = 300) -> list[dict[str, Any]]:
    url = f"{SPARQL}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))["results"]["bindings"]


def qid(value: str) -> str:
    return value.rsplit("/", 1)[-1]


def thumbnail_url(image: str, width: int) -> str:
    """Ask Commons for a scaled copy rather than the full-resolution master.

    The masters run to tens of megabytes each; every encoder here resizes to 224
    pixels anyway, so downloading them would cost bandwidth the institution pays
    for and buy nothing.
    """
    name = urllib.parse.unquote(image.rsplit("/", 1)[-1])
    quoted = urllib.parse.quote(name.replace(" ", "_"), safe="")
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width={width}"


def download(url: str, destination: Path, *, timeout: int, attempts: int = 4) -> str:
    """Fetch one image, reporting the status code and backing off when throttled.

    Two lessons are encoded here, both learned the hard way. A failure must name
    its HTTP status: a run that reports only "HTTPError" was once diagnosed as a
    timeout, then as contention, and was actually a 404. And a burst of a couple
    of thousand requests draws a 429 from Wikimedia that clears by itself, so the
    right response is to wait and retry rather than to record two thousand
    failures and move on.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return "cached"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 503} and attempt < attempts:
                wait = float(exc.headers.get("Retry-After") or delay)
                time.sleep(min(wait, 120.0))
                delay *= 2
                continue
            return f"http {exc.code}"
        except Exception as exc:
            if attempt < attempts:
                time.sleep(delay)
                delay *= 2
                continue
            return f"error {type(exc).__name__}"
        if not payload:
            return "empty"
        # The staging name carries the process id. Two runs over the same
        # directory -- which happens whenever a resume is launched before the
        # previous one has finished -- otherwise write the same .part file, and
        # the loser of the race fails renaming a path the winner already moved.
        staging = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.part")
        staging.write_bytes(payload)
        staging.replace(destination)
        return "downloaded"
    return "exhausted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        default="Q190804",
        help="Wikidata QID of the holding collection (P195)",
    )
    parser.add_argument("--mapping", default="data/derived/iconclass-wikidata.jsonl")
    parser.add_argument("--output-dir", default="data/raw/rijksmuseum")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent downloads. Three sustained over a couple of thousand "
             "files drew a 429; two with backoff does not.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument(
        "--from-records",
        action="store_true",
        help="Download the images listed in an existing records.jsonl and query "
             "nothing. Metadata and images are separate concerns and separating "
             "them keeps a download from re-running a query that has already "
             "succeeded -- and keeps a failed query from discarding a good file.",
    )
    args = parser.parse_args()

    mapping: dict[str, set[str]] = collections.defaultdict(set)
    for line in Path(args.mapping).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for entry in row["wikidata"]:
            mapping[entry["wikidata"]].add(row["notation"])

    collection = args.collection
    if not collection.startswith("wd:"):
        collection = f"wd:{collection}"
    started = time.monotonic()

    if args.from_records:
        existing = Path(args.output_dir) / "records.jsonl"
        records = [
            json.loads(line)
            for line in existing.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        images = Path(args.output_dir) / "images"
        images.mkdir(parents=True, exist_ok=True)
        statuses: collections.Counter[str] = collections.Counter()

        def fetch_one(record: dict[str, Any]) -> str:
            return download(
                record["image_url"], images / record["filename"], timeout=args.timeout
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, status in enumerate(pool.map(fetch_one, records), start=1):
                statuses[status] += 1
                if index % 500 == 0:
                    print(f"{index}/{len(records)} — {dict(statuses)}", flush=True)
        print(
            json.dumps(
                {
                    "from_records": str(existing),
                    "records": len(records),
                    "image_download": dict(statuses),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
                ensure_ascii=False,
            )
        )
        return

    works: dict[str, dict[str, Any]] = {}
    for row in sparql(WORKS_QUERY % collection):
        item = qid(row["work"]["value"])
        record = works.setdefault(
            item,
            {"image": row["image"]["value"], "depicts": set(), "iconclass": set()},
        )
        subject = qid(row["depicts"]["value"])
        record["depicts"].add(subject)
        record["iconclass"] |= mapping.get(subject, set())

    for field, prop in ATTRIBUTE_PROPERTIES.items():
        for row in sparql(ATTRIBUTE_QUERY % (collection, prop)):
            item = qid(row["work"]["value"])
            if item not in works:
                continue
            value = row["value"]["value"]
            if field == "inception":
                works[item]["inception"] = value[:4]
            else:
                works[item].setdefault(field, set()).add(qid(value))
        print(f"  {field} ({prop}) récupéré", flush=True)

    labelled = {item: row for item, row in works.items() if row["iconclass"]}
    if args.limit:
        labelled = dict(sorted(labelled.items())[: args.limit])

    output = Path(args.output_dir)
    (output / "images").mkdir(parents=True, exist_ok=True)

    records = []
    for item, row in sorted(labelled.items()):
        records.append(
            {
                "id": f"rijksmuseum:{item}",
                "wikidata": item,
                "filename": f"{item}.jpg",
                "iconclass": sorted(row["iconclass"]),
                "depicts": sorted(row["depicts"]),
                "creator": sorted(row.get("creator", ())),
                "material": sorted(row.get("material", ())),
                "genre": sorted(row.get("genre", ())),
                "type": sorted(row.get("type", ())),
                "inception": row.get("inception"),
                "collection": "rijksmuseum",
                "image_url": thumbnail_url(row["image"], args.width),
                "source_dataset": "Rijksmuseum via Wikidata/Wikimedia Commons",
                "source_url": f"https://www.wikidata.org/wiki/{item}",
            }
        )

    statuses: collections.Counter[str] = collections.Counter()
    if not args.skip_images:
        def work(record: dict[str, Any]) -> str:
            return download(
                record["image_url"],
                output / "images" / record["filename"],
                timeout=args.timeout,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, status in enumerate(pool.map(work, records), start=1):
                statuses[status] += 1
                if index % 200 == 0:
                    print(f"{index}/{len(records)} — {dict(statuses)}", flush=True)

    records_path = output / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    report = {
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "works_with_image_and_depicts": len(works),
        "works_with_an_iconclass_aligned_subject": len(labelled),
        "records_written": len(records),
        "image_download": dict(statuses),
        "distinct_creators": len({c for row in records for c in row["creator"]}),
        "records_without_creator": sum(1 for row in records if not row["creator"]),
        "mean_notations_per_record": round(
            sum(len(row["iconclass"]) for row in records) / max(len(records), 1), 3
        ),
        "collection": args.collection,
        "thumbnail_width": args.width,
    }
    (output / "fetch-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
