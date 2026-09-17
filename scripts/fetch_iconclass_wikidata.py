#!/usr/bin/env python3
"""Join Iconclass notations to Wikidata, the one stable public alignment there is.

Phase 7 asks for concepts to be mapped to existing vocabularies "when stable
mappings exist". The Iconclass SKOS export shipped with the benchmark carries
only notation, broader and narrower -- no exactMatch, no sameAs -- so the
alignment has to come from outside. Wikidata's property P1256 holds an Iconclass
notation, which makes a single SPARQL query the whole mapping.

The result is reported as coverage against the corpus rather than merged into the
benchmark: a mapping that reaches a small fraction of assignments cannot be used
as ground truth, and saying so precisely is the deliverable.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from caypollard.graphs.iconclass import normalize_notation
from caypollard.provenance import read_jsonl

ENDPOINT = "https://query.wikidata.org/sparql"
QUERY = """
SELECT ?item ?itemLabel ?notation WHERE {
  ?item wdt:P1256 ?notation .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,fr,de,la". }
}
"""
# Wikidata asks for a descriptive agent naming the tool and a contact route; an
# anonymous agent is throttled or refused.
USER_AGENT = "caypollard-research/0.6 (https://github.com/maribakulj/caypollard) python-urllib"


def fetch(endpoint: str, query: str, *, timeout: int) -> dict:
    url = f"{endpoint}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Benchmark manifest whose notations are being covered")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--mapping-output", default="data/derived/iconclass-wikidata.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    payload = fetch(args.endpoint, QUERY, timeout=args.timeout)
    rows = payload["results"]["bindings"]
    mapping: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        notation = row["notation"]["value"].strip()
        mapping[notation].append(
            {
                "wikidata": row["item"]["value"].rsplit("/", 1)[-1],
                "label": row.get("itemLabel", {}).get("value", ""),
            }
        )

    records = read_jsonl(args.manifest)
    assignments = collections.Counter(
        str(label) for row in records for label in row.get("iconclass", [])
    )
    exact = sum(count for notation, count in assignments.items() if notation in mapping)
    normalised = sum(
        count
        for notation, count in assignments.items()
        if notation in mapping or (normalize_notation(notation) or "") in mapping
    )
    items_covered = sum(
        1
        for row in records
        if any(
            str(label) in mapping or (normalize_notation(str(label)) or "") in mapping
            for label in row.get("iconclass", [])
        )
    )

    output_path = Path(args.mapping_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for notation in sorted(mapping):
            handle.write(
                json.dumps(
                    {"notation": notation, "wikidata": mapping[notation]}, ensure_ascii=False
                )
                + "\n"
            )

    total = sum(assignments.values())
    report = {
        "endpoint": args.endpoint,
        "property": "P1256 (Iconclass notation)",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "wikidata_rows": len(rows),
        "distinct_notations_in_wikidata": len(mapping),
        "manifest": args.manifest,
        "distinct_notations_in_corpus": len(assignments),
        "assignments_in_corpus": total,
        "assignments_covered_exact": exact,
        "assignments_covered_after_normalisation": normalised,
        "coverage_of_assignments": round(normalised / total, 4) if total else 0.0,
        "items_with_at_least_one_mapped_notation": items_covered,
        "coverage_of_items": round(items_covered / len(records), 4) if records else 0.0,
        "mapping_file": str(output_path),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
