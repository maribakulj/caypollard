#!/usr/bin/env python3
"""Audit Joconde's schema against the task this project defines.

Phase 9 asks whether the representation and fusion interface ports to a
different graph schema, and where task definitions diverge. The second question
turns out to answer the first, so this script measures the divergence rather
than asserting it.

The national export is 1.2 GB, and nothing here needs all of it: the questions
are about the *shape* of the subject vocabulary, and a stratified sample of byte
ranges answers them. Ranges are fixed fractions of the file, so the audit is
reproducible, and the first slice of the file is deliberately not used alone --
records are ordered by reference, so the head is one museum's holdings and looks
nothing like the whole.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import re
import urllib.request
from pathlib import Path

USER_AGENT = "caypollard-research/0.6 (https://github.com/maribakulj/caypollard) python-urllib"
DEFAULT_URL = "https://ministere-culture.s3.sbg.io.cloud.ovh.net/POP/joconde.csv"
DEFAULT_FRACTIONS = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
# A Joconde subject reads "head (facet, facet : sub-facet)", which is a facet
# structure rather than a classification: there is no parent notation to walk up.
FACETED = re.compile(r"^([^(]+)\((.*)\)\s*$")


def ranged(url: str, start: int, length: int, *, timeout: int) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{start + length}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def total_size(url: str, *, timeout: int) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--slice-bytes", type=int, default=4_000_000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    size = total_size(args.url, timeout=args.timeout)
    header = ranged(args.url, 0, 20_000, timeout=args.timeout).splitlines()[0]
    columns = header.split("|")

    rows: list[dict[str, str]] = []
    for fraction in DEFAULT_FRACTIONS:
        chunk = ranged(args.url, int(size * fraction), args.slice_bytes, timeout=args.timeout)
        body = chunk.splitlines()[1:-1]  # drop the partial lines at both ends
        reader = csv.DictReader(io.StringIO(header + "\n" + "\n".join(body)), delimiter="|")
        rows.extend(row for row in reader if row.get("Reference"))

    images = collections.Counter(row.get("Presence_image") for row in rows)
    subjects: collections.Counter[str] = collections.Counter()
    heads: collections.Counter[str] = collections.Counter()
    faceted = with_subject = 0
    for row in rows:
        value = (row.get("Sujet_Represente") or "").strip()
        if not value:
            continue
        with_subject += 1
        for term in (part.strip() for part in value.split(",")):
            if not term:
                continue
            subjects[term] += 1
            match = FACETED.match(term)
            if match:
                faceted += 1
                heads[match.group(1).strip()] += 1
            else:
                heads[term] += 1

    singletons = sum(1 for count in subjects.values() if count == 1)
    report = {
        "url": args.url,
        "file_bytes": size,
        "sampled_fractions": list(DEFAULT_FRACTIONS),
        "slice_bytes": args.slice_bytes,
        "records_sampled": len(rows),
        "columns": len(columns),
        "has_image_url_column": any("image" in c.lower() and "url" in c.lower() for c in columns),
        "image_flag": dict(images),
        "records_with_a_subject": with_subject,
        "subject_coverage": round(with_subject / max(len(rows), 1), 4),
        "distinct_subject_terms": len(subjects),
        "subject_term_occurrences": sum(subjects.values()),
        "terms_occurring_once": singletons,
        "share_of_terms_occurring_once": round(singletons / max(len(subjects), 1), 4),
        "terms_with_facet_syntax": faceted,
        "distinct_head_terms": len(heads),
        "most_common_terms": subjects.most_common(15),
        "most_common_head_terms": heads.most_common(15),
        "divergence": (
            "Sujet_Represente is a free-text faceted phrase -- a head term with parenthesised "
            "qualifiers -- not a classification. There is no parent relation to walk, so graded "
            "hierarchical relevance, this project's primary endpoint, is undefined on it. Exact "
            "match degenerates too: most terms occur once in the sample."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("most_common_terms", "most_common_head_terms")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
