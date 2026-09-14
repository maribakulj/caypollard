"""Parse the open Iconclass core hierarchy and compute graded relevance.

The official ``iconclass/data`` repository stores ``notations.txt`` in a small
record-oriented text format. A record is terminated by ``$``; ``N`` identifies
the notation and ``C`` lists child notations, with continuation values beginning
with ``;``. We intentionally parse this source format directly so the benchmark
can be pinned to a specific vocabulary revision.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, SKOS

_QUALIFIER_RE = re.compile(r"\([^()]*\)")


def parse_notations(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Parse Iconclass ``notations.txt`` into ``notation -> fields`` records."""
    records: dict[str, dict[str, list[str]]] = {}
    current: dict[str, list[str]] = defaultdict(list)
    last_field: str | None = None

    def flush() -> None:
        nonlocal current, last_field
        if not current:
            return
        if "N" not in current or not current["N"]:
            raise ValueError("Encountered Iconclass record without an N field")
        notation = current["N"][0]
        records[notation] = {key: list(values) for key, values in current.items()}
        current = defaultdict(list)
        last_field = None

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            if line == "$":
                flush()
                continue
            if line.startswith(";"):
                if last_field is None:
                    raise ValueError("Continuation line found before a field")
                current[last_field].append(line[1:].strip())
                continue
            field, sep, value = line.partition(" ")
            if not sep:
                raise ValueError(f"Malformed Iconclass line: {line!r}")
            last_field = field
            current[field].append(value.strip())
    flush()
    return records


def child_edges(records: dict[str, dict[str, list[str]]]) -> list[tuple[str, str]]:
    """Return sorted unique ``(parent, child)`` edges."""
    edges = {
        (notation, child)
        for notation, fields in records.items()
        for child in fields.get("C", [])
        if child
    }
    return sorted(edges)


def build_parent_index(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Create a child -> parents index; multiple parents are preserved."""
    parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        parents[child].add(parent)
        parents.setdefault(parent, set())
    return dict(parents)


def normalize_notation(notation: str) -> str:
    """Strip bracketed name/key qualifiers for conservative base-node lookup.

    Examples include ``31A24(+1) -> 31A24`` and
    ``11H(BASIL THE GREAT) -> 11H``. Exact matches should always be attempted
    before this fallback because expanded key nodes can exist in some exports.
    """
    previous = notation.strip()
    while True:
        cleaned = _QUALIFIER_RE.sub("", previous)
        if cleaned == previous:
            return cleaned.strip()
        previous = cleaned


def resolve_notation(notation: str, parents: dict[str, set[str]]) -> str | None:
    if notation in parents:
        return notation
    normalized = normalize_notation(notation)
    return normalized if normalized in parents else None


def _ancestor_distances(node: str, parents: dict[str, set[str]]) -> dict[str, int]:
    distances = {node: 0}
    queue: deque[tuple[str, int]] = deque([(node, 0)])
    while queue:
        current, distance = queue.popleft()
        for parent in parents.get(current, set()):
            candidate = distance + 1
            if parent not in distances or candidate < distances[parent]:
                distances[parent] = candidate
                queue.append((parent, candidate))
    return distances



def hierarchy_depth(notation: str, parents: dict[str, set[str]]) -> int | None:
    """Return minimum number of parent edges from a resolved concept to a root."""
    node = resolve_notation(notation, parents)
    if node is None:
        return None
    queue: deque[tuple[str, int]] = deque([(node, 0)])
    seen = {node}
    while queue:
        current, depth = queue.popleft()
        node_parents = parents.get(current, set())
        if not node_parents:
            return depth
        for parent in node_parents:
            if parent not in seen:
                seen.add(parent)
                queue.append((parent, depth + 1))
    return None


def semantic_distance(
    notation_a: str,
    notation_b: str,
    parents: dict[str, set[str]],
) -> int | None:
    """Shortest hierarchy distance through a common ancestor.

    Returns ``None`` when either notation cannot be resolved or when the two
    nodes have no common ancestor in the parsed graph.
    """
    a = resolve_notation(notation_a, parents)
    b = resolve_notation(notation_b, parents)
    if a is None or b is None:
        return None
    distances_a = _ancestor_distances(a, parents)
    distances_b = _ancestor_distances(b, parents)
    common = set(distances_a).intersection(distances_b)
    if not common:
        return None
    return min(distances_a[node] + distances_b[node] for node in common)


def hierarchical_similarity(
    notation_a: str,
    notation_b: str,
    parents: dict[str, set[str]],
) -> float:
    """Convert hierarchy distance into a graded relevance score in ``[0, 1]``.

    ``1`` denotes the same resolved concept, parent/child concepts receive
    ``0.5``, and unrelated or unresolved branches receive ``0``. This simple,
    transparent definition is the phase-1 baseline, not a claim that taxonomy
    distance exhausts iconographic similarity.
    """
    distance = semantic_distance(notation_a, notation_b, parents)
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + distance)


def to_skos_graph(records: dict[str, dict[str, list[str]]]) -> Graph:
    """Export parsed parent/child structure as a compact SKOS graph."""
    graph = Graph()
    iconclass = Namespace("https://iconclass.org/")
    graph.bind("skos", SKOS)
    graph.bind("iconclass", iconclass)

    def uri(notation: str) -> URIRef:
        return URIRef(f"https://iconclass.org/{quote(notation, safe='')}")

    for notation in records:
        subject = uri(notation)
        graph.add((subject, RDF.type, SKOS.Concept))
        graph.add((subject, SKOS.notation, Literal(notation)))
    for parent, child in child_edges(records):
        graph.add((uri(child), SKOS.broader, uri(parent)))
        graph.add((uri(parent), SKOS.narrower, uri(child)))
    return graph


def image_hierarchical_similarity(
    labels_a: Iterable[str],
    labels_b: Iterable[str],
    parents: dict[str, set[str]],
) -> float:
    """Maximum pairwise hierarchy similarity between two multi-label images.

    This implements the protocol-v0.1 aggregation baseline. Alternative
    aggregations (mean, bipartite matching, information-content weighting) are
    reserved for explicit ablation rather than silently changing evaluation.
    """
    left = list(labels_a)
    right = list(labels_b)
    if not left or not right:
        return 0.0
    return max(hierarchical_similarity(a, b, parents) for a in left for b in right)
