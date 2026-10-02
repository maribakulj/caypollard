"""Name what a picture depicts, from a closed vocabulary, with Claude Code as the namer.

The frozen named-node tables were produced by `scripts/name_nodes_vlm.py` with a Mistral
model. Here the same question is put to Claude (Sonnet) through the Claude Code command
line in non-interactive mode, which uses the user's own subscription and needs no key: it
reads the image file with its Read tool and answers JSON. The prompt keeps the original's
question, its closed vocabulary, its nine places and its cap of six nodes, so the answer
vectorises exactly as the frozen ones did. What cannot be kept is the model: a different
namer sees differently, and the reproduction check on a pool picture shows by how much.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .bridge import ROOT, script

VOCABULARY_FILE = ROOT / "data/vocabularies/everyday-nouns.txt"


def vocabulary() -> list[str]:
    words = [
        line.strip()
        for line in VOCABULARY_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return list(dict.fromkeys(words))


def cells() -> list[str]:
    return list(script("name_nodes_vlm").CELLS)


def prompt(image_path: Path, *, max_nodes: int) -> str:
    original = script("name_nodes_vlm").PROMPT
    return (
        f"Read the image file {image_path} with the Read tool, then answer.\n"
        f"{original}\n"
        f"Answer ONLY with a JSON object of the form "
        f'{{"nodes": [{{"concept": "...", "place": "..."}}]}} and nothing else. '
        f"At most {max_nodes} nodes. Each concept MUST be one of these words exactly: "
        f"{', '.join(vocabulary())}. Each place MUST be one of: {', '.join(cells())}. "
        f"List the most prominent things first."
    )


def available() -> bool:
    return shutil.which("claude") is not None


def parse(text: str, *, max_nodes: int) -> list[dict]:
    """The nodes the answer names, filtered and deduplicated as the original script did."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", body, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    words, places = set(vocabulary()), cells()
    seen: set[tuple[str, str]] = set()
    nodes: list[dict] = []
    for node in data.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        concept, place = (
            str(node.get("concept", "")).strip().lower(),
            str(node.get("place", "")).strip(),
        )
        if concept in words and place in places and (concept, place) not in seen:
            seen.add((concept, place))
            nodes.append({"name": concept, "cell": places.index(place)})
        if len(nodes) >= max_nodes:
            break
    return nodes


def name_nodes(
    image_path: Path, *, model: str = "sonnet", max_nodes: int = 6, timeout: float = 150.0
) -> dict:
    """Ask Claude Code, headless, to name the picture. Returns nodes, raw answer and timing."""
    if not available():
        return {
            "nodes": [],
            "raw": "",
            "model": model,
            "error": "la commande `claude` est introuvable",
        }
    command = [
        "claude",
        "-p",
        "--model",
        model,
        "--allowedTools",
        "Read",
        "--output-format",
        "json",
        prompt(image_path, max_nodes=max_nodes),
    ]
    try:
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT)
        )
    except subprocess.TimeoutExpired:
        return {
            "nodes": [],
            "raw": "",
            "model": model,
            "error": f"pas de réponse en {timeout:.0f} s",
        }
    if run.returncode != 0:
        return {"nodes": [], "raw": run.stderr[-500:], "model": model, "error": "claude a échoué"}
    try:
        payload = json.loads(run.stdout)
        text = str(payload.get("result", ""))
        used = ", ".join(payload.get("modelUsage", {}).keys()) or model
    except json.JSONDecodeError:
        text, used = run.stdout, model
    return {"nodes": parse(text, max_nodes=max_nodes), "raw": text, "model": used, "error": None}
