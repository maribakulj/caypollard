#!/usr/bin/env python3
"""Run workshop bricks on a rented GPU (Colab or Kaggle) and bring back their scores.

Everything sits in ``ROOT`` (default ``/content``): ``code.tar`` (``git archive HEAD src
scripts``) and one ``<folder>.tar.gz`` per bench folder of ``data/derived`` (``copies``,
``subjects``, ``based-on``, ``relations``; dereferenced). ``CAY_RUNS`` lists the runs as
``command:bench:brick`` separated by commas, with ``command`` ``vecteur`` or ``regions``,
e.g. ``vecteur:copies:pe-spatial,regions:copies:pe-spatial-patchs``. Each run is the
workshop's own command (``scripts/atelier.py``), so scores, chance and caches are the same
as on the Mac. Output: ``ROOT/atelier-out.tar.gz`` with ``results/atelier`` and the
distance matrices (``data/derived/atelier/*/*.npy``), plus ``ROOT/atelier.log``.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.environ.get("CAY_REMOTE_ROOT", "/content")
RUNS = [r.split(":") for r in os.environ["CAY_RUNS"].split(",") if r]
WORK = ROOT + "/cay"

shutil.rmtree(WORK, ignore_errors=True)  # no result of an earlier run is carried over
os.makedirs(WORK + "/data/derived", exist_ok=True)
with tarfile.open(ROOT + "/code.tar") as archive:
    archive.extractall(WORK)
for path in glob.glob(ROOT + "/*.tar.gz"):
    if not path.endswith("atelier-out.tar.gz"):
        with tarfile.open(path) as archive:
            archive.extractall(WORK + "/data/derived")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==5.17.0",
        "timm",
        "open_clip_torch",
    ],
    check=True,
)
env = dict(os.environ, PYTHONPATH=WORK + "/src")
with open(ROOT + "/code.tar", "rb") as f:
    code_sha = hashlib.sha256(f.read()).hexdigest()
versions = subprocess.run(
    [
        sys.executable,
        "-c",
        "import torch, transformers, timm; "
        "print(torch.__version__, transformers.__version__, timm.__version__)",
    ],
    capture_output=True,
    text=True,
).stdout.split()
os.makedirs(WORK + "/results/atelier", exist_ok=True)
with open(WORK + "/results/atelier/remote-run.json", "w") as f:
    json.dump({"code_sha256": code_sha, "torch_transformers_timm": versions, "runs": RUNS}, f)
failed = []
for command, bench, brick in RUNS:
    print(f"== {command} {bench} {brick}", flush=True)
    done = subprocess.run(
        [sys.executable, "scripts/atelier.py", command, bench, "--brique", brick],
        cwd=WORK,
        env=env,
    )
    if done.returncode:
        failed.append(f"{command}:{bench}:{brick}")
with tarfile.open(ROOT + "/atelier-out.tar.gz", "w:gz") as out:
    out.add(WORK + "/results/atelier", arcname="results/atelier")
    for npy in glob.glob(WORK + "/data/derived/atelier/*/*.npy"):
        out.add(npy, arcname=os.path.relpath(npy, WORK))
print("failed: " + (", ".join(failed) or "none"), flush=True)
print("done", flush=True)
sys.exit(1 if failed else 0)
