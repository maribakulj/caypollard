"""Import the experiment scripts as modules, so their functions are reused and not copied.

The scripts live in `scripts/` and end with an `if __name__ == "__main__"` guard, so
importing one runs nothing. A few of them insert `scripts/` on `sys.path` themselves to
import each other; that is done here first, once.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from functools import cache
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
_lock = threading.Lock()


@cache
def script(name: str) -> ModuleType:
    """Load `scripts/<name>.py` as a module named `<name>`, once, and never half-loaded.

    Two threads asking for the same script at the same moment would otherwise get a module
    registered in `sys.modules` before its body has run, and find it without its names.
    """
    with _lock:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"no script named {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module
