from pathlib import Path
import tomllib

import caypollard


def test_package_version_matches_project_metadata():
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert caypollard.__version__ == metadata["project"]["version"]
