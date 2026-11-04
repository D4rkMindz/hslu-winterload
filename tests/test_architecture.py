"""Keeps data access in one place. A failure here means a notebook or module took a shortcut around
the repositories: move that access into winterload.repository instead of loosening this test.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "winterload"
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))
NOTEBOOK_FORBIDDEN = [r"\brequests\.", r"\.to_csv\(", r"read_csv\(", r"\.read_range\(", r"read_parquet\(", r"PROJECT_ROOT"]


def _code_cells(notebook: Path) -> str:
    cells = json.loads(notebook.read_text())["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def _sources(*allowed: str) -> list[tuple[Path, str]]:
    return [(path, path.read_text()) for path in SRC.rglob("*.py")
            if not any(str(path.relative_to(SRC)).startswith(prefix) for prefix in allowed)]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_only_use_repositories(notebook):
    if notebook.name == "1_Feature_Store_Overview.ipynb":
        pytest.skip("Uses PROJECT_ROOT to locate the Feast repo, and reads through Feast.")
    code = _code_cells(notebook)
    for pattern in NOTEBOOK_FORBIDDEN:
        assert not re.search(pattern, code), f"{notebook.name} uses {pattern}"


def test_no_generated_csv_files_next_to_notebooks():
    assert not list((ROOT / "notebooks").rglob("*.csv"))


def test_only_repositories_read_stored_data():
    for path, text in _sources("repository/", "storage/"):
        assert ".read_range(" not in text, path


def test_only_the_result_repository_touches_csv_files():
    for path, text in _sources("repository/results.py"):
        assert not re.search(r"\.to_csv\(|read_csv\(", text), path


def test_only_sources_call_http():
    for path, text in _sources("source/"):
        assert not re.search(r"requests\.(get|post)\(", text), path


def test_repositories_import_no_config():
    for path in (SRC / "repository").glob("*.py"):
        if path.name == "factory.py":
            continue
        code = re.sub(r"if TYPE_CHECKING:\n(    .*\n)+", "", path.read_text())
        assert "winterload.config" not in code and "winterload.storage" not in code, path
