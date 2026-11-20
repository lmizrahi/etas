"""The documentation's code examples must run."""
import os
import re
import subprocess
import sys

import pytest

from tests.conftest import INPUT, REPO


def _python_blocks(markdown):
    return re.findall(r"```python\n(.*?)```", markdown, flags=re.S)


@pytest.mark.slow
def test_quickstart_runs(tmp_path):
    """Run all code blocks of docs/quickstart.md in order, in a fresh
    directory that contains input_data/ and output_data/."""
    blocks = _python_blocks((REPO / "docs" / "quickstart.md").read_text())
    assert len(blocks) >= 4
    os.symlink(INPUT, tmp_path / "input_data")
    (tmp_path / "output_data").mkdir()
    script = "import matplotlib\nmatplotlib.use('Agg')\n" + "\n".join(blocks)
    res = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                         capture_output=True, text=True, timeout=1200)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "output_data"
    assert (out / "parameters_example.json").exists()
    assert (out / "forecast_example.csv").exists()
    assert list(out.glob("*.pdf"))


def test_docs_build(tmp_path):
    """mkdocs builds without warnings (if mkdocs is installed)."""
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocstrings")
    res = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict",
         "-d", str(tmp_path / "site")],
        cwd=REPO, capture_output=True, text=True,
        env=dict(os.environ, NO_MKDOCS_2_WARNING="1"))
    assert res.returncode == 0, res.stderr[-3000:]
