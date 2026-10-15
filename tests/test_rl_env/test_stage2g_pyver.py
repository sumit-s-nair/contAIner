"""Stage 2G-fix: pilot corpus excludes repos whose declared Python constraint
rejects the sandbox interpreter (3.11), and heavy repos by their own name.

Found by the first real winnability run (2026-10-15): flask-blueprint-tutorial
(poetry python ^3.12; every requirements.txt line marker-gated to >=3.12) made
`pip install -r` succeed while installing nothing (req 0/12), and OpenManus
(setup.py python_requires >=3.12) sat in pip resolution until the 600s timeout.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_pilot_corpus as b  # noqa: E402


@pytest.mark.parametrize("poetry,expected", [
    ("^3.12,<4", ">=3.12,<4.0,<4"),
    ("^3.8", ">=3.8,<4.0"),
    ("~3.10", ">=3.10,<3.11"),
    (">=3.9,<3.13", ">=3.9,<3.13"),
    ("*", ""),
    ("3.9", "==3.9.*"),
])
def test_poetry_to_pep440(poetry, expected):
    assert b._poetry_to_pep440(poetry) == expected


def test_poetry_or_constraint_raises():
    with pytest.raises(ValueError):
        b._poetry_to_pep440(">=2.7 || >=3.6")


def _repo(tmp_path, monkeypatch, files):
    d = tmp_path / "owner_repo_abc123"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(b, "CACHE", tmp_path)
    return {"clone_dir": d.name, "manifest_files": [n for n in files if n.endswith(".txt")]}


def test_poetry_python_312_excluded(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"pyproject.toml": '[tool.poetry.dependencies]\npython = "^3.12,<4"\nflask = "*"\n'})
    assert b._python_incompatibility(s).startswith("python_version_incompatible")


def test_requires_python_311_kept(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"pyproject.toml": '[project]\nname="x"\nrequires-python=">=3.11"\n'})
    assert b._python_incompatibility(s) is None


def test_setup_py_python_requires_excluded(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"setup.py": "setup(name='x', python_requires='>=3.12')\n"})
    assert "setup.py" in b._python_incompatibility(s)


def test_setup_cfg_python_requires_excluded(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"setup.cfg": "[options]\npython_requires = ==3.13.*\n"})
    assert "setup.cfg" in b._python_incompatibility(s)


def test_requirements_all_marker_gated_excluded(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"requirements.txt":
        'flask==3.1.2 ; python_version >= "3.12"\nblack==25.1.0 ; python_version >= "3.12"\n'})
    assert "gated off" in b._python_incompatibility(s)


def test_requirements_partly_gated_kept(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"requirements.txt":
        'flask==3.1.2\ncolorama ; platform_system == "Windows"\n'})
    assert b._python_incompatibility(s) is None


def test_no_constraint_kept(tmp_path, monkeypatch):
    s = _repo(tmp_path, monkeypatch, {"requirements.txt": "requests\n"})
    assert b._python_incompatibility(s) is None


def test_heavy_repo_by_own_name():
    s = {"repo": "pytorch/pytorch", "ecosystems": ["python"], "declared": ["numpy"],
         "required": ["numpy"], "manifest_files": ["requirements.txt"], "clone_dir": "none"}
    assert b._reason(s).startswith("heavy_packages")
