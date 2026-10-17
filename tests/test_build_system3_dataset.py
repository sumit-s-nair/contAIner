"""Stage 3A: dataset builder pipeline (dry-run env, fake teacher, no Docker / GPU)."""

import json
from pathlib import Path

import pytest

from scripts import build_system3_dataset as B
from src.rl_env.dry_run_executor import DryRunExecutor
from src.sandbox.models import ExecutionResult, ExecutionStatus, RiskTier
from src.system3_command_generation.slot import parse_atomic_steps

STUB_MANIFEST_CMD = "python -m pip install -r requirements.txt"


@pytest.fixture
def failing_manifest_install(monkeypatch):
    """The stub's exact manifest install fails with a wheel error; anything else succeeds."""
    orig = DryRunExecutor._handle_safe

    def fake(self, step, classification):
        if step.command == STUB_MANIFEST_CMD:
            return ExecutionResult(step=step, status=ExecutionStatus.FAILED, tier=RiskTier.SAFE,
                                   classification=classification, returncode=1, verified=False,
                                   stderr="ERROR: Failed building wheel for ruamel.yaml\n"
                                          "[notice] A new release of pip is available")
        return orig(self, step, classification)

    monkeypatch.setattr(DryRunExecutor, "_handle_safe", fake)


@pytest.fixture
def workdir(tmp_path):
    corpus = {"splits": {
        "train": [{"repo": "a/one", "category": "manifest_present"},
                  {"repo": "a/two", "category": "manifest_less"}],
        "val": [{"repo": "b/one", "category": "manifest_present"}],
        "test": [{"repo": "c/one", "category": "manifest_present"}],
    }}
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))
    docs = tmp_path / "out" / "docs"
    docs.mkdir(parents=True)
    (docs / "pip_install.help.txt").write_text(
        "Install Options:\n"
        "  -r, --requirement <file>    Install from the given requirements file.\n"
        "  --prefer-binary             Prefer binary packages over source packages.\n"
        "  --only-binary <format_control>\n"
        "                              Do not use source packages.\n")
    return tmp_path


def _args(wd, *extra):
    return ["--corpus", str(wd / "corpus.json"), "--out", str(wd / "out"), "--dry_run", "--workers", "1", *extra]


def test_collect_recover_export(workdir, failing_manifest_install):
    B.main(["--phase", "collect", *_args(workdir)])
    cases = B._read_jsonl(workdir / "out" / "cases.jsonl")
    assert {c["split"] for c in cases} == {"train", "val", "test"}
    seeds = [c for c in cases if c["recovery_seed"]]
    assert seeds and all(c["planned"]["target"] == "requirements.txt" for c in seeds)
    ok = [c for c in cases if c["accepted"]]
    assert ok and all(c["result"]["exec_status"] == "success" for c in ok)
    # every case carries what the prompt needs
    for c in cases:
        assert c["context"]["python_version"] and isinstance(c["docs"], list)
        parse_atomic_steps(json.dumps({"steps": c["reference_steps"]}))

    # restart is idempotent
    B.main(["--phase", "collect", *_args(workdir)])
    assert len(B._read_jsonl(workdir / "out" / "cases.jsonl")) == len(cases)

    B.main(["--phase", "recover", *_args(workdir, "--teacher", "fake", "--k", "4")])
    rec = B._read_jsonl(workdir / "out" / "recovery.jsonl")
    assert rec and {r["split"] for r in rec} <= {"train", "val"}       # test seeds never get teacher labels
    r = rec[0]
    assert r["status"] == "ok"
    assert r["context"]["previous"]["commands"] == [STUB_MANIFEST_CMD]
    assert "Failed building wheel" in r["context"]["previous"]["stderr"]
    srcs = [c["source"] for c in r["candidates"]]
    assert srcs[0] == "stub_repeat" and srcs.count("teacher") == 4
    stub = r["candidates"][0]
    assert stub["result"]["exec_status"] == "failed"                 # repeating fails again
    assert any(c.get("policy") for c in r["candidates"])              # sudo candidate rejected, not run
    assert any(c.get("parse_error") for c in r["candidates"])
    assert any(c.get("duplicate") for c in r["candidates"])
    best = r["candidates"][r["best_idx"]]
    assert "--prefer-binary" in best["steps"][0]["command"]
    assert best["result"]["exec_status"] == "success"

    B.main(["--phase", "export", *_args(workdir, "--recovery_upsample", "1")])
    out = workdir / "out"
    train = B._read_jsonl(out / "train.jsonl")
    test_cases = B._read_jsonl(out / "test_cases.jsonl")
    assert train and test_cases
    assert not ({row["id"].split("#")[0] for row in train} & {"c/one"})   # no test repo in train
    rec_rows = [row for row in train if row["kind"] == "recovery"]
    assert rec_rows and "--prefer-binary" in rec_rows[0]["messages"][-1]["content"]
    assert "Previous attempt (failed)" in rec_rows[0]["messages"][1]["content"]
    for row in train:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        parse_atomic_steps(row["messages"][-1]["content"])
    stats = json.loads((out / "stats.json").read_text())
    assert stats["files"]["train"] == len(train)


def test_evaluate_system3_stub_baseline(workdir, failing_manifest_install):
    from scripts import evaluate_system3 as E
    B.main(["--phase", "collect", *_args(workdir, "--splits", "test")])
    B.main(["--phase", "export", *_args(workdir)])
    out = workdir / "eval"
    E.main(["--cases", str(workdir / "out" / "test_cases.jsonl"), "--corpus", str(workdir / "corpus.json"),
            "--docs_dir", str(workdir / "out" / "docs"), "--no_base", "--dry_run", "--out_dir", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    rec = summary["stub"]["recovery"]
    assert rec["n"] >= 1 and rec["repeats_failed_rate"] == 1.0 and rec["exec_success_rate"] == 0.0
    assert summary["stub"]["canonical"]["exec_success_rate"] == 1.0


def test_non_python_repo_has_no_cases():
    from types import SimpleNamespace
    env = SimpleNamespace(
        _repo_entry={"repo": "react/create-react-app"},
        _obs=SimpleNamespace(manifest_paths={"javascript": ["package.json"]}),
        _instance=SimpleNamespace(manifest=SimpleNamespace(ecosystems={"javascript": object()})),
    )
    assert B.canonical_cases(env, None) == []
