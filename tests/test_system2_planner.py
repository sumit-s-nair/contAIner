"""
tests/test_system2_planner.py
=============================
Tests for System 2 Canonical Workflow Templates and Structural Validator.
"""

from src.system2_planner.models import ActionType, PlannedStep, TemplateInstance
from src.system2_planner.templates import (
    SETUP_PROJECT_TEMPLATE,
    FIX_ENVIRONMENT_TEMPLATE,
    SETUP_ENVIRONMENT_TEMPLATE,
)
from src.system2_planner.validator import validate_plan
from src.repo_scan.models import RepoManifest

def _make_manifest(conflicts=None):
    if conflicts is None:
        conflicts = []
    manifest = RepoManifest(
        ecosystems={},
        environment_configs=[],
        conflicts=conflicts
    )
    return manifest

def test_setup_project_no_conflict_valid():
    manifest = _make_manifest(conflicts=[])
    instance = TemplateInstance(SETUP_PROJECT_TEMPLATE, manifest)
    plan = [PlannedStep(ActionType.INSTALL, "Install deps")]
    result = validate_plan(instance, plan)
    assert result.is_valid
    assert len(result.errors) == 0

def test_setup_project_no_conflict_missing_install():
    manifest = _make_manifest(conflicts=[])
    instance = TemplateInstance(SETUP_PROJECT_TEMPLATE, manifest)
    plan = [PlannedStep(ActionType.ISOLATE, "Isolate")]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert any("Must INSTALL" in err for err in result.errors)

def test_setup_project_with_conflict_valid():
    manifest = _make_manifest(conflicts=["conflict1"])
    instance = TemplateInstance(SETUP_PROJECT_TEMPLATE, manifest)
    plan = [
        PlannedStep(ActionType.ISOLATE, "Isolate env"),
        PlannedStep(ActionType.INSTALL, "Install deps")
    ]
    result = validate_plan(instance, plan)
    assert result.is_valid

def test_setup_project_with_conflict_missing_isolate():
    manifest = _make_manifest(conflicts=["conflict1"])
    instance = TemplateInstance(SETUP_PROJECT_TEMPLATE, manifest)
    plan = [PlannedStep(ActionType.INSTALL, "Install deps")]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert any("must ISOLATE" in err for err in result.errors)

def test_setup_project_with_conflict_wrong_order():
    manifest = _make_manifest(conflicts=["conflict1"])
    instance = TemplateInstance(SETUP_PROJECT_TEMPLATE, manifest)
    plan = [
        PlannedStep(ActionType.INSTALL, "Install deps"),
        PlannedStep(ActionType.ISOLATE, "Isolate env")
    ]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert any("Ordering violation" in err for err in result.errors)

def test_fix_environment_detect_fix_valid():
    manifest = _make_manifest()
    instance = TemplateInstance(FIX_ENVIRONMENT_TEMPLATE, manifest)
    plan = [
        PlannedStep(ActionType.DETECT, "Detect issue"),
        PlannedStep(ActionType.FIX, "Fix issue")
    ]
    result = validate_plan(instance, plan)
    assert result.is_valid
    
def test_fix_environment_detect_escalate_valid():
    manifest = _make_manifest()
    instance = TemplateInstance(FIX_ENVIRONMENT_TEMPLATE, manifest)
    plan = [
        PlannedStep(ActionType.DETECT, "Detect issue"),
        PlannedStep(ActionType.ESCALATE, "Escalate")
    ]
    result = validate_plan(instance, plan)
    assert result.is_valid
    
def test_fix_environment_missing_detect():
    manifest = _make_manifest()
    instance = TemplateInstance(FIX_ENVIRONMENT_TEMPLATE, manifest)
    plan = [PlannedStep(ActionType.FIX, "Fix issue")]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert any("requires 'detect' to occur first" in err.lower() or "expected detect" in err.lower() or "missing detect" in err.lower() or "detect" in err.lower() for err in result.errors)

def test_fix_environment_reinstall_before_detect():
    manifest = _make_manifest()
    instance = TemplateInstance(FIX_ENVIRONMENT_TEMPLATE, manifest)
    plan = [
        PlannedStep(ActionType.ISOLATE, "Isolate"),
        PlannedStep(ActionType.INSTALL, "Install"),
        PlannedStep(ActionType.DETECT, "Detect issue")
    ]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert len(result.errors) > 0

def test_setup_environment_valid():
    manifest = _make_manifest()
    instance = TemplateInstance(SETUP_ENVIRONMENT_TEMPLATE, manifest)
    plan = [
        PlannedStep(ActionType.CHECK, "Check req"),
        PlannedStep(ActionType.INSTALL, "Install if missing")
    ]
    result = validate_plan(instance, plan)
    assert result.is_valid
    
def test_setup_environment_update_without_check():
    manifest = _make_manifest()
    instance = TemplateInstance(SETUP_ENVIRONMENT_TEMPLATE, manifest)
    plan = [PlannedStep(ActionType.UPDATE, "Update req")]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert any("check" in err.lower() for err in result.errors)

def test_setup_environment_install_without_check():
    manifest = _make_manifest()
    instance = TemplateInstance(SETUP_ENVIRONMENT_TEMPLATE, manifest)
    plan = [PlannedStep(ActionType.INSTALL, "Install req")]
    result = validate_plan(instance, plan)
    assert not result.is_valid
    assert any("check" in err.lower() for err in result.errors)

