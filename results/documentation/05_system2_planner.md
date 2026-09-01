# System 2: Planner

## WorkflowTemplate DSL
System 2 generates a high-level sequence of abstract `PlannedStep` actions using canonical Workflow Templates that enforce hard ordering constraints via a Directed Acyclic Graph (DAG) approach. 

The three teammate-provided canonical flows are encoded as:
1. **`setup_project`**: Initializing a codebase.
2. **`fix_environment`**: Diagnosing and repairing a broken state.
3. **`setup_environment`**: Upgrading or altering an existing environment.

## Structural Validator
The structural validator (`src/system2_planner/validator.py`) enforces hard ordering rules against the `TemplateInstance` to ensure safety before System 3 generates concrete shell commands. 

- **`setup_project` rules**: If dependency conflicts exist (`has_conflicts`), it MUST `ISOLATE` the environment before it `INSTALL`s dependencies. Otherwise, it just MUST `INSTALL`.
- **`fix_environment` rules**: Must `DETECT` before attempting to `FIX` or `ESCALATE`.
- **`setup_environment` rules**: Must `CHECK` the environment before attempting to `INSTALL` or `UPDATE`.

### The `fix_environment` Validator Bug
During testing, a critical bug was found in the `fix_environment` validator: it was silently missing the reinstall-before-detect check. The policy could theoretically attempt to blindly reinstall packages before actually diagnosing the problem. 
**The Fix**: Explicit ordering constraints were added to ensure `ISOLATE` and `INSTALL` cannot precede `DETECT` (`_require_ordering(actual_types, ActionType.DETECT, ActionType.ISOLATE, errors)`).

## Concrete Example: Sequence Validation

**Valid Sequence (`setup_project` with conflicts)**:
- `ISOLATE`
- `INSTALL`
*Result: Passes validation. The mandatory isolation happens before the install.*

**Invalid Sequence (`fix_environment`)**:
- `INSTALL`
- `DETECT`
- `FIX`
*Result: Rejected.*
*Structured Rejection Reason:*
```text
["Ordering violation: 'detect' requires 'install' to occur first."] 
```
*(Wait, the code output is actually: `Ordering violation: 'detect' must occur before 'install'.`)*
Let me correct that:
*Structured Rejection Reason:*
```json
{
  "is_valid": false,
  "errors": [
    "Ordering violation: 'detect' must occur before 'install'."
  ]
}
```
