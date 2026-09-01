# System 4: Sandbox Safety Gate (`sandbox_root`)

## Three-Tier Classifier Design
System 4 acts as the ultimate gatekeeper, evaluating each `AtomicStep` through `CommandRiskClassifier` in `src/sandbox/classifier.py`. It implements a strict three-pass hierarchy:

1. **BLOCKED**: Hard denylist via compiled regex; final, no override. Matches catastrophic actions or out-of-scope filesystem writes. Execution is unconditionally halted.
2. **REVIEW**: Requires user confirmation. Evaluated via a pattern list check, force-flag verb escalation, or if the planner flagged the step as `destructive`.
3. **SAFE**: Known-safe action allowlist. 
- **Fail-Safe Default**: Any unknown commands default to REVIEW.

## Force-Flag Scoping and Regex Bugs
Initially, a blanket `--force` match triggered REVIEW. The fix scoped this check to apply *only* when the first meaningful verb of the command is in a higher-risk set (e.g., `rm`, `delete`, `push`, `overwrite`). This avoids friction on routine commands like `npm install --force`.

During testing, three critical regex bugs were found and fixed. Here are the concrete exploit walkthroughs:
- **`rm -rf` scope exploit**: The original regex `rm\s+-r.*f` was overly broad. A routine command like `rm -rf ./dist` triggered a BLOCKED tier, frustrating users. The fix scoped it explicitly to root-level paths, such as `rm -rf /` or `rm -rf /*`, using the pattern `[\"']?/[\*]*[\"']?(?:\s|$)`. Now, `rm -rf /` is BLOCKED, but `rm -rf ./dist` falls back correctly to the REVIEW tier.
- **Fork-bomb bug**: The ORIGINAL pattern had an invalid `\1` backreference causing `re.PatternError` at import time. The FIX removed the backreference and replaced it with a structural pattern detecting `word|word&` inside a shell function body: `\(\s*\)\s*\{[^}]*\|\s*[^\s|&;]+\s*&`.
- **`--env` false positive**: The `env` command allowlist was written loosely as `env\b`. This meant a command like `npm run build --env production` incorrectly triggered the SAFE tier (bypassing REVIEW). The fix anchored the pattern to the start of the string: `^\s*(env|printenv)\b` to only match when `env` is used as the primary command.

## Shared Confirmation Gate
The system uses a shared confirmation gate design in `src/sandbox/confirmation.py`. A single function manages both:
1. Low-confidence disambiguation prompts from System 1.
2. REVIEW-tier risk confirmations from System 4.
This ensures a unified user experience when manual intervention is required.

## Concrete Example: Command Classification
Here are 3 real commands evaluated through the classifier:

1. **Command**: `pip install requests`
   - **Tier**: `SAFE`
   - **Reason**: Matches known safe action type (`\b(pip|pip3)\s+install\b`).

2. **Command**: `rm -rf node_modules`
   - **Tier**: `REVIEW`
   - **Reason**: Deletes files permanently — no Trash/recycle bin (Matches `\brm\s`).

3. **Command**: `:(){ :|:& };:`
   - **Tier**: `BLOCKED`
   - **Reason**: Canonical Bash fork bomb detected — would crash the system by exhausting process slots (Matches `:\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:.*&\s*\}`).
