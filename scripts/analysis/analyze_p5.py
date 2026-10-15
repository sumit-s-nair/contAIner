import json
import subprocess
from pathlib import Path

manifest_path = Path("datasets/repo_corpus/corpus_manifest_v3.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

splits = manifest.get("splits", {})
print("=== P5.a Corpus Manifest Entries per Split ===")
for split_name, entries in splits.items():
    print(f"  {split_name}: {len(entries)} entries")

total_manifest_entries = sum(len(entries) for entries in splits.values())
print(f"Total manifest entries: {total_manifest_entries}")

# Manifest expected clone directory names:
# In repo_loader.py:
# safe_name = repo_name.replace("/", "_")
# local_path = self._cache_dir / f"{safe_name}_{sha_tag}"
# sha_tag is repo_entry.get("sha", "")[:8] or "latest"
# When cloned, it tries to rename to f"{safe_name}_{actual_sha[:8]}"

clones_dir = Path("cache/repo_clones")
clone_dirs = [d for d in clones_dir.iterdir() if d.is_dir()]
print(f"\nNumber of directories in cache/repo_clones: {len(clone_dirs)}")

# Map manifest entries to clone directories
# Each entry has repo "owner/name" -> safe prefix "owner_name"
manifest_repos = {}
for split_name, entries in splits.items():
    for e in entries:
        r = e["repo"]
        safe = r.replace("/", "_")
        manifest_repos[r] = (split_name, safe, e)

# Find matching clone directories for manifest repos
clone_dir_names = set(d.name for d in clone_dirs)

# Check matches by prefix (since suffix may be _latest or _<sha>)
matched_manifest = set()
matched_clone_dirs = set()

for r, (split, safe, e) in manifest_repos.items():
    # look for dir named safe, safe + "_latest", or safe + "_<sha>"
    matches = [d for d in clone_dirs if d.name == safe or d.name.startswith(safe + "_")]
    if matches:
        matched_manifest.add(r)
        for m in matches:
            matched_clone_dirs.add(m.name)

unmatched_manifest = [r for r in manifest_repos if r not in matched_manifest]
unmatched_clones = [d.name for d in clone_dirs if d.name not in matched_clone_dirs]

print(f"\nManifest entries with matching clone directory: {len(matched_manifest)}/{len(manifest_repos)}")
print(f"Manifest entries with NO matching clone directory: {len(unmatched_manifest)}")
if unmatched_manifest:
    print("Unmatched manifest entries (up to 20):")
    for r in unmatched_manifest[:20]:
        print(f"  {r} ({manifest_repos[r][0]})")

print(f"\nClone directories matching manifest: {len(matched_clone_dirs)}/{len(clone_dirs)}")
print(f"Clone directories NOT in manifest: {len(unmatched_clones)}")
if unmatched_clones:
    print("Unmatched clone directories (up to 20):")
    for d in unmatched_clones[:20]:
        print(f"  {d}")

# P5.b: How many clone directories contain a .git folder?
print("\n=== P5.b .git and rev-parse HEAD Check ===")
dirs_with_git = []
for d in clone_dirs:
    if (d / ".git").exists():
        dirs_with_git.append(d)

print(f"Clone directories containing .git folder: {len(dirs_with_git)}/{len(clone_dirs)}")

rev_parse_success = 0
examples = []

for d in dirs_with_git:
    try:
        r = subprocess.run(
            ["git", "-C", str(d), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            rev_parse_success += 1
            if len(examples) < 3:
                examples.append((d.name, r.stdout.strip()))
    except Exception:
        pass

print(f"git rev-parse HEAD successes: {rev_parse_success}/{len(dirs_with_git)}")
print("3 Example name -> SHA lines:")
for name, sha in examples:
    print(f"  {name} -> {sha}")
