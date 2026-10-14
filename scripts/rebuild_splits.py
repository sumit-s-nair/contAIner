import json
import re
import random
import os
from collections import defaultdict
from pathlib import Path

# Audit logic from tools.audit_spans
def audit_row(row):
    instruction = row.get("instruction", "")
    entities = row.get("entities", {})
    entity_spans = row.get("entity_spans", {})
    if not isinstance(entity_spans, dict):
        entity_spans = {}
    
    row_bucket = "VALID"
    
    for ent_type, ent_val in entities.items():
        if ent_val is None:
            continue
            
        span = entity_spans.get(ent_type)
        if span is not None:
            start, end, text = span.get("start"), span.get("end"), span.get("text")
            if start is not None and end is not None and text is not None:
                actual_text = instruction[start:end]
                if actual_text == text:
                    ent_bucket = "VALID"
                else:
                    occurrences = list(re.finditer(re.escape(text), instruction))
                    if len(occurrences) == 1:
                        ent_bucket = "FIXABLE_OFFSET"
                    else:
                        ent_bucket = "NEEDS_REVIEW"
            else:
                ent_bucket = "NEEDS_REVIEW"
        else:
            occurrences = list(re.finditer(re.escape(str(ent_val)), instruction, re.IGNORECASE))
            if len(occurrences) == 1:
                ent_bucket = "FIXABLE_MISSING_SPAN"
            else:
                ent_bucket = "NEEDS_REVIEW"
                
        severity = {"VALID": 0, "FIXABLE_OFFSET": 1, "FIXABLE_MISSING_SPAN": 2, "NEEDS_REVIEW": 3}
        if severity[ent_bucket] > severity[row_bucket]:
            row_bucket = ent_bucket

    return row_bucket

def norm_inst(s):
    return re.sub(r'\s+', '', s).lower()

def main():
    random.seed(42)
    base_dir = Path(r"c:\Users\sumit\Documents\contAIner\datasets")
    
    intent_in = base_dir / "intent-dataset" / "data"
    cmd_in = base_dir / "command-dataset" / "data"
    
    intent_out = base_dir / "intent-dataset-v2" / "data"
    cmd_out = base_dir / "command-dataset-v2" / "data"
    
    intent_out.mkdir(parents=True, exist_ok=True)
    cmd_out.mkdir(parents=True, exist_ok=True)

    report_lines = ["# Rebuild Report\n"]

    # ==========================================
    # PROCESS INTENT DATASET
    # ==========================================
    intent_rows_raw = []
    for split in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        p = intent_in / split
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        intent_rows_raw.append(json.loads(line))
                        
    # Drop exact duplicate rows (keep one)
    intent_exact_seen = set()
    intent_unique = []
    for row in intent_rows_raw:
        row_str = json.dumps(row, sort_keys=True)
        if row_str not in intent_exact_seen:
            intent_exact_seen.add(row_str)
            intent_unique.append(row)
            
    # Group by normalized instruction
    intent_groups = defaultdict(list)
    for row in intent_unique:
        intent_groups[norm_inst(row.get("instruction", ""))].append(row)
        
    # Check conflicts
    conflicts = []
    for inst, rows in intent_groups.items():
        intents = set(r.get("intent_type") for r in rows)
        if len(intents) > 1:
            conflicts.append({"normalized_instruction": inst, "rows": rows})
            
    with open(r"c:\Users\sumit\Documents\contAIner\scratch\intent_label_conflicts.jsonl", "w", encoding="utf-8") as f:
        for c in conflicts:
            f.write(json.dumps(c) + "\n")
            
    # Stratify by intent_type
    intent_groups_by_type = defaultdict(list)
    for inst, rows in intent_groups.items():
        # majority intent
        intent_counts = defaultdict(int)
        for r in rows:
            intent_counts[r.get("intent_type")] += 1
        maj_intent = max(intent_counts.items(), key=lambda x: x[1])[0]
        intent_groups_by_type[maj_intent].append(rows)
        
    intent_splits = {"train": [], "validation": [], "test": []}
    
    for itype, group_list in intent_groups_by_type.items():
        # Sort for determinism before shuffle
        group_list.sort(key=lambda g: g[0].get("instruction", ""))
        random.shuffle(group_list)
        
        n = len(group_list)
        n_test = int(n * 0.1)
        n_val = int(n * 0.1)
        
        for g in group_list[:n_test]:
            intent_splits["test"].extend(g)
        for g in group_list[n_test:n_test+n_val]:
            intent_splits["validation"].extend(g)
        for g in group_list[n_test+n_val:]:
            intent_splits["train"].extend(g)
            
    # Write Intent V2
    for split, rows in intent_splits.items():
        with open(intent_out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
                
    # ==========================================
    # PROCESS COMMAND DATASET
    # ==========================================
    cmd_rows_raw = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        p = cmd_in / split
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cmd_rows_raw.append(json.loads(line))
                        
    # Drop normalized exact duplicates
    cmd_exact_seen = set()
    cmd_unique = []
    for row in cmd_rows_raw:
        row_str = json.dumps(row, sort_keys=True)
        if row_str not in cmd_exact_seen:
            cmd_exact_seen.add(row_str)
            # Add span_audit
            bucket = audit_row(row)
            row["span_audit"] = "needs_review" if bucket == "NEEDS_REVIEW" else "pass"
            cmd_unique.append(row)
            
    # Group by normalized instruction
    cmd_groups = defaultdict(list)
    for row in cmd_unique:
        cmd_groups[norm_inst(row.get("instruction", ""))].append(row)
        
    # Group classification: does this group contain ANY needs_review?
    cmd_groups_by_strata = defaultdict(list)
    
    for inst, rows in cmd_groups.items():
        has_needs_review = any(r.get("span_audit") == "needs_review" for r in rows)
        
        # Determine majority source and intent_type for stratification
        src_counts = defaultdict(int)
        intent_counts = defaultdict(int)
        for r in rows:
            src_counts[r.get("source")] += 1
            intent_counts[r.get("intent_type")] += 1
            
        maj_src = max(src_counts.items(), key=lambda x: x[1])[0] if src_counts else "unknown"
        maj_intent = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else "unknown"
        
        strata_key = (maj_src, maj_intent, has_needs_review)
        cmd_groups_by_strata[strata_key].append(rows)
        
    cmd_splits = {"train": [], "val": [], "test": []}
    
    for (src, intent, has_review), group_list in cmd_groups_by_strata.items():
        group_list.sort(key=lambda g: g[0].get("instruction", ""))
        random.shuffle(group_list)
        
        if has_review:
            # allocate entirely to train to avoid in val/test
            for g in group_list:
                cmd_splits["train"].extend(g)
        else:
            n = len(group_list)
            # 70 / 15 / 15
            n_val = int(n * 0.15)
            n_test = int(n * 0.15)
            for g in group_list[:n_val]:
                cmd_splits["val"].extend(g)
            for g in group_list[n_val:n_val+n_test]:
                cmd_splits["test"].extend(g)
            for g in group_list[n_val+n_test:]:
                cmd_splits["train"].extend(g)
                
    # Write Command V2
    for split, rows in cmd_splits.items():
        with open(cmd_out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
                
    # ==========================================
    # VALIDATION & REPORTING
    # ==========================================
    # Assert zero overlap
    def get_norm_insts(rows):
        return set(norm_inst(r.get("instruction", "")) for r in rows)
        
    i_tr = get_norm_insts(intent_splits["train"])
    i_va = get_norm_insts(intent_splits["validation"])
    i_te = get_norm_insts(intent_splits["test"])
    
    assert len(i_tr.intersection(i_va)) == 0, "Intent Train/Val overlap"
    assert len(i_tr.intersection(i_te)) == 0, "Intent Train/Test overlap"
    assert len(i_va.intersection(i_te)) == 0, "Intent Val/Test overlap"
    
    c_tr = get_norm_insts(cmd_splits["train"])
    c_va = get_norm_insts(cmd_splits["val"])
    c_te = get_norm_insts(cmd_splits["test"])
    
    assert len(c_tr.intersection(c_va)) == 0, "Cmd Train/Val overlap"
    assert len(c_tr.intersection(c_te)) == 0, "Cmd Train/Test overlap"
    assert len(c_va.intersection(c_te)) == 0, "Cmd Val/Test overlap"
    
    report_lines.append("## Intent Dataset\n")
    report_lines.append(f"**Rows per split (Before -> After):**\n")
    old_intent_sizes = {"train": 7713, "validation": 969, "test": 970} # approx based on earlier runs
    for s in ["train", "validation", "test"]:
        new_count = len(intent_splits[s])
        old_count = sum(1 for _ in open(intent_in / f"{s}.jsonl", "r")) if (intent_in / f"{s}.jsonl").exists() else 0
        report_lines.append(f"- {s}: {old_count} -> {new_count}\n")
        
    report_lines.append("\n**Intent Types per split (After):**\n")
    for s in ["train", "validation", "test"]:
        report_lines.append(f"### {s}\n")
        counts = defaultdict(int)
        for r in intent_splits[s]:
            counts[r.get("intent_type")] += 1
        for k, v in counts.items():
            report_lines.append(f"- {k}: {v}\n")
            
    report_lines.append("\n## Command Dataset\n")
    report_lines.append(f"**Rows per split (Before -> After):**\n")
    for s in ["train", "val", "test"]:
        new_count = len(cmd_splits[s])
        old_count = sum(1 for _ in open(cmd_in / f"{s}.jsonl", "r")) if (cmd_in / f"{s}.jsonl").exists() else 0
        report_lines.append(f"- {s}: {old_count} -> {new_count}\n")
        
    for s in ["train", "val", "test"]:
        report_lines.append(f"\n### {s} (After)\n")
        icounts = defaultdict(int)
        ocounts = defaultdict(int)
        shcounts = defaultdict(int)
        srccounts = defaultdict(int)
        
        for r in cmd_splits[s]:
            icounts[r.get("intent_type")] += 1
            ocounts[r.get("os")] += 1
            shcounts[r.get("shell")] += 1
            srccounts[r.get("source")] += 1
            
        report_lines.append("**Intent Types:**\n")
        for k, v in icounts.items(): report_lines.append(f"- {k}: {v}\n")
        report_lines.append("**OS:**\n")
        for k, v in ocounts.items(): report_lines.append(f"- {k}: {v}\n")
        report_lines.append("**Shells:**\n")
        for k, v in shcounts.items(): report_lines.append(f"- {k}: {v}\n")
        report_lines.append("**Sources:**\n")
        for k, v in srccounts.items(): report_lines.append(f"- {k}: {v}\n")
        
    with open(r"c:\Users\sumit\Documents\contAIner\scratch\rebuild_report.md", "w", encoding="utf-8") as f:
        f.writelines(report_lines)
        
if __name__ == "__main__":
    main()
