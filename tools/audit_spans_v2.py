import json
import os
import re
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

def audit_row(row):
    instruction = row.get("instruction", "")
    entities = row.get("entities", {})
    entity_spans = row.get("entity_spans", {})
    if not isinstance(entity_spans, dict):
        entity_spans = {}
    
    row_bucket = "VALID"
    fixed_entity_spans = deepcopy(entity_spans)
    entity_buckets = {}
    
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
                        fixed_entity_spans[ent_type] = {
                            "start": occurrences[0].start(),
                            "end": occurrences[0].end(),
                            "text": text
                        }
                    else:
                        ent_bucket = "NEEDS_REVIEW"
            else:
                ent_bucket = "NEEDS_REVIEW"
        else:
            occurrences = list(re.finditer(re.escape(str(ent_val)), instruction, re.IGNORECASE))
            if len(occurrences) == 1:
                ent_bucket = "FIXABLE_MISSING_SPAN"
                fixed_entity_spans[ent_type] = {
                    "start": occurrences[0].start(),
                    "end": occurrences[0].end(),
                    "text": occurrences[0].group(0)
                }
            else:
                ent_bucket = "NEEDS_REVIEW"
                
        entity_buckets[ent_type] = ent_bucket
        
        severity = {"VALID": 0, "FIXABLE_OFFSET": 1, "FIXABLE_MISSING_SPAN": 2, "NEEDS_REVIEW": 3}
        if severity[ent_bucket] > severity[row_bucket]:
            row_bucket = ent_bucket

    return row_bucket, fixed_entity_spans, entity_buckets

def simulate_old_logic(row):
    old_row = deepcopy(row)
    instruction = old_row.get("instruction", "")
    entities = old_row.get("entities", {})
    old_spans = {}
    for ent_type, ent_val in entities.items():
        if ent_val is None:
            old_spans[ent_type] = None
        else:
            occurrences = list(re.finditer(re.escape(str(ent_val)), instruction, re.IGNORECASE))
            if len(occurrences) == 1:
                old_spans[ent_type] = {
                    "start": occurrences[0].start(),
                    "end": occurrences[0].end(),
                    "text": occurrences[0].group(0)
                }
            else:
                old_spans[ent_type] = None
    
    old_row["entity_spans"] = old_spans
    return old_row

def main():
    jsonl_files = list(Path("datasets").rglob("*.jsonl"))
    
    alias_table = defaultdict(set)
    processed_data = []
    
    old_logic_additional_needs_review = 0
    total_rows = 0
    
    # Pass 1: Audit rows and build alias table
    for filepath in jsonl_files:
        if "corrected" in filepath.name or "audit_report" in filepath.name or filepath.name == "dataset_audit_report.json":
            continue
            
        with open(filepath, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                total_rows += 1
                row = json.loads(line)
                
                bucket, fixed_spans, ent_buckets = audit_row(row)
                
                # Populate alias table
                for ent_type, ent_bucket in ent_buckets.items():
                    if ent_bucket in ("VALID", "FIXABLE_OFFSET", "FIXABLE_MISSING_SPAN"):
                        ent_val = row["entities"][ent_type]
                        text = fixed_spans[ent_type]["text"]
                        alias_table[str(ent_val)].add(text)
                        
                # Evaluate old logic
                old_row = simulate_old_logic(row)
                old_bucket, _, _ = audit_row(old_row)
                if old_bucket == "NEEDS_REVIEW" and bucket != "NEEDS_REVIEW":
                    old_logic_additional_needs_review += 1
                
                processed_data.append({
                    "filepath": filepath,
                    "row": row,
                    "bucket": bucket,
                    "fixed_spans": fixed_spans,
                    "ent_buckets": ent_buckets
                })

    # Pass 2: Resolve NEEDS_REVIEW using alias table
    resolved_needs_review_count = 0
    
    for item in processed_data:
        if item["bucket"] == "NEEDS_REVIEW":
            row = item["row"]
            instruction = row.get("instruction", "")
            fixed_spans = item["fixed_spans"]
            ent_buckets = item["ent_buckets"]
            
            new_row_bucket = "VALID"
            
            for ent_type, ent_bucket in ent_buckets.items():
                if ent_bucket == "NEEDS_REVIEW":
                    ent_val = row["entities"][ent_type]
                    aliases = alias_table.get(str(ent_val), set())
                    
                    possible_matches = []
                    for alias in aliases:
                        occurrences = list(re.finditer(re.escape(alias), instruction))
                        for occ in occurrences:
                            possible_matches.append({
                                "start": occ.start(),
                                "end": occ.end(),
                                "text": occ.group(0)
                            })
                            
                    if len(possible_matches) == 1:
                        # Exactly one alias matched at exactly one position
                        ent_buckets[ent_type] = "FIXABLE_MISSING_SPAN"
                        fixed_spans[ent_type] = possible_matches[0]
                
                severity = {"VALID": 0, "FIXABLE_OFFSET": 1, "FIXABLE_MISSING_SPAN": 2, "NEEDS_REVIEW": 3}
                if severity.get(ent_buckets[ent_type], 0) > severity[new_row_bucket]:
                    new_row_bucket = ent_buckets[ent_type]
            
            if item["bucket"] == "NEEDS_REVIEW" and new_row_bucket != "NEEDS_REVIEW":
                resolved_needs_review_count += 1
            
            item["bucket"] = new_row_bucket

    # Pass 3: Write corrected files and report
    counts = {"VALID": 0, "FIXABLE_OFFSET": 0, "FIXABLE_MISSING_SPAN": 0, "NEEDS_REVIEW": 0}
    counts_by_file = defaultdict(lambda: {"VALID": 0, "FIXABLE_OFFSET": 0, "FIXABLE_MISSING_SPAN": 0, "NEEDS_REVIEW": 0})
    needs_review_rows = []
    
    # Group by filepath for writing
    files_to_write = defaultdict(list)
    for item in processed_data:
        files_to_write[item["filepath"]].append(item)
        counts[item["bucket"]] += 1
        counts_by_file[str(item["filepath"])][item["bucket"]] += 1
        
        if item["bucket"] == "NEEDS_REVIEW":
            needs_review_rows.append({
                "file": str(item["filepath"]),
                "instruction": item["row"].get("instruction"),
                "entities": item["row"].get("entities"),
                "entity_spans": item["row"].get("entity_spans")
            })
            
    for filepath, items in files_to_write.items():
        corrected_filepath = filepath.with_name(filepath.stem + "_corrected" + filepath.suffix)
        with open(corrected_filepath, "w", encoding="utf-8") as fout:
            for item in items:
                if item["bucket"] != "NEEDS_REVIEW":
                    corrected_row = deepcopy(item["row"])
                    corrected_row["entity_spans"] = item["fixed_spans"]
                    fout.write(json.dumps(corrected_row) + "\n")
                    
    # Output report
    report = {
        "counts": counts,
        "counts_by_file": dict(counts_by_file),
        "resolved_needs_review_count": resolved_needs_review_count,
        "needs_review_rows": needs_review_rows,
        "old_logic_additional_missed_rows": old_logic_additional_needs_review
    }
    with open("artifacts/dataset_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    # Output alias table
    serializable_alias_table = {k: list(v) for k, v in alias_table.items()}
    with open("artifacts/alias_table.json", "w", encoding="utf-8") as f:
        json.dump(serializable_alias_table, f, indent=2)
        
    print(f"--- Audit Summary ---")
    print(f"Total rows processed: {total_rows}")
    print(f"VALID: {counts['VALID']}")
    print(f"FIXABLE_OFFSET: {counts['FIXABLE_OFFSET']}")
    print(f"FIXABLE_MISSING_SPAN: {counts['FIXABLE_MISSING_SPAN']}")
    print(f"NEEDS_REVIEW: {counts['NEEDS_REVIEW']}")
    if total_rows > 0:
        print(f"NEEDS_REVIEW %: {counts['NEEDS_REVIEW'] / total_rows * 100:.2f}%")
        
    print(f"\n--- Alias Table Resolution ---")
    print(f"Rows resolved from NEEDS_REVIEW to FIXABLE_MISSING_SPAN via aliases: {resolved_needs_review_count}")
        
    print(f"\n--- Blast Radius of Old Logic ---")
    print(f"Additional rows that old logic silently mislabels as all-O (and would be caught by this audit): {old_logic_additional_needs_review}")

if __name__ == '__main__':
    main()
