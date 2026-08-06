import json
import os
import re
from pathlib import Path
from copy import deepcopy

def audit_row(row):
    instruction = row.get("instruction", "")
    entities = row.get("entities", {})
    entity_spans = row.get("entity_spans", {})
    if not isinstance(entity_spans, dict):
        entity_spans = {}
    
    row_bucket = "VALID"
    fixed_entity_spans = deepcopy(entity_spans)
    
    for ent_type, ent_val in entities.items():
        if ent_val is None:
            continue
            
        span = entity_spans.get(ent_type)
        if span is not None:
            # Check if valid
            start, end, text = span.get("start"), span.get("end"), span.get("text")
            # Account for missing start/end
            if start is not None and end is not None and text is not None:
                actual_text = instruction[start:end]
                if actual_text == text:
                    # VALID for this entity
                    ent_bucket = "VALID"
                else:
                    # FIXABLE_OFFSET?
                    # text matches instruction substring?
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
            # FIXABLE_MISSING_SPAN?
            # case insensitive search for ent_val
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
                
        # Update row bucket severity
        severity = {"VALID": 0, "FIXABLE_OFFSET": 1, "FIXABLE_MISSING_SPAN": 2, "NEEDS_REVIEW": 3}
        if severity[ent_bucket] > severity[row_bucket]:
            row_bucket = ent_bucket

    return row_bucket, fixed_entity_spans

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
    
    # We must preserve keys for null entities so as to closely mimic old behaviour
    old_row["entity_spans"] = old_spans
    return old_row

def main():
    jsonl_files = list(Path("datasets").rglob("*.jsonl"))
    
    counts = {"VALID": 0, "FIXABLE_OFFSET": 0, "FIXABLE_MISSING_SPAN": 0, "NEEDS_REVIEW": 0}
    needs_review_rows = []
    
    old_logic_additional_needs_review = 0
    total_rows = 0
    
    for filepath in jsonl_files:
        # Avoid processing corrected files or the report itself if they exist
        if "corrected" in filepath.name or "audit_report" in filepath.name or filepath.name == "dataset_audit_report.json":
            continue
            
        corrected_filepath = filepath.with_name(filepath.stem + "_corrected" + filepath.suffix)
        
        with open(filepath, "r", encoding="utf-8") as fin, \
             open(corrected_filepath, "w", encoding="utf-8") as fout:
            
            for line in fin:
                if not line.strip():
                    continue
                total_rows += 1
                row = json.loads(line)
                
                bucket, fixed_spans = audit_row(row)
                counts[bucket] += 1
                
                if bucket == "NEEDS_REVIEW":
                    needs_review_rows.append({
                        "file": str(filepath),
                        "instruction": row.get("instruction"),
                        "entities": row.get("entities"),
                        "entity_spans": row.get("entity_spans")
                    })
                else:
                    # Write corrected row to new file
                    corrected_row = deepcopy(row)
                    corrected_row["entity_spans"] = fixed_spans
                    fout.write(json.dumps(corrected_row) + "\n")
                    
                # Evaluate old logic
                old_row = simulate_old_logic(row)
                old_bucket, _ = audit_row(old_row)
                # If the old logic would result in NEEDS_REVIEW (i.e. it fails to extract a valid span),
                # but our current dataset is NOT NEEDS_REVIEW (meaning it caught something old logic missed).
                if old_bucket == "NEEDS_REVIEW" and bucket != "NEEDS_REVIEW":
                    old_logic_additional_needs_review += 1
                    
    # Output report
    report = {
        "counts": counts,
        "needs_review_rows": needs_review_rows,
        "old_logic_additional_missed_rows": old_logic_additional_needs_review
    }
    with open("artifacts/dataset_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"--- Audit Summary ---")
    print(f"Total rows processed: {total_rows}")
    print(f"VALID: {counts['VALID']}")
    print(f"FIXABLE_OFFSET: {counts['FIXABLE_OFFSET']}")
    print(f"FIXABLE_MISSING_SPAN: {counts['FIXABLE_MISSING_SPAN']}")
    print(f"NEEDS_REVIEW: {counts['NEEDS_REVIEW']}")
    if total_rows > 0:
        print(f"NEEDS_REVIEW %: {counts['NEEDS_REVIEW'] / total_rows * 100:.2f}%")
        
    print(f"\n--- Blast Radius of Old Logic ---")
    print(f"Additional rows that old logic silently mislabels as all-O (and would be caught by this audit): {old_logic_additional_needs_review}")

if __name__ == '__main__':
    main()
