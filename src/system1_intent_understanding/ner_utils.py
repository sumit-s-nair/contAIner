from typing import Dict, List, Tuple

def extract_entities_from_offsets(
    text: str,
    ner_ids: List[int],
    offsets: List[Tuple[int, int]],
    id_to_ner: Dict[int, str]
) -> Dict[str, str]:
    """
    Extracts contiguous entity spans from character offsets.
    Slices the original string to avoid subword spacing artifacts.
    """
    entities = {}
    current_entity_type = None
    current_start = -1
    current_end = -1

    for ner_id, (start, end) in zip(ner_ids, offsets):
        if start == 0 and end == 0:
            continue
            
        tag = id_to_ner.get(int(ner_id), "O")
        
        if tag.startswith("B-"):
            # Save previous entity
            if current_entity_type is not None:
                val = text[current_start:current_end].strip()
                if val:
                    entities.setdefault(current_entity_type, []).append(val)
                    
            current_entity_type = tag[2:]
            current_start = start
            current_end = end
            
        elif tag.startswith("I-") and current_entity_type == tag[2:]:
            # Extend current entity span
            current_end = max(current_end, end)
            
        else:
            # End of an entity due to O or mismatched I- tag
            if current_entity_type is not None:
                val = text[current_start:current_end].strip()
                if val:
                    entities.setdefault(current_entity_type, []).append(val)
                current_entity_type = None

    # Save the last active entity
    if current_entity_type is not None:
        val = text[current_start:current_end].strip()
        if val:
            entities.setdefault(current_entity_type, []).append(val)

    # Join multiple occurrences with a space (maintaining original dict return structure)
    return {k: " ".join(v) for k, v in entities.items() if v}
