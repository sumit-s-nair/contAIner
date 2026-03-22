"""
System 1: Intent Understanding Module

This module handles Stage 1A (Intent Classification) of the contAIner pipeline.
It fine-tunes DistilBERT to classify user instructions into intent types.

Training Input (from intent-dataset):
    {instruction, intent_type, entities, entity_spans, context, paraphrase_group}

Training Output (model learns to predict):
    intent_type (e.g., install_package, update_package, install_runtime, update_runtime)

At Inference Time:
    Raw user text → System 1 → CanonicalIntent → System 2
"""

__all__ = []
