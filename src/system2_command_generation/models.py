"""Model loading and inference wrappers for System 2.

Supports CodeT5+ and Qwen2.5-Coder-1.5B with a shared interface for tokenizer/model
loading, text generation, and saving/loading trained checkpoints.
"""

import os
import inspect
from typing import Dict, Any, Optional, List, Union
from enum import Enum

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import ModelConfig, ModelType, MODEL_CONFIGS, TrainingConfig


# =============================================================================
# Model Type Enum (re-exported from config)
# =============================================================================

# Re-export for convenience
ModelType = ModelType


# =============================================================================
# Model Loading Utilities
# =============================================================================

def load_tokenizer(
    model_config: ModelConfig,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizer:
    """
    Load the tokenizer for a model.
    
    Args:
        model_config: Configuration for the model
        cache_dir: Directory to cache the tokenizer
        
    Returns:
        Loaded tokenizer
    """
    print(f"📥 Loading tokenizer: {model_config.tokenizer_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    return tokenizer


def load_model(
    model_config: ModelConfig,
    cache_dir: Optional[str] = None,
    device_map: Optional[str] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    quantization_config: Optional[Any] = None,
) -> PreTrainedModel:
    """
    Load a pre-trained model.
    
    Args:
        model_config: Configuration for the model
        cache_dir: Directory to cache the model
        device_map: Device mapping strategy
        torch_dtype: Data type for model weights
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        quantization_config: Optional transformers quantization config
        
    Returns:
        Loaded model
    """
    print(f"📥 Loading model: {model_config.model_id}")
    
    # Determine dtype
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    # Build kwargs
    kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }
    
    # Only add device_map and dtype for CUDA
    if torch.cuda.is_available():
        kwargs["device_map"] = device_map
        model_cls = AutoModelForSeq2SeqLM if model_config.is_encoder_decoder else AutoModelForCausalLM
        from_pretrained_params = inspect.signature(model_cls.from_pretrained).parameters
        dtype_arg = "dtype" if "dtype" in from_pretrained_params else "torch_dtype"
        kwargs[dtype_arg] = torch_dtype

        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            kwargs["load_in_4bit"] = True

    model_cls = AutoModelForSeq2SeqLM if model_config.is_encoder_decoder else AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_config.model_id,
        **kwargs,
    )
    
    # Move to CPU if no CUDA
    if not torch.cuda.is_available():
        model = model.to("cpu")
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Model loaded")
    print(f"    Total parameters: {num_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    return model


# =============================================================================
# Command Generation Model Wrapper
# =============================================================================

class CommandGenerationModel:
    """
    Wrapper class for command generation models.
    
    Provides a unified interface for loading, training, and inference
    with different model architectures (CodeT5+, Qwen2.5-Coder-1.5B).
    """
    
    def __init__(
        self,
        model_type: ModelType = ModelType.CODET5_PLUS, # type: ignore
        config: Optional[TrainingConfig] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Initialize the command generation model.
        
        Args:
            model_type: Type of model to use
            config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.model_type = model_type
        self.config = config or TrainingConfig(model_type=model_type)
        self.model_config = MODEL_CONFIGS[model_type]
        
        self._model = model
        self._tokenizer = tokenizer
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def model(self) -> PreTrainedModel:
        """Lazy-load the model."""
        if self._model is None:
            self._model = load_model(
                self.model_config,
                cache_dir=self.config.cache_dir,
            )
        return self._model
    
    @model.setter
    def model(self, value: PreTrainedModel):
        self._model = value
    
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(
                self.model_config,
                cache_dir=self.config.cache_dir,
            )
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizer):
        self._tokenizer = value
    
    def to(self, device: Union[str, torch.device]) -> "CommandGenerationModel":
        """Move model to device."""
        self.device = torch.device(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self
    
    def train_mode(self) -> "CommandGenerationModel":
        """Set model to training mode."""
        self.model.train()
        return self
    
    def eval_mode(self) -> "CommandGenerationModel":
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = False,
        early_stopping: bool = True,
        repetition_penalty: float = 1.3,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate command plans from input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum output length
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling
            early_stopping: Whether to stop early
            repetition_penalty: Penalty for repeated tokens (>1.0 discourages loops).
                Default 1.3 prevents the common --flag × N repetition loops seen
                with undertrained or greedy-decoded causal LMs.
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        max_length = max_length or self.model_config.max_output_length
        
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
                **kwargs,
            }

            # Sampling controls are only relevant when sampling is enabled.
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p

            # early_stopping is intended for beam search in seq2seq models.
            if self.model_config.is_encoder_decoder:
                generation_kwargs["early_stopping"] = early_stopping

            if self.model_config.is_encoder_decoder:
                generation_kwargs["max_length"] = max_length
                outputs = self.model.generate(**generation_kwargs)
            else:
                generation_kwargs["max_new_tokens"] = max_length
                outputs = self.model.generate(**generation_kwargs)
                # Decoder-only models return prompt + completion; keep completion only.
                prompt_len = input_ids.shape[1]
                outputs = outputs[:, prompt_len:]
        
        return outputs
    
    def generate_text(
        self,
        input_text: str,
        **generation_kwargs,
    ) -> str:
        """
        Generate command plan text from input text.
        
        Args:
            input_text: Input text (formatted CanonicalIntent)
            **generation_kwargs: Arguments passed to generate()
            
        Returns:
            Generated text
        """
        # Use max_input_length from the training config when available so that
        # inference uses the same sequence length the model was trained on.
        # Falling back to the model architecture default (1024 for Qwen) would
        # force padding far beyond the training distribution (192 tokens).
        max_tok = (
            getattr(self.config, "max_input_length", None)
            or self.model_config.max_input_length
        )

        if self.model_config.is_encoder_decoder:
            # Seq2seq: padding is required; model attends to full encoder output.
            encoding = self.tokenizer(
                input_text,
                max_length=max_tok,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Decoder-only (Qwen, GPT-*): do NOT pad at single-sample inference.
            # Padding inserts EOS tokens (pad==eos for Qwen) which confuse the
            # model into early-stopping or repetition loops.  No padding needed
            # for batch_size=1 because there is nothing to align with.
            encoding = self.tokenizer(
                input_text,
                max_length=max_tok,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Generate
        output_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        
        # Decode
        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        
        return output_text
    
    def save(self, output_dir: str):
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving model to: {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        self.config.save(os.path.join(output_dir, "training_config.json"))
        print("  ✓ Model saved")
    
    @classmethod
    def load(
        cls,
        model_dir: str,
        model_type: Optional[ModelType] = None, # pyright: ignore[reportInvalidTypeForm]
    ) -> "CommandGenerationModel":
        """
        Load a saved model.
        
        Args:
            model_dir: Directory containing the saved model
            model_type: Type of model (auto-detected if not provided)
            
        Returns:
            Loaded CommandGenerationModel
        """
        print(f"📥 Loading model from: {model_dir}")
        
        # Load config if available
        config_path = os.path.join(model_dir, "training_config.json")
        if os.path.exists(config_path):
            config = TrainingConfig.load(config_path)
            model_type = config.model_type
        else:
            config = TrainingConfig(model_type=model_type or ModelType.QWEN2_5_CODER_1_5B)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_config = config.get_model_config()
        model_cls = AutoModelForSeq2SeqLM if model_config.is_encoder_decoder else AutoModelForCausalLM
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "This checkpoint contains LoRA adapters. Install peft to load it."
                ) from exc

            peft_config = PeftConfig.from_pretrained(model_dir)
            base_model_id = peft_config.base_model_name_or_path or model_config.model_id
            base_model = model_cls.from_pretrained(
                base_model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                base_model,
                model_dir,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        else:
            model = model_cls.from_pretrained(
                model_dir,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        
        return cls(
            model_type=model_type or ModelType.QWEN2_5_CODER_1_5B,
            config=config,
            model=model,
            tokenizer=tokenizer,
        )


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    model_type: ModelType, # type: ignore
    config: Optional[TrainingConfig] = None,
    load_pretrained: bool = True,
) -> CommandGenerationModel:
    """
    Factory function to create a command generation model.
    
    Args:
        model_type: Type of model to create
        config: Training configuration
        load_pretrained: Whether to load pre-trained weights
        
    Returns:
        CommandGenerationModel instance
    """
    if config is None:
        config = TrainingConfig(model_type=model_type)
    
    wrapper = CommandGenerationModel(
        model_type=model_type,
        config=config,
    )
    
    if load_pretrained:
        # Access properties to trigger loading
        _ = wrapper.tokenizer
        _ = wrapper.model
    
    return wrapper


def create_codet5_plus(config: Optional[TrainingConfig] = None) -> CommandGenerationModel:
    """Create a CodeT5+ model for command generation."""
    return create_model(ModelType.CODET5_PLUS, config)


def create_qwen2_5_coder_1_5b(config: Optional[TrainingConfig] = None) -> CommandGenerationModel:
    """Create a Qwen2.5-Coder-1.5B model for command generation."""
    return create_model(ModelType.QWEN2_5_CODER_1_5B, config)


# =============================================================================
# Generation Utilities
# =============================================================================

def generate_command_plan(
    model: CommandGenerationModel,
    canonical_intent: Dict[str, Any],
    num_beams: int = 4,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Generate a CommandPlan from a CanonicalIntent (System 1 output).
    
    Args:
        model: The command generation model
        canonical_intent: Input CanonicalIntent from System 1
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        
    Returns:
        Generated CommandPlan dictionary
    """
    from .data_preprocessing import parse_model_output, format_input
    
    # Build row from CanonicalIntent for format_input
    row = {
        "intent_type": canonical_intent["intent_type"],
        "entities": canonical_intent["entities"],
        "os": canonical_intent.get("os_hint") or "linux",
        "shell": canonical_intent["shell_type"],
    }
    
    # Format input
    input_text = format_input(row, compact=True)
    
    # Generate
    output_text = model.generate_text(
        input_text,
        num_beams=num_beams,
        temperature=temperature,
    )
    
    # Parse output
    command_plan, error = parse_model_output(output_text)
    
    if error:
        raise ValueError(f"Failed to generate valid CommandPlan: {error}")
    
    return command_plan


def batch_generate(
    model: CommandGenerationModel,
    inputs: List[Dict[str, Any]],
    batch_size: int = 8,
    **generation_kwargs,
) -> List[Dict[str, Any]]:
    """
    Generate CommandPlans for a batch of CanonicalIntents.
    
    Args:
        model: The command generation model
        inputs: List of CanonicalIntent dictionaries (from System 1)
        batch_size: Batch size for generation
        **generation_kwargs: Arguments passed to generate()
        
    Returns:
        List of generated CommandPlan dictionaries
    """
    from .data_preprocessing import parse_model_output, format_input
    
    results = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        
        # Convert CanonicalIntents to row format and format inputs
        input_texts = []
        for intent in batch:
            row = {
                "intent_type": intent["intent_type"],
                "entities": intent["entities"],
                "os": intent.get("os_hint") or "linux",
                "shell": intent["shell_type"],
            }
            input_texts.append(format_input(row, compact=True))
        
        # Tokenize
        encodings = model.tokenizer(
            input_texts,
            max_length=model.model_config.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)
        
        # Generate
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        
        # Decode and parse
        for j, output in enumerate(output_ids):
            output_text = model.tokenizer.decode(output, skip_special_tokens=True)
            command_plan, error = parse_model_output(output_text)
            
            if command_plan:
                results.append(command_plan)
            else:
                # Return error info
                results.append({
                    "error": error,
                    "raw_output": output_text,
                    "input": batch[j],
                })
    
    return results
