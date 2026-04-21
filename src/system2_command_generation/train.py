"""Training entrypoint for System 2 command-generation models.

This module orchestrates data loading, model initialization, training,
evaluation, and checkpointing for CodeT5+ and Qwen2.5-Coder-1.5B experiments using the
command-dataset schema.
"""

import os
import sys
import json
import inspect
import importlib.util
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import DatasetDict

# Local imports
try:
    from .config import (
        TrainingConfig,
        ModelType,
        MODEL_CONFIGS,
    )
    from .data_preprocessing import (
        CommandDataProcessor,
        CommandGenerationDataset,
        MCPClient,
        parse_model_output,
        format_input,
        format_output,
    )
    from .models import (
        CommandGenerationModel,
        create_model,
        load_tokenizer,
        load_model,
    )
    from .metrics import (
        CommandMetrics,
        MetricResults,
        check_exit_criteria,
        compute_metrics_for_trainer,
    )
except ImportError as exc:
    # Allow `python src/system2_command_generation/train.py` in addition to
    # module execution (`python -m src.system2_command_generation.train`).
    if "attempted relative import with no known parent package" not in str(exc):
        raise

    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    from system2_command_generation.config import (  # type: ignore
        TrainingConfig,
        ModelType,
        MODEL_CONFIGS,
    )
    from system2_command_generation.data_preprocessing import (  # type: ignore
        CommandDataProcessor,
        CommandGenerationDataset,
        MCPClient,
        parse_model_output,
        format_input,
        format_output,
    )
    from system2_command_generation.models import (  # type: ignore
        CommandGenerationModel,
        create_model,
        load_tokenizer,
        load_model,
    )
    from system2_command_generation.metrics import (  # type: ignore
        CommandMetrics,
        MetricResults,
        check_exit_criteria,
        compute_metrics_for_trainer,
    )

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("command_generation")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, f"training_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def _cli_option_provided(option: str) -> bool:
    """Return True if a CLI option was explicitly provided."""
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in sys.argv[1:])


def _auto_tune_for_low_vram(config: TrainingConfig) -> List[str]:
    """Apply conservative Qwen settings on lower-memory GPUs."""
    if config.model_type != ModelType.QWEN2_5_CODER_1_5B:
        return []

    if not torch.cuda.is_available():
        return []

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)

    profile: Optional[Dict[str, int]] = None
    if total_vram_gb <= 10:
        profile = {
            "batch_size": 1,
            "eval_batch_size": 1,
            "max_input_length": 192,
            "max_output_length": 192,
            "gradient_accumulation_steps": 24,
        }
    elif total_vram_gb <= 14:
        profile = {
            "batch_size": 1,
            "eval_batch_size": 1,
            "max_input_length": 256,
            "max_output_length": 256,
            "gradient_accumulation_steps": 16,
        }
    elif total_vram_gb <= 20:
        profile = {
            "batch_size": 1,
            "eval_batch_size": 1,
            "max_input_length": 320,
            "max_output_length": 320,
            "gradient_accumulation_steps": 12,
        }
    elif total_vram_gb <= 24:
        profile = {
            "batch_size": 2,
            "eval_batch_size": 2,
            "max_input_length": 384,
            "max_output_length": 384,
            "gradient_accumulation_steps": 8,
        }

    if profile is None:
        return []

    changes = [f"Detected GPU memory: {total_vram_gb:.1f} GB"]

    def _cap(attr: str, target: int):
        current = getattr(config, attr)
        if current > target:
            setattr(config, attr, target)
            changes.append(f"{attr}: {current} -> {target}")

    _cap("batch_size", profile["batch_size"])
    _cap("eval_batch_size", profile["eval_batch_size"])
    _cap("max_input_length", profile["max_input_length"])
    _cap("max_output_length", profile["max_output_length"])

    if config.gradient_accumulation_steps < profile["gradient_accumulation_steps"]:
        old = config.gradient_accumulation_steps
        config.gradient_accumulation_steps = profile["gradient_accumulation_steps"]
        changes.append(
            "gradient_accumulation_steps: "
            f"{old} -> {config.gradient_accumulation_steps}"
        )

    if config.generation_num_beams > 1:
        old = config.generation_num_beams
        config.generation_num_beams = 1
        changes.append(f"generation_num_beams: {old} -> 1")

    if not config.gradient_checkpointing:
        config.gradient_checkpointing = True
        changes.append("gradient_checkpointing: False -> True")

    if len(changes) == 1:
        return []

    return changes


# =============================================================================
# Custom Callbacks
# =============================================================================

class MetricsLoggingCallback(TrainerCallback):
    """Callback to log custom metrics during training."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Log metrics after evaluation."""
        self.logger.info("=" * 60)
        self.logger.info(f"Evaluation at step {state.global_step}")
        
        for key, value in sorted(metrics.items()):
            if key.startswith("eval_"):
                self.logger.info(f"  {key}: {value:.4f}")
        
        self.logger.info("=" * 60)


class ExitCriteriaCallback(TrainerCallback):
    """Callback to check exit criteria and stop training if met."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.criteria_met = False
    
    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Check exit criteria after evaluation."""
        # Extract relevant metrics
        results = MetricResults(
            exact_match=metrics.get("eval_exact_match", 0),
            normalized_match=metrics.get("eval_normalized_match", 0),
            intent_preservation=metrics.get("eval_intent_preservation", 0),
            entity_preservation=metrics.get("eval_entity_preservation", 0),
            syntax_validity=metrics.get("eval_syntax_validity", 0),
            os_shell_compatibility=metrics.get("eval_os_shell_compatibility", 0),
        )
        
        passed, failures = check_exit_criteria(results)
        
        if passed:
            self.logger.info("🎉 All exit criteria met! Training can be stopped.")
            self.criteria_met = True
            # Optionally stop training
            # control.should_training_stop = True
        else:
            self.logger.info(f"Exit criteria not yet met. Failures: {failures}")


class MCPHealthCheckCallback(TrainerCallback):
    """Callback that explains MCP's role and verifies server connectivity.

    Fires once at the start of training and again at each evaluation so you
    can see that documentation context is actively being used.
    """

    def __init__(self, mcp_client: MCPClient, logger: logging.Logger):
        self.mcp_client = mcp_client
        self.logger = logger
        self._server_online: Optional[bool] = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        online = self.mcp_client.is_available()
        self._server_online = online

        self.logger.info("=" * 60)
        self.logger.info("[MCP] Documentation enrichment — status report")
        self.logger.info("=" * 60)
        if online:
            self.logger.info("  Server : ONLINE  (%s)", self.mcp_client.base_url)
            self.logger.info("  Status : Training inputs contain live documentation.")
        else:
            self.logger.warning("  Server : OFFLINE (%s)", self.mcp_client.base_url)
            self.logger.warning("  Status : Stub fallbacks were used during pre-fetch.")
            self.logger.warning("  Tip    : python src/mcp/server.py --port 11435")

        self.logger.info("")
        self.logger.info("  How MCP enrichment works during training:")
        self.logger.info("    1. Each sample's intent_type + entities.runtime →")
        self.logger.info("       (tool, operation) pair  e.g. python/install_package → pip/install")
        self.logger.info("    2. POST /fetch_docs  →  DocChunk with real command syntax,")
        self.logger.info("       key flags, a working example, and OS-specific notes")
        self.logger.info("    3. DocChunk injected into <docs>…</docs> block in the input")
        self.logger.info("    4. Model learns to generate commands *grounded in actual docs*")
        self.logger.info("       rather than just from training-set patterns")
        self.logger.info("=" * 60)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self._server_online:
            self.logger.info(
                "[MCP] Eval step %d — inputs were doc-enriched via %s",
                state.global_step, self.mcp_client.base_url,
            )


# =============================================================================
# Trainer Class
# =============================================================================

class CommandGenerationTrainer:
    """
    Main trainer class for command generation models.
    
    Handles the full training pipeline including:
    - Data loading and preprocessing
    - Model initialization
    - Training loop
    - Evaluation
    - Checkpointing
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            logger: Logger instance (created if not provided)
        """
        self.config = config
        self.logger = logger or setup_logging(config.output_dir)
        
        # Set random seeds for reproducibility
        self._set_seeds(config.seed)
        
        # Initialize components (lazy loading)
        self._model = None
        self._tokenizer = None
        self._data_processor = None
        self._datasets = None
        self._trainer = None
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger.info(f"Random seed set to {seed}")

    def _should_use_qlora(self) -> bool:
        """Return True when QLoRA should be applied for this run."""
        return (
            self.config.use_qlora
            and self.config.model_type == ModelType.QWEN2_5_CODER_1_5B
            and torch.cuda.is_available()
        )

    def _resolve_qlora_compute_dtype(self) -> torch.dtype:
        """Resolve configured QLoRA compute dtype to a torch dtype."""
        dtype_name = str(self.config.qlora_compute_dtype).lower()
        mapping: Dict[str, torch.dtype] = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = mapping.get(dtype_name)
        if dtype is None:
            self.logger.warning(
                "Unknown qlora_compute_dtype '%s'; using float16.",
                self.config.qlora_compute_dtype,
            )
            return torch.float16

        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.logger.warning(
                "bf16 not supported on this GPU; falling back to float16 for QLoRA."
            )
            return torch.float16

        return dtype

    def _create_qlora_quantization_config(self) -> BitsAndBytesConfig:
        """Build bitsandbytes quantization settings for QLoRA."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.qlora_quant_type,
            bnb_4bit_use_double_quant=self.config.qlora_double_quant,
            bnb_4bit_compute_dtype=self._resolve_qlora_compute_dtype(),
        )

    def _apply_qlora_adapters(self, model):
        """Attach LoRA adapters to a k-bit base model."""
        if (
            LoraConfig is None
            or TaskType is None
            or get_peft_model is None
            or prepare_model_for_kbit_training is None
        ):
            raise RuntimeError(
                "QLoRA requested but peft is not installed. "
                "Install dependencies from src/system2_command_generation/requirements.txt"
            )

        if not self.config.qlora_target_modules:
            raise ValueError("qlora_target_modules cannot be empty when QLoRA is enabled")

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        lora_config = LoraConfig(
            r=self.config.qlora_r,
            lora_alpha=self.config.qlora_alpha,
            lora_dropout=self.config.qlora_dropout,
            target_modules=self.config.qlora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params if total_params else 0.0
        self.logger.info(
            "QLoRA adapters attached: trainable params=%s / total params=%s (%.4f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            trainable_pct,
        )

        return model
    
    @property
    def model(self) -> CommandGenerationModel:
        """Lazy-load the model."""
        if self._model is None:
            self.logger.info(f"Loading model: {self.config.model_type.value}")

            if self.config.use_qlora and self.config.model_type != ModelType.QWEN2_5_CODER_1_5B:
                self.logger.warning(
                    "QLoRA requested but model type is %s. QLoRA is only applied for qwen2_5_coder_1_5b.",
                    self.config.model_type.value,
                )
            if self.config.use_qlora and not torch.cuda.is_available():
                self.logger.warning(
                    "QLoRA requested but CUDA is unavailable; falling back to non-QLoRA model loading."
                )

            wrapper = create_model(
                self.config.model_type,
                self.config,
                load_pretrained=False,
            )
            model_config = self.config.get_model_config()
            tokenizer = load_tokenizer(
                model_config,
                cache_dir=self.config.cache_dir,
            )

            if self._should_use_qlora():
                if importlib.util.find_spec("bitsandbytes") is None:
                    raise RuntimeError(
                        "QLoRA requires bitsandbytes, but it is not installed in this environment."
                    )

                self.logger.info(
                    "Using QLoRA: 4-bit base model + LoRA adapters (r=%d, alpha=%d, dropout=%.3f)",
                    self.config.qlora_r,
                    self.config.qlora_alpha,
                    self.config.qlora_dropout,
                )

                model = load_model(
                    model_config,
                    cache_dir=self.config.cache_dir,
                    device_map="auto",
                    torch_dtype=self._resolve_qlora_compute_dtype(),
                    quantization_config=self._create_qlora_quantization_config(),
                )
                model = self._apply_qlora_adapters(model)
            else:
                model = load_model(
                    model_config,
                    cache_dir=self.config.cache_dir,
                )

            wrapper.tokenizer = tokenizer
            wrapper.model = model
            self._model = wrapper
        return self._model
    
    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self.model.tokenizer
    
    @property
    def data_processor(self) -> CommandDataProcessor:
        """Lazy-load the data processor (with optional MCP client from config)."""
        if self._data_processor is None:
            mcp_client = None
            if self.config.use_mcp:
                mcp_client = MCPClient(
                    url=self.config.mcp_url,
                    timeout=self.config.mcp_timeout,
                )
            self._data_processor = CommandDataProcessor(
                tokenizer=self.tokenizer,
                config=self.config,
                mcp_client=mcp_client,
            )
        return self._data_processor
    
    def load_data(
        self,
        dataset_source: str = "huggingface",
        dataset_name: str = "sumit-s-nair/command-dataset",
        local_data_dir: Optional[str] = None,
    ) -> Dict[str, CommandGenerationDataset]:
        """
        Load and preprocess training data.
        
        Args:
            dataset_source: "huggingface" or "local"
            dataset_name: HuggingFace dataset name
            local_data_dir: Path to local data directory
            
        Returns:
            Dictionary of datasets by split
        """
        self.logger.info("Loading training data...")
        
        if dataset_source == "huggingface":
            raw_data = self.data_processor.load_huggingface_dataset(dataset_name)
        elif dataset_source == "local":
            if local_data_dir is None:
                local_data_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "datasets", "command-dataset", "data"
                )
            raw_data = self.data_processor.load_local_dataset(local_data_dir)
        else:
            raise ValueError(f"Unknown dataset source: {dataset_source}")
        
        self._datasets = self.data_processor.create_datasets(raw_data)
        
        # Log dataset sizes
        for split, dataset in self._datasets.items():
            self.logger.info(f"  {split}: {len(dataset)} samples")
        
        return self._datasets
    
    def _create_compute_metrics(self):
        """Create the compute_metrics function for the Trainer."""
        tokenizer = self.tokenizer
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            # Decode predictions
            decoded_preds = tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,
            )
            
            # Replace -100 in labels with pad token id
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
            )
            
            # Parse and evaluate
            metrics_calc = CommandMetrics()
            
            for pred_text, ref_text in zip(decoded_preds, decoded_labels):
                pred_plan, _ = parse_model_output(pred_text)
                ref_plan, _ = parse_model_output(ref_text)
                
                if ref_plan:
                    # Use reference as input approximation
                    input_intent = {
                        "intent_type": ref_plan.get("intent_type"),
                        "entities": ref_plan.get("entities", {}),
                    }
                    metrics_calc.add_prediction(
                        pred_plan or {},
                        ref_plan,
                        input_intent,
                    )
            
            results = metrics_calc.compute()
            
            return {
                "exact_match": results.exact_match,
                "normalized_match": results.normalized_match,
                "intent_preservation": results.intent_preservation,
                "entity_preservation": results.entity_preservation,
                "syntax_validity": results.syntax_validity,
                "schema_validity": results.schema_validity,
                "os_shell_compatibility": results.os_shell_compatibility,
                "json_validity": results.json_validity,
            }
        
        return compute_metrics
    
    def _get_training_args(self) -> Seq2SeqTrainingArguments:
        """Create HuggingFace training arguments."""
        model_config = MODEL_CONFIGS[self.config.model_type]
        generation_max_length = (
            self.config.max_output_length
            if model_config.is_encoder_decoder
            else self.config.max_input_length + self.config.max_output_length
        )

        has_tensorboard = (
            importlib.util.find_spec("tensorboard") is not None
            or importlib.util.find_spec("tensorboardX") is not None
        )
        report_to = ["tensorboard"] if has_tensorboard else []

        if not has_tensorboard:
            self.logger.warning(
                "TensorBoard not installed; disabling Trainer metric reporting. "
                "Install tensorboard or tensorboardX to enable it."
            )

        qlora_enabled = self._should_use_qlora()
        optim_name = self.config.optim
        if qlora_enabled and self.config.optim == "adamw_torch":
            optim_name = "paged_adamw_8bit"

        if qlora_enabled:
            self.logger.info(
                "QLoRA enabled: disabling Trainer fp16/bf16 AMP scaler to avoid k-bit gradient unscale issues."
            )

        use_fp16 = self.config.fp16 and torch.cuda.is_available() and not qlora_enabled
        use_bf16 = (
            self.config.bf16
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and not qlora_enabled
        )

        training_kwargs: Dict[str, Any] = {
            "output_dir": self.config.output_dir,

            # Training settings
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,

            # Optimization
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "optim": optim_name,
            "gradient_checkpointing": self.config.gradient_checkpointing,

            # Precision
            "fp16": use_fp16,
            "bf16": use_bf16,

            # Evaluation
            "eval_strategy": self.config.eval_strategy,
            "eval_steps": self.config.eval_steps,

            # Checkpointing
            "save_strategy": self.config.save_strategy,
            "save_steps": self.config.save_steps,
            "save_total_limit": self.config.save_total_limit,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,

            # Logging
            "logging_steps": self.config.logging_steps,
            "report_to": report_to,

            # Generation settings for evaluation
            "predict_with_generate": True,
            "generation_max_length": generation_max_length,
            "generation_num_beams": self.config.generation_num_beams,

            # Misc
            "seed": self.config.seed,
            "dataloader_num_workers": self.config.num_workers,
            "remove_unused_columns": False,
        }

        if self.config.warmup_steps > 0:
            training_kwargs["warmup_steps"] = self.config.warmup_steps
        else:
            training_kwargs["warmup_ratio"] = self.config.warmup_ratio

        return Seq2SeqTrainingArguments(**training_kwargs)
    
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.config.model_type.value}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Ensure data is loaded
        if self._datasets is None:
            self.load_data()
        
        # Check for empty datasets
        if "train" not in self._datasets or len(self._datasets["train"]) == 0:
            self.logger.warning("⚠️ Training dataset is empty! Please prepare the dataset first.")
            self.logger.info("The training script is ready. Run again after the dataset is prepared.")
            return {"status": "waiting_for_data"}
        
        # Get training arguments
        training_args = self._get_training_args()
        
        # Datasets already return fixed-size tensors; simple stack avoids
        # architecture-specific truncation issues between seq2seq and causal LMs.
        data_collator = self.data_processor.get_data_collator()
        
        # Create callbacks
        callbacks = [
            MetricsLoggingCallback(self.logger),
            ExitCriteriaCallback(self.config, self.logger),
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold,
            ),
        ]

        # Add MCP callback when enrichment is enabled
        if self.config.use_mcp and self.data_processor.mcp_client is not None:
            callbacks.append(
                MCPHealthCheckCallback(self.data_processor.mcp_client, self.logger)
            )
        
        # Create trainer
        model_for_training = self.model.model
        if self.config.gradient_checkpointing:
            if hasattr(model_for_training, "gradient_checkpointing_enable"):
                model_for_training.gradient_checkpointing_enable()
            if hasattr(model_for_training, "config") and hasattr(model_for_training.config, "use_cache"):
                model_for_training.config.use_cache = False

        trainer_kwargs = {
            "model": model_for_training,
            "args": training_args,
            "train_dataset": self._datasets.get("train"),
            "eval_dataset": self._datasets.get("validation"),
            "data_collator": data_collator,
            "compute_metrics": self._create_compute_metrics(),
            "callbacks": callbacks,
        }

        # Transformers >=5 moved tokenizer -> processing_class.
        trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
        if "processing_class" in trainer_init_params:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_init_params:
            trainer_kwargs["tokenizer"] = self.tokenizer

        self._trainer = Seq2SeqTrainer(**trainer_kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        self.logger.info("Starting training loop...")
        train_result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.logger.info("Saving final model...")
        self.model.save(os.path.join(self.config.output_dir, "final_model"))
        
        # Save training results
        results = {
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "train_loss": train_result.metrics.get("train_loss"),
        }
        
        with open(os.path.join(self.config.output_dir, "train_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Training completed!")
        return results
    
    def evaluate(
        self,
        dataset_split: str = "test",
    ) -> MetricResults:
        """
        Evaluate the model on a dataset split.
        
        Args:
            dataset_split: Which split to evaluate on
            
        Returns:
            MetricResults containing all evaluation metrics
        """
        self.logger.info(f"Evaluating on {dataset_split} set...")
        
        if self._datasets is None:
            self.load_data()
        
        if dataset_split not in self._datasets:
            raise ValueError(f"Split '{dataset_split}' not found in datasets")
        
        dataset = self._datasets[dataset_split]
        
        if len(dataset) == 0:
            self.logger.warning(f"⚠️ {dataset_split} dataset is empty!")
            return MetricResults()
        
        # Set model to eval mode
        self.model.eval_mode()
        
        # Initialize metrics
        metrics = CommandMetrics()
        
        # Process in batches
        batch_size = self.config.eval_batch_size
        device = self.model.device
        
        self.logger.info(f"Processing {len(dataset)} samples...")
        
        for i in range(0, len(dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            
            for idx in batch_indices:
                # Get raw item
                raw_item = dataset.get_raw_item(idx)
                input_text = raw_item["input_text"]
                reference_plan = raw_item["command_plan"]
                input_intent = raw_item["canonical_intent"]
                
                # Generate prediction
                try:
                    output_text = self.model.generate_text(
                        input_text,
                        num_beams=4,
                        max_length=self.config.max_output_length,
                    )
                    
                    # Parse output
                    pred_plan, error = parse_model_output(output_text)
                    
                    if pred_plan:
                        metrics.add_prediction(pred_plan, reference_plan, input_intent)
                    else:
                        metrics.add_prediction({}, reference_plan, input_intent)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing sample {idx}: {e}")
                    metrics.add_prediction({}, reference_plan, input_intent)
            
            # Log progress
            if (i + batch_size) % (batch_size * 10) == 0:
                self.logger.info(f"  Processed {i + batch_size}/{len(dataset)} samples")
        
        # Compute final metrics
        results = metrics.compute()
        
        # Log summary
        self.logger.info(results.summary())
        
        # Check exit criteria
        passed, failures = check_exit_criteria(results)
        if passed:
            self.logger.info("✅ All exit criteria met!")
        else:
            self.logger.info(f"❌ Exit criteria not met: {failures}")
        
        # Save results
        results_path = os.path.join(
            self.config.output_dir,
            f"eval_results_{dataset_split}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        
        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train command generation models (CodeT5+ or Qwen2.5-Coder-1.5B)"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["codet5plus", "qwen2_5_coder_1_5b"],
        default="qwen2_5_coder_1_5b",
        help=(
            "Model to train (default: qwen2_5_coder_1_5b). "
            "Qwen2.5-Coder-1.5B is recommended for stronger code generation and "
            "better instruction-following on command planning tasks."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train CodeT5+ as baseline comparison"
    )
    
    # Data source
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["huggingface", "local"],
        default="huggingface",
        help="Data source (default: huggingface)"
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default=None,
        help="Local data directory (for --data-source local)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sumit-s-nair/command-dataset",
        help="HuggingFace dataset name"
    )
    
    # Training configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/system2_command_generation",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Maximum encoder/prompt length"
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=1024,
        help="Maximum target/completion length"
    )
    parser.add_argument(
        "--generation-num-beams",
        type=int,
        default=1,
        help="Beam count during evaluation generation"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps (overrides warmup_ratio when > 0)"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--disable-auto-memory-tuning",
        action="store_true",
        help="Disable automatic low-VRAM safety tuning for Qwen training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # QLoRA configuration
    qlora_group = parser.add_mutually_exclusive_group()
    qlora_group.add_argument(
        "--use-qlora",
        action="store_true",
        help="Enable QLoRA (4-bit quantized base model + LoRA adapters)",
    )
    qlora_group.add_argument(
        "--no-qlora",
        action="store_true",
        help="Disable QLoRA and train full model weights",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (r)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="Target modules to apply LoRA adapters to",
    )
    parser.add_argument(
        "--qlora-compute-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Compute dtype used by 4-bit QLoRA kernels",
    )
    parser.add_argument(
        "--qlora-quant-type",
        type=str,
        choices=["nf4", "fp4"],
        default="nf4",
        help="4-bit quantization type",
    )
    parser.add_argument(
        "--qlora-no-double-quant",
        action="store_true",
        help="Disable nested (double) quantization for QLoRA",
    )
    
    # MCP documentation enrichment
    parser.add_argument(
        "--use-mcp",
        action="store_true",
        help=(
            "Enrich training inputs with live documentation from the MCP server. "
            "Each sample's input will include the real command syntax, key flags, "
            "and an example fetched from the relevant package manager (pip, npm, apt, …). "
            "Start the server first: python src/mcp/server.py --port 11435"
        ),
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default="http://localhost:11435",
        help="MCP server base URL (default: http://localhost:11435)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from"
    )
    
    # Actions
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --model-path)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model for evaluation"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()

    # Load config first to support config-file-first behavior.
    config = TrainingConfig.load(args.config) if args.config else None
    
    # Determine model type
    if config is not None and not args.baseline and not _cli_option_provided("--model"):
        model_type = config.model_type
    else:
        if args.baseline:
            model_type = ModelType.CODET5_PLUS
        else:
            model_map = {
                "codet5plus": ModelType.CODET5_PLUS,
                "qwen2_5_coder_1_5b": ModelType.QWEN2_5_CODER_1_5B,
            }
            model_type = model_map.get(args.model, ModelType.QWEN2_5_CODER_1_5B)

    default_use_qlora = model_type == ModelType.QWEN2_5_CODER_1_5B
    if args.use_qlora:
        use_qlora = True
    elif args.no_qlora:
        use_qlora = False
    else:
        use_qlora = default_use_qlora

    # Create or load configuration
    if config is None:
        config = TrainingConfig(
            model_type=model_type,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            generation_num_beams=args.generation_num_beams,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            seed=args.seed,
            use_qlora=use_qlora,
            qlora_r=args.lora_r,
            qlora_alpha=args.lora_alpha,
            qlora_dropout=args.lora_dropout,
            qlora_target_modules=args.lora_target_modules,
            qlora_compute_dtype=args.qlora_compute_dtype,
            qlora_quant_type=args.qlora_quant_type,
            qlora_double_quant=not args.qlora_no_double_quant,
            use_mcp=args.use_mcp,
            mcp_url=args.mcp_url,
        )

    # Explicit CLI overrides for config-file runs.
    if args.config:
        if _cli_option_provided("--model") or args.baseline:
            config.model_type = model_type
        if _cli_option_provided("--output-dir"):
            config.output_dir = args.output_dir
        if _cli_option_provided("--epochs"):
            config.num_epochs = args.epochs
        if _cli_option_provided("--batch-size"):
            config.batch_size = args.batch_size
        if _cli_option_provided("--eval-batch-size"):
            config.eval_batch_size = args.eval_batch_size
        if _cli_option_provided("--grad-accum-steps"):
            config.gradient_accumulation_steps = args.grad_accum_steps
        if _cli_option_provided("--learning-rate"):
            config.learning_rate = args.learning_rate
        if _cli_option_provided("--seed"):
            config.seed = args.seed
        if _cli_option_provided("--max-input-length"):
            config.max_input_length = args.max_input_length
        if _cli_option_provided("--max-output-length"):
            config.max_output_length = args.max_output_length
        if _cli_option_provided("--generation-num-beams"):
            config.generation_num_beams = args.generation_num_beams
        if _cli_option_provided("--warmup-steps"):
            config.warmup_steps = args.warmup_steps
        if _cli_option_provided("--use-mcp"):
            config.use_mcp = args.use_mcp
        if _cli_option_provided("--mcp-url"):
            config.mcp_url = args.mcp_url
        if _cli_option_provided("--use-qlora"):
            config.use_qlora = True
        if _cli_option_provided("--no-qlora"):
            config.use_qlora = False
        if _cli_option_provided("--lora-r"):
            config.qlora_r = args.lora_r
        if _cli_option_provided("--lora-alpha"):
            config.qlora_alpha = args.lora_alpha
        if _cli_option_provided("--lora-dropout"):
            config.qlora_dropout = args.lora_dropout
        if _cli_option_provided("--lora-target-modules"):
            config.qlora_target_modules = args.lora_target_modules
        if _cli_option_provided("--qlora-compute-dtype"):
            config.qlora_compute_dtype = args.qlora_compute_dtype
        if _cli_option_provided("--qlora-quant-type"):
            config.qlora_quant_type = args.qlora_quant_type
        if _cli_option_provided("--qlora-no-double-quant"):
            config.qlora_double_quant = False
        if args.no_gradient_checkpointing:
            config.gradient_checkpointing = False

    # Keep config model aligned with selected CLI model in non-config runs.
    if not args.config:
        config.model_type = model_type

    if not args.disable_auto_memory_tuning:
        tuned = _auto_tune_for_low_vram(config)
        if tuned:
            print("\n[Auto Memory Tuning] Applied conservative settings:")
            for line in tuned:
                print(f"  - {line}")
            print("  - Use --disable-auto-memory-tuning to keep raw values.\n")
    
    # Update output directory with model name
    config.output_dir = os.path.join(
        config.output_dir,
        f"{config.model_type.value}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    
    # Save configuration
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "config.json"))
    
    # Initialize trainer
    trainer = CommandGenerationTrainer(config)
    
    # Load data
    trainer.load_data(
        dataset_source=args.data_source,
        dataset_name=args.dataset_name,
        local_data_dir=args.local_data_dir,
    )
    
    if args.eval_only:
        # Evaluation only mode
        if args.model_path:
            trainer._model = CommandGenerationModel.load(args.model_path)
        results = trainer.evaluate(dataset_split="test")
    else:
        # Training mode
        train_results = trainer.train(resume_from_checkpoint=args.resume)
        
        if train_results.get("status") != "waiting_for_data":
            # Evaluate on test set
            eval_results = trainer.evaluate(dataset_split="test")
    
    print("\n" + "=" * 60)
    print("🎉 Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
