"""
Training Script for System 2: Command Generation

Main training pipeline for CodeT5+ and FLAN-T5 models that convert
structured intent data to CommandPlan objects.

Training Data (command-dataset schema):
    {instruction, intent_type, entities, os, shell, command, source}

Output (model learns):
    CommandPlan JSON

Usage:
    python -m src.system2_command_generation.train --model codet5plus
    python -m src.system2_command_generation.train --model flan_t5 --baseline

    # With custom config
    python -m src.system2_command_generation.train --config config.json

    # Resume training
    python -m src.system2_command_generation.train --resume checkpoint-1000
"""

import os
import sys
import json
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
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import DatasetDict

# Local imports
from .config import (
    TrainingConfig,
    ModelType,
    MODEL_CONFIGS,
)
from .data_preprocessing import (
    CommandDataProcessor,
    CommandGenerationDataset,
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
    
    @property
    def model(self) -> CommandGenerationModel:
        """Lazy-load the model."""
        if self._model is None:
            self.logger.info(f"Loading model: {self.config.model_type.value}")
            self._model = create_model(
                self.config.model_type,
                self.config,
                load_pretrained=True,
            )
        return self._model
    
    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self.model.tokenizer
    
    @property
    def data_processor(self) -> CommandDataProcessor:
        """Lazy-load the data processor."""
        if self._data_processor is None:
            self._data_processor = CommandDataProcessor(
                tokenizer=self.tokenizer,
                config=self.config,
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
        return Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            
            # Training settings
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,
            
            # Precision
            fp16=self.config.fp16 and torch.cuda.is_available(),
            bf16=self.config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            
            # Evaluation
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            
            # Checkpointing
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Logging
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            report_to=["tensorboard"],
            
            # Generation settings for evaluation
            predict_with_generate=True,
            generation_max_length=self.config.max_output_length,
            generation_num_beams=4,
            
            # Misc
            seed=self.config.seed,
            dataloader_num_workers=self.config.num_workers,
            remove_unused_columns=False,
        )
    
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
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model.model,
            padding=True,
            max_length=self.config.max_output_length,
        )
        
        # Create callbacks
        callbacks = [
            MetricsLoggingCallback(self.logger),
            ExitCriteriaCallback(self.config, self.logger),
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold,
            ),
        ]
        
        # Create trainer
        self._trainer = Seq2SeqTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self._datasets.get("train"),
            eval_dataset=self._datasets.get("validation"),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._create_compute_metrics(),
            callbacks=callbacks,
        )
        
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
        description="Train command generation models (CodeT5+ or FLAN-T5)"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["codet5plus", "flan_t5"],
        default="codet5plus",
        help="Model to train (default: codet5plus)"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train FLAN-T5 as baseline comparison"
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
    
    # Determine model type
    if args.baseline:
        model_type = ModelType.FLAN_T5
    else:
        model_type = ModelType.CODET5_PLUS if args.model == "codet5plus" else ModelType.FLAN_T5
    
    # Create or load configuration
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig(
            model_type=model_type,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    
    # Update output directory with model name
    config.output_dir = os.path.join(
        args.output_dir,
        f"{model_type.value}_{datetime.now():%Y%m%d_%H%M%S}"
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
