# contAIner  
AI-Driven OS-Level Environment Manager

contAIner is a research-driven system for converting natural-language user intent into
safe, OS-aware shell command plans. The system is designed as a modular agent rather
than a monolithic language model.

The current focus (Step-1) is on intent understanding and command generation.
Environment execution and dependency installation are intentionally deferred.

---

## Motivation

Software environment setup is repetitive, error-prone, and highly platform-specific.
Existing tools rely on manual configuration and lack semantic understanding of user intent.

contAIner aims to:
- Interpret vague user requests safely
- Generate OS-aware shell command plans
- Reduce hallucinations using structured intent and validation
- Provide a foundation for automated environment management

---

## Design Philosophy

- Modular agent, not a single LLM
- Explicit intent representation
- OS awareness as first-class input
- Safety before execution
- Incremental learning from validated outputs

---

## Architecture (High-Level)

```mermaid
flowchart LR
    Input["User Intent"]
    Canon["Intent\nCanonicalization"]
    OS["OS Detection"]
    Planner["NL → Command\nPlanner"]
    Safety["Safety\nFilters"]
    Exec["Dry-Run /\nSandbox Execution"]
    
    Input --> Canon --> OS --> Planner --> Safety --> Exec
    
    style Input fill:#2d5f8d,stroke:#5dade2,stroke-width:2px,color:#fff
    style Canon fill:#1e5f74,stroke:#48c9b0,stroke-width:2px,color:#fff
    style OS fill:#5b2c6f,stroke:#a569bd,stroke-width:2px,color:#fff
    style Planner fill:#935116,stroke:#f39c12,stroke-width:2px,color:#fff
    style Safety fill:#78281f,stroke:#ec7063,stroke-width:2px,color:#fff
    style Exec fill:#1e8449,stroke:#52be80,stroke-width:2px,color:#fff
```  

## Quick Start

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/contAIner.git
cd contAIner
```

2. **Set up Python environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r src/utils/requirements.txt
```

3. **Configure environment variables:**
Create a `.env` file in the project root:
```bash
# .env
HF_TOKEN=your_huggingface_token_here
```

Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### Loading Datasets

#### Using the Dataset Loader Utility

```python
from src.utils.load_datasets import load_datasets

# Load all datasets (auto-loads .env)
intent_data, command_data = load_datasets()

# Access specific splits
train_intents = intent_data['train']
train_commands = command_data['train']

# Load only specific splits
intent_data, command_data = load_datasets(splits=['train', 'validation'])
```

#### Direct Hugging Face Usage

```python
from datasets import load_dataset

# Load individual datasets
intent_dataset = load_dataset("sumit-s-nair/intent-dataset")
command_dataset = load_dataset("sumit-s-nair/command-dataset")
```

### Environment Variable Loading

The project includes a centralized environment loader that automatically finds and loads `.env` files:

```python
from src.utils.env_loader import ensure_env_loaded

# Call this before accessing any environment variables
ensure_env_loaded()

# Now safely access env vars
import os
token = os.environ.get("HF_TOKEN")
```

### Contributing to Datasets

#### 1. Validate Your Data

Before adding data, validate against the schema:

```bash
# Install validation dependency
pip install jsonschema

# Validate intent data
python datasets/intent-dataset/scripts/validate_schema.py datasets/intent-dataset/data/train.jsonl

# Validate command data  
python datasets/command-dataset/scripts/validate_schema.py datasets/command-dataset/data/train.jsonl
```

#### 2. Add Data to Local Files

Follow the exact schemas defined in `datasets/*/schema.json`:

**Intent Dataset Format:**
```jsonl
{"instruction": "install numpy for python", "intent_type": "install_package", "entities": {"runtime": "python", "package": "numpy", "version": null}, "entity_spans": {"runtime": {"start": 20, "end": 26, "text": "python"}, "package": {"start": 8, "end": 13, "text": "numpy"}, "version": null}, "context": {"os_hint": null, "shell_type": null}, "paraphrase_group": null}
```

**Command Dataset Format:**
```jsonl
{"instruction": "install numpy package", "intent_type": "install_package", "entities": {"runtime": "python", "package": "numpy", "version": null}, "os": "linux", "shell": "bash", "command": "pip install numpy", "source": "manual"}
```

#### 3. Upload to Hugging Face

```bash
# Set your HF token (if not already in .env)
export HF_TOKEN=your_token_here

# Upload both datasets
python scripts/push_datasets_to_hf.py --all

# Or upload individually
python scripts/push_datasets_to_hf.py \
    --local-path datasets/intent-dataset \
    --repo-id sumit-s-nair/intent-dataset
```

---

## Step-1 Scope (Current)
Natural language → intent canonicalization  
Linux and Windows command generation  
Dataset normalization and augmentation  
LLM-based command planning  
Safety validation and CLI testing  

---

## Datasets Used

- NL2Bash  
  Natural language to Bash command corpus

- NL2SH-ALFA  
  Instructions-to-shell dataset  
  https://huggingface.co/datasets/westenfelder/NL2SH-ALFA

- Repository execution traces from prior research

---

## Models

The contAIner system uses a single primary model for command generation.

- **Primary Model:** CodeT5+ (770M)  
  Used for all NL → OS-aware shell command generation due to its strong
  performance on code-oriented tasks and structured output control.

- **Baseline Model:** FLAN-T5 (base)  
  Used only for benchmarking and evaluation.

- **Experimental Model:** Qwen-Coder (QLoRA)  
  Used for exploratory comparison and upper-bound performance analysis.

All models are explicitly conditioned on operating system and shell type
to prevent ambiguity and unsafe command generation.


---

## Evaluation Metrics

- Exact command match
- Normalized command match
- Execution validity (dry-run)

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Architecture Diagrams](docs/ARCHITECTURE_DIAGRAM.md) - Visual system architecture with Mermaid diagrams
- [System 1: Intent Understanding](docs/system-1-intent-understanding/README.md) - Intent classification, entity extraction, and decomposition
- [System 2: Command Generation](docs/system-2-command-generation/README.md) - OS-aware command generation and validation
- [Integration Guide](docs/integration/README.md) - System orchestration and deployment architectures
- [Dataset Documentation](docs/datasets/README.md) - Training data sources and schemas


## Repository Structure

```
contAIner/
├── src/
│   └── utils/
│       ├── load_datasets.py      # HF dataset loader utility
│       ├── env_loader.py         # Environment variable loader
│       ├── requirements.txt      # Python dependencies
│       └── README.md             # Usage documentation
├── datasets/
│   ├── intent-dataset/           # Intent understanding dataset
│   │   ├── README.md
│   │   ├── schema.json
│   │   ├── data/
│   │   │   ├── train.jsonl
│   │   │   ├── validation.jsonl
│   │   │   └── test.jsonl
│   │   └── scripts/
│   │       └── validate_schema.py
│   └── command-dataset/          # Command generation dataset
│       ├── README.md
│       ├── schema.json
│       ├── data/
│       │   ├── train.jsonl
│       │   ├── validation.jsonl
│       │   └── test.jsonl
│       └── scripts/
│           └── validate_schema.py
├── scripts/
│   └── push_datasets_to_hf.py    # Upload datasets to HF Hub
├── docs/
│   ├── ARCHITECTURE_DIAGRAM.md
│   ├── system-1-intent-understanding/
│   ├── system-2-command-generation/
│   ├── integration/
│   └── datasets/
├── .env                          # Environment variables (create this)
├── LICENSE
└── README.md
```

---
## Dataset Schemas

### Intent Dataset Schema

Captures user instructions and their semantic decomposition:

- `instruction` (string): Raw user input
- `intent_type` (string): Classified intent (e.g., "install_package")
- `entities` (object): Extracted entities (runtime, package, version)
- `entity_spans` (object): Character positions of entities in text
- `context` (object): Optional OS/shell hints
- `paraphrase_group` (string|null): Groups similar instructions

### Command Dataset Schema

Maps intents to executable shell commands:

- `instruction` (string): Canonicalized instruction
- `intent_type` (string): Intent classification
- `entities` (object): Extracted entities (runtime, package, version)
- `os` (string): Target operating system
- `shell` (string): Target shell environment
- `command` (string): Executable command
- `source` (string): Data origin (e.g., "manual", "NL2SH-ALFA")


## License

Apache License 2.0
