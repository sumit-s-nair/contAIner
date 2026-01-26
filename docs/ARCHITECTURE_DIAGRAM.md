# contAIner Architecture Diagram

## High-Level System Overview

```mermaid
flowchart TD
    Start(["User Input: set up python 3.10 dev environment"]) --> System1
    
    System1["SYSTEM 1: INTENT UNDERSTANDING
Purpose: Validate and structure user intent
Output: CanonicalIntent JSON"]
    System1 --> Stage1A
    
    Stage1A["Stage 1A: Intent Classification
Model: DistilBERT 110M params
Output: intent_type + confidence"] --> Stage1B
    
    Stage1B["Stage 1B: Entity Extraction
Model: BERT-NER
Output: runtime, package, version"] --> Stage1C
    
    Stage1C["Stage 1C: OS and Shell Detection
Method: System APIs no ML
Output: os_hint + shell_type"] --> Stage1D
    
    Stage1D["Stage 1D: Hierarchical Decomposition
Method: Rule-based patterns
Output: Array CanonicalIntent"] --> Stage1E
    
    Stage1E["Stage 1E: Clarification Check
Model: T5-small
Output: needs_clarification flag"] --> Complete{Complete?}
    
    Complete -->|"No missing info"| Ready["CanonicalIntent ready for System 2"]
    Complete -->|"Missing info"| AskUser["Ask user for info"]
    
    Ready --> System2
    
    System2["SYSTEM 2: COMMAND GENERATION
Purpose: Generate OS-specific executable commands
Output: CommandPlan with shell commands"]
    System2 --> Generator
    
    Generator["Command Generator
Model: CodeT5+ 770M params
Trained on: NL2Bash, NL2SH-ALFA, Windows augmentations"] --> Plan
    
    Plan["CommandPlan
os: windows
shell: powershell
steps: winget install ..."] --> Final
    
    Final(["FINAL OUTPUT: Array CommandPlan
Ready for execution"])
    
    style System1 fill:#e1f5ff
    style System2 fill:#fff4e1
    style Final fill:#e8f5e9
```

---

## Training Pipeline

```mermaid
flowchart TD
    subgraph Datasets["Data Preparation"]
        IntentData["Intent Parser Dataset\n7,000 samples"]
        CommandData["Command Generation Dataset\n5,000+ samples"]
    end
    
    IntentData --> Stage1ATrain["Stage 1A Training"]
    IntentData --> Stage1BTrain["Stage 1B Training"]
    
    Stage1ATrain --> Model1A["DistilBERT\nIntent Classifier"]
    Stage1BTrain --> Model1B["BERT-NER\nEntity Extractor"]
    
    CommandData --> Normalize["Dataset\nNormalization"]
    CommandData --> Augment["Dataset Augmentation\nWindows"]
    
    Normalize --> CodeT5Train["CodeT5+ Training"]
    Augment --> CodeT5Train
    
    CodeT5Train --> Model2["CodeT5+\nCommand Generator"]
    
    style IntentData fill:#e1f5ff
    style CommandData fill:#fff4e1
    style Model1A fill:#c8e6c9
    style Model1B fill:#c8e6c9
    style Model2 fill:#c8e6c9
```

---

## Inference Pipeline

```mermaid
flowchart TD
    Input["User Input: install python 3.10"]
    
    Input --> Stage1A["Stage 1A: Intent Classifier"]
    Stage1A --> |"intent_type: install_runtime\nconfidence: 0.92"| Stage1B
    
    Stage1B["Stage 1B: Entity Extractor"]
    Stage1B --> |"entities:\nruntime=python\nversion=3.10"| Stage1C
    
    Stage1C["Stage 1C: OS/Shell Detector"]
    Stage1C --> |"os_hint: windows\nshell_type: powershell"| Stage1D
    
    Stage1D["Stage 1D: Decomposition"]
    Stage1D --> |"Single atomic intent"| Stage1E
    
    Stage1E["Stage 1E: Clarification Check"]
    Stage1E --> |"needs_clarification: false"| CanonicalIntent
    
    CanonicalIntent["CanonicalIntent\nintent_type: install_runtime\nentities: runtime=python, version=3.10\nos_hint: windows\nshell_type: powershell\nconfidence: 0.92"]
    
    CanonicalIntent --> Generator["Command Generator\nCodeT5+"]
    
    Generator --> CommandPlan["CommandPlan\nos: windows\nshell: powershell\nsteps: winget install Python.Python.3.10\nrequires_elevation: false"]
    
    style Input fill:#e8f5e9
    style CanonicalIntent fill:#fff3e0
    style CommandPlan fill:#f3e5f5
```

---

## Data Flow with Hierarchical Intent

```mermaid
flowchart TD
    Input(["User Input: set up python dev environment"]) --> Stage1A
    
    Stage1A["Stage 1A: Classify"] -->|"setup_environment hierarchical"| Stage1B
    
    Stage1B["Stage 1B: Extract Entities"] -->|"runtime=python, version=null"| Stage1E
    
    Stage1E["Stage 1E: Check Clarification"] -->|"needs_clarification: true"| UserResponse
    
    UserResponse(["USER RESPONSE: 3.10"]) --> Stage1D
    
    Stage1D["Stage 1D: Decompose"] --> Intent1
    Stage1D --> Intent2
    Stage1D --> Intent3
    
    Intent1["CanonicalIntent 1: install_runtime python 3.10"] --> Gen1["Command Generator"]
    Intent2["CanonicalIntent 2: install_package pip"] --> Gen2["Command Generator"]
    Intent3["CanonicalIntent 3: create_virtual_env python"] --> Gen3["Command Generator"]
    
    Gen1 --> Cmd1["CommandPlan 1: winget install Python.Python.3.10"]
    Gen2 --> Cmd2["CommandPlan 2: python -m pip install --upgrade pip"]
    Gen3 --> Cmd3["CommandPlan 3: python -m venv venv"]
    
    Cmd1 --> Final
    Cmd2 --> Final
    Cmd3 --> Final
    
    Final(["Array CommandPlan: 3 commands ready for execution"])
    
    style Input fill:#e8f5e9
    style UserResponse fill:#fff3e0
    style Final fill:#f3e5f5
```

---

## Component Dependencies

```mermaid
flowchart TD
    TrainingData["Training Data Preparation"]
    
    TrainingData --> IntentDataset["Intent Parser Dataset: 7K samples"]
    TrainingData --> NormDataset["Command Gen Dataset Normalization"]
    TrainingData --> AugDataset["Command Gen Augmentation: Windows"]
    
    IntentDataset --> System1["System 1: Intent Understanding - 5 stages"]
    
    NormDataset --> System2["System 2: Command Generation - 1 model"]
    AugDataset --> System2
    
    System1 -->|"CanonicalIntent"| Integration["Integration Layer: Orchestration"]
    System2 --> Integration
    
    style IntentDataset fill:#e1f5ff
    style System1 fill:#e1f5ff
    style System2 fill:#fff4e1
    style Integration fill:#e8f5e9
```

---

## System Boundaries

```mermaid
flowchart TB
    subgraph System1["SYSTEM 1"]
        direction TB
        S1Title["Responsibility: Understand user intent
Input: Raw natural language
Output: CanonicalIntent structured JSON"]
        S1Components["Components: 5 stages - 1A, 1B, 1C, 1D, 1E
Models: DistilBERT, BERT-NER, T5-small
No command generation"]
    end
    
    System1 -.->|"CanonicalIntent Interface Contract"| System2
    
    subgraph System2["SYSTEM 2"]
        direction TB
        S2Title["Responsibility: Generate executable commands
Input: CanonicalIntent from System 1
Output: CommandPlan with shell commands"]
        S2Components["Components: 1 model - CodeT5+
Training: NL2Bash + NL2SH-ALFA + Windows
OS-aware: windows, linux, macos
Shell-aware: bash, powershell, cmd, zsh"]
    end
    
    style System1 fill:#e1f5ff
    style System2 fill:#fff4e1
```

---

## Integration Interface: CanonicalIntent

The **CanonicalIntent** object is the contract between System 1 (output) and System 2 (input).

### Structure

```json
{
  "intent_type": "install_runtime",
  "entities": {
    "runtime": "python",
    "package": null,
    "version": "3.10"
  },
  "scope": "system",
  "os_hint": "windows",
  "shell_type": "powershell",
  "confidence": 0.92,
  "missing_fields": [],
  "needs_clarification": false,
  "clarification_question": null
}
```

### Validation Rules for System 2

```mermaid
flowchart LR
    Intent["CanonicalIntent"] --> Check1{"needs_clarification == false?"}
    Check1 -->|"No"| Reject1["Reject"]
    Check1 -->|"Yes"| Check2{"missing_fields == empty?"}
    Check2 -->|"No"| Reject2["Reject"]
    Check2 -->|"Yes"| Check3{"intent_type is atomic?"}
    Check3 -->|"No"| Reject3["Reject: Must decompose first"]
    Check3 -->|"Yes"| Accept["Valid for System 2"]
    
    style Accept fill:#c8e6c9
    style Reject1 fill:#ffcdd2
    style Reject2 fill:#ffcdd2
    style Reject3 fill:#ffcdd2
```

**Required Conditions:**
- `needs_clarification` must be `false`
- `missing_fields` must be empty array `[]`
- `intent_type` must be atomic (not hierarchical)
- All required entities must be present (no `null` for mandatory fields)

---

## See Also

- [System 1 Documentation](./system-1-intent-understanding/README.md)
- [System 2 Documentation](./system-2-command-generation/README.md)
- [Dataset Documentation](./datasets/README.md)
- [Integration Guide](./integration/README.md)
