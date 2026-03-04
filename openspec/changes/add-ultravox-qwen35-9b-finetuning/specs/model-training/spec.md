## ADDED Requirements

### Requirement: Knowledge Distillation Training
The system SHALL train the multimodal projector (and optionally the Whisper encoder) using knowledge distillation loss, matching the frozen Qwen 3.5 9B backbone's token logit distribution when given text-only input.

#### Scenario: KD loss computation
- **WHEN** an audio sample and its transcript are provided
- **THEN** the system computes the Qwen 3.5 9B text-only logits from the transcript
- **AND** trains the audio pathway to produce matching logits from the audio input

#### Scenario: Frozen backbone
- **WHEN** training is running
- **THEN** Qwen 3.5 9B backbone weights remain frozen (no gradient updates)
- **AND** only the projector and optionally the Whisper encoder receive gradient updates

### Requirement: Training Data Pipeline
The system SHALL support a training data mix including ASR datasets (speech-to-text), ASR datasets with LLM-generated continuations, and speech translation datasets, following the Ultravox v0.7 data preparation approach.

#### Scenario: Dataset loading
- **WHEN** training is initiated
- **THEN** datasets are loaded from configured sources via the Ultravox data registry
- **AND** each sample contains at minimum an audio field and a text continuation field

### Requirement: Distributed Training
The system SHALL support distributed multi-GPU training via torchrun with gradient checkpointing enabled for memory efficiency.

#### Scenario: Multi-GPU training
- **WHEN** training is launched with torchrun on multiple GPUs
- **THEN** training runs in distributed data-parallel mode with bf16 precision
- **AND** gradient checkpointing reduces peak VRAM usage

### Requirement: Training Configuration
The system SHALL provide YAML-based training configuration specifying the Qwen 3.5 9B text model, Whisper-large-v3-turbo audio model, projector hyperparameters, learning rate schedule, and dataset mix.

#### Scenario: Config-driven training
- **WHEN** a training YAML config is provided with `text_model: Qwen/Qwen3.5-9B`
- **THEN** the system initializes the correct model architecture and training pipeline
- **AND** all hyperparameters are configurable via the YAML file

### Requirement: Checkpoint Management
The system SHALL save training checkpoints at configurable intervals and support resuming from checkpoints.

#### Scenario: Checkpoint save and resume
- **WHEN** a checkpoint interval is reached during training
- **THEN** the projector weights, encoder weights (if trained), optimizer state, and training step are saved
- **AND** training can resume from any saved checkpoint
