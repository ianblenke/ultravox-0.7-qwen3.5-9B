## ADDED Requirements

### Requirement: Three-Component Model Architecture
The system SHALL implement a multimodal speech-language model with three components: an audio encoder (whisper-large-v3-turbo), a multimodal projector (UltravoxProjector), and a frozen LLM backbone (Qwen 3.5 9B).

#### Scenario: Audio input processing pipeline
- **WHEN** raw audio waveform is provided as input
- **THEN** the Whisper encoder extracts continuous audio features
- **AND** the UltravoxProjector transforms features to LLM embedding space
- **AND** the Qwen 3.5 9B backbone generates text output

### Requirement: Multimodal Projector Configuration
The UltravoxProjector SHALL bridge Whisper-large-v3-turbo output dimension (1280) to Qwen 3.5 9B hidden dimension (4096) using StackAudioFrames (stack_factor=8), LayerNorm, Linear projection, SwiGLU activation, RMSNorm, and final Linear projection.

#### Scenario: Projector dimension mapping
- **WHEN** Whisper encoder outputs features of dimension 1280
- **THEN** the projector transforms them to dimension 4096 matching Qwen 3.5 9B

#### Scenario: Temporal compression
- **WHEN** audio frames are processed with stack_factor=8
- **THEN** 8 consecutive audio frames are stacked, reducing sequence length by 8x

### Requirement: Audio Token Integration
The system SHALL register an `<|audio|>` special token in the Qwen 3.5 tokenizer and replace placeholder embeddings with projected audio embeddings during the forward pass.

#### Scenario: Audio placeholder replacement
- **WHEN** input text contains `<|audio|>` placeholder tokens
- **THEN** the model replaces those token embeddings with projected audio features from the Whisper encoder
- **AND** no collision occurs with Qwen 3.5's existing 248,320 vocabulary tokens

### Requirement: Thinking Mode Disabled
The system SHALL disable Qwen 3.5 9B's chain-of-thought thinking mode for real-time speech processing to minimize latency.

#### Scenario: No thinking tokens in output
- **WHEN** generating text from audio input
- **THEN** no `<think>` or `</think>` tokens are produced in the output
- **AND** generation proceeds without chain-of-thought reasoning overhead

### Requirement: bf16 Precision
The system SHALL use bf16 mixed precision for all model components. QLoRA/4-bit quantization SHALL NOT be used due to Qwen 3.5's hybrid DeltaNet architecture sensitivity to quantization.

#### Scenario: Precision enforcement
- **WHEN** the model is loaded for training or inference
- **THEN** all parameters use bf16 or fp32 precision
- **AND** no 4-bit or 8-bit quantization is applied to the Qwen 3.5 backbone
