## ADDED Requirements

### Requirement: HuggingFace Model Export
The system SHALL export the trained model in standard HuggingFace format, compatible with `transformers` AutoModel loading patterns.

#### Scenario: Model loading
- **WHEN** a user loads the model via `AutoModel.from_pretrained("model-id")`
- **THEN** the full Ultravox model (encoder + projector + Qwen 3.5 backbone) is loaded correctly
- **AND** the model is ready for inference without additional setup

### Requirement: vLLM Serving Compatibility
The system SHALL be compatible with vLLM for high-throughput production inference serving.

#### Scenario: vLLM deployment
- **WHEN** the model is loaded in vLLM with the Ultravox architecture config
- **THEN** the model serves audio+text requests with batched inference
- **AND** streaming text generation is supported

### Requirement: Real-Time Speech Inference
The system SHALL support streaming text generation from audio input with latency suitable for real-time conversational AI.

#### Scenario: Streaming generation
- **WHEN** audio input is provided for inference
- **THEN** text tokens are generated and streamed as they become available
- **AND** first-token latency is minimized by disabling thinking mode
