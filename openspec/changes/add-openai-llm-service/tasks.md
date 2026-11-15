## 1. Device Manager LLM Runtime

- [x] 1.1 Add an LLM model handler using the manager-owned VDevice, with separate model instances keyed by model type and artifact path, prompt-template/context metadata, generation options, and context clearing; verify multi-model loading/unloading and isolation with mocked GenAI tests.
- [x] 1.2 Add manager actions for LLM metadata and context clearing, and expose a thread-safe `active_inference` snapshot from `/v1/device/status` with request ID, public model ID, model type, start time, and queue depth; verify idle and active status responses while a stream is running.
- [x] 1.3 Add an opt-in streamed inference IPC path and async client iterator, keeping the manager's single worker occupied for each stream; verify token ordering, terminal success/error frames, timeout handling, active-status cleanup, and unchanged non-streaming calls with tests.
- [x] 1.4 Register the LLM handler and validate per-model context reset/isolation and prompt-capacity failures; verify manager-level tests that model switching and successive requests do not reuse another model's runtime context.

## 2. OpenAI-Compatible LLM API

- [x] 2.1 Create the `system_services/hailo-llm` service skeleton and YAML-configured `models` registry with unique IDs, per-model artifacts/LoRA/defaults, `default_model`, and port `11441`; verify YAML-to-JSON rendering and reject missing defaults, duplicate IDs, or invalid artifacts.
- [x] 2.2 Implement selected-model loading via the device-manager client plus readiness and `/v1/models` endpoints; verify all configured IDs are listed, loaded state is reported, and unavailable-manager health responses are clear.
- [x] 2.3 Implement `/v1/chat/completions` request validation, model-ID selection/defaulting, ordered role/template rendering, context-capacity checks, and JSON completion responses; verify success for different IDs, unknown-model, unsupported-input, overflow, load failure, and manager-error cases.
- [x] 2.4 Implement streamed SSE completions and OpenAI-style tool-call normalization without executing tools; verify incremental content chunks, finish reasons, `[DONE]`, and tool-call/tool-result round trips.
- [x] 2.5 Implement model-specific `/v1/context/clear` (defaulting to `default_model`) and expose per-model context usage/capacity in health; verify target selection, default selection, and unknown-model/unavailable/busy errors.

## 3. Deployment, Documentation, and Integration

- [x] 3.1 Add installer, systemd unit, service user/permissions, configuration examples, and uninstall support following neighboring managed services; verify install/uninstall scripts pass static checks and the service depends on the device manager.
- [x] 3.2 Document the YAML model registry, per-model LoRA/defaults, OpenAI client model selection, streaming, context behavior, supported options, and service lifecycle; verify examples use registered IDs and port `11441`.
- [ ] 3.3 Run relevant manager and service tests, then on Hailo hardware verify loading/switching between two configured model IDs, OpenAI client non-streaming and streaming requests, `/v1/device/status` visibility during a stream and idle afterward, tool-call formatting where supported, model-specific context clearing, resource-failure handling, and serialization alongside another managed service.
