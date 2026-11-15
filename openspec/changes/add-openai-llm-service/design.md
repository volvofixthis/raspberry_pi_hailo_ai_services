## Context

See proposal.md for the motivation and specs/openai-llm-api/spec.md for the observable API contract. Existing managed services submit JSON requests through `device_manager/device_client.py`; the manager owns one VDevice and serializes synchronous inference. It currently has no plain GenAI LLM handler. The `hailo-llm-server` reference provides useful OpenAI message handling, response shapes, tool-call normalization, context checks, and Hailo GenAI usage, but constructs its own VDevice and therefore cannot be copied as-is into this multi-service runtime.

## Goals / Non-Goals

**Goals:**
- Keep Hailo VDevice and LLM model ownership in the device manager.
- Preserve an OpenAI-compatible external API while adapting generation and context operations to the manager IPC contract.
- Keep manager protocol additions compatible with current one-request/one-response clients.
- Keep conversation messages request-scoped by default and serialize all Hailo operations.

**Non-Goals:**
- Replace or alter the existing `hailo-ollama` service or Ollama API.
- Execute declared client-side functions on behalf of API callers.
- Implement API-key authentication, remote model downloads, guaranteed simultaneous residency of every configured model, or persistent named conversations in this change.
- Claim compatibility with every optional OpenAI API field; unsupported fields/features return a structured client error.

## Decisions

1. **Add an `llm` model type to the device manager.** The manager will construct each selected Hailo GenAI LLM using its existing VDevice and register/cache it under the existing model registry keyed by model type and artifact path. The LLM handler owns load, unload, prompt-template/context metadata, generation, context clearing, and context-capacity reporting. The service process will not import a Hailo runtime or access `/dev/hailo0`.
   - *Alternative considered:* run the reference server with direct `VDevice` ownership. Rejected because it recreates the resource conflict this project is designed to avoid.

2. **Extend the manager IPC with an opt-in streaming operation.** Keep existing `infer` request/response framing unchanged. Add a streaming request mode that emits token/event frames followed by a terminal success/error frame; update the client to consume those frames as an async iterator. The manager's single worker remains responsible for running the Hailo generator and forwarding tokens, so another device operation cannot interleave with an active stream.
   - *Alternative considered:* return a completed response and simulate SSE by splitting the final string. Rejected because it is not token streaming and increases perceived response latency.

3. **Expose the active manager operation through an out-of-band status snapshot.** The manager records active request metadata when its worker begins inference and clears it only after inference and cleanup finish. Its existing threaded HTTP status endpoint reads this metadata directly rather than enqueueing a status action, so operators can observe an active stream while the inference worker is occupied. Include the public model ID in the manager request metadata; protect the snapshot shared between the worker and HTTP status thread.
   - *Alternative considered:* query active state through the manager's serialized socket action. Rejected because that status request would wait behind the inference it is meant to report.

4. **Use a thin HTTP service with OpenAI normalization.** Add a separately installable `system_services/hailo-llm` app. It validates OpenAI-style request models, resolves the request's public model ID against the operator-configured registry, renders ordered messages through that model's HEF prompt template, applies per-model/request generation options, and formats JSON or SSE responses. Reuse the reference's tool-call parsing concepts but keep tool execution with API clients. The registry contains local HEF paths and optional LoRA settings; API callers cannot supply paths or initiate downloads. If `model` is omitted, use `default_model`.
   - *Alternative considered:* expose only the manager's proprietary JSON protocol. Rejected because clients would need a project-specific adapter instead of standard OpenAI client configuration.
   - *Alternative considered:* accept arbitrary model paths in chat requests or download model names at runtime. Rejected because it bypasses operator-controlled artifact configuration and complicates resource and error handling.

5. **Treat conversation history as request input and clear runtime state between requests.** The API caller resends conversation history in `messages`. The manager serializes the complete operation; the LLM handler clears residual SDK context for the selected model before each request by default and offers model-specific context clearing (defaulting to `default_model`). Context usage/capacity and loaded state are reported through manager metadata rather than by opening a second runtime in the HTTP service.
   - *Alternative considered:* retain a shared implicit conversation in the model across HTTP requests. Rejected because API clients are independent and requests from different callers must not contaminate one another.

6. **Deploy as an independent managed service depending on the device manager.** Follow other services' YAML-to-JSON config-rendering, systemd, installer, health, and documentation conventions. Configure a `models` registry mapping public IDs to local HEF paths with optional per-model LoRA and generation settings, a `default_model`, host/port, timeout, and any supported global generation defaults. Load a selected model through the manager on demand and let the manager's cache/resource policy determine reuse or eviction; the service does not promise every configured model stays resident. Set the default listener port to `11441`, the next available service port after the current `11440` assignment.
   - *Alternative considered:* fold this API into `hailo-ollama`. Rejected to retain that binary's existing lifecycle/API and provide the LLM a manager-compatible path.

## Risks / Trade-offs

- [GenAI SDK/context behavior differs between runtime versions] → Validate the required LLM, context, and streaming APIs on the installed HailoRT target and keep SDK interaction behind the manager handler.
- [Switching among configured models can exceed Hailo memory or increase first-token latency during model loads] → Load models on demand through the manager, report load/resource failures for the selected model, and avoid promising simultaneous residency.
- [Long-running token streams occupy the manager's single worker] → Enforce the configured inference timeout, surface busy/unavailable errors through the API, and terminate streams with an explicit error/final frame.
- [Prompt templates or tool-call formats vary by model] → Use the configured model's template and normalize only supported tool-call output; reject tool use for models without configured support.
- [Existing manager IPC assumes one response frame] → Introduce streaming as an explicit opt-in action and leave the existing non-stream framing and client methods unchanged.
- [LLM context capacity may be smaller than submitted history] → Estimate prompt usage, check against manager-reported capacity before inference, and return an actionable context-overflow client error.

## Migration Plan

1. Add and validate the manager LLM handler and backward-compatible streaming IPC before adding the HTTP service.
2. Install/enable the manager as usual, install the new `hailo-llm` service with its model registry and port, then verify model discovery, requests selecting different model IDs, health, normal/streamed chat, and model-specific context clearing.
3. Roll back by stopping/disabling `hailo-llm` and reverting the manager/service package changes. Existing services and `hailo-ollama` continue using their current protocol and deployment.
