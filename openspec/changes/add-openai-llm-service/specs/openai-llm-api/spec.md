## Purpose

Provide a configurable local Hailo LLM service through a broadly usable OpenAI-compatible chat interface, while sharing exclusive Hailo device ownership with the project's other managed APIs.

## ADDED Requirements

### Requirement: OpenAI-compatible chat completion requests
The service SHALL expose `POST /v1/chat/completions` accepting a model identifier, an ordered list of `system`, `user`, `assistant`, and `tool` messages, and supported generation options. The service SHALL preserve message order and roles when forming the model input. A request for an unknown model or an unsupported input SHALL return a structured client error without running inference.

#### Scenario: Complete a chat request
- **WHEN** a client posts a valid model identifier and chat messages with `stream` omitted or false
- **THEN** the service returns an OpenAI-shaped `chat.completion` response containing an assistant choice, finish reason, and token usage

#### Scenario: Include conversation history
- **WHEN** a request contains system instructions and prior user, assistant, or tool messages before the latest user message
- **THEN** the model receives those messages in their original order as the request's conversation history

#### Scenario: Reject an unknown model
- **WHEN** a client requests a model identifier not served by this instance
- **THEN** the service returns a structured invalid-model client error and does not submit inference

#### Scenario: Reject unsupported message content
- **WHEN** a client submits malformed or unsupported message content
- **THEN** the service returns a structured invalid-request client error that identifies the request problem

### Requirement: OpenAI-compatible streamed responses
The service SHALL support `stream: true` for chat completions using Server-Sent Events. A successful stream SHALL emit OpenAI-compatible chat completion chunks, a terminal finish-reason chunk, and the `[DONE]` sentinel. Streamed inference SHALL continue to be serialized by the shared device manager.

#### Scenario: Stream generated tokens
- **WHEN** a client sends a valid chat completion request with `stream: true`
- **THEN** the service emits assistant content incrementally as OpenAI-compatible SSE chunks and terminates the stream with `[DONE]`

#### Scenario: NPU is unavailable or busy during streaming
- **WHEN** a streamed request cannot start because the managed inference device is unavailable or busy
- **THEN** the stream reports an OpenAI-compatible error and terminates without returning a successful completion

### Requirement: Model discovery and readiness
The service SHALL expose `GET /v1/models` listing every model ID in its configured model registry. Its health endpoint SHALL report service readiness and per-model loaded state and LLM context capacity/usage when the device manager can provide them.

#### Scenario: Discover configured models
- **WHEN** a client requests `GET /v1/models` from a ready service
- **THEN** the response is an OpenAI-shaped model list containing every public model ID in the configured registry

#### Scenario: Report service readiness
- **WHEN** a client requests the service health endpoint
- **THEN** the response distinguishes a ready LLM service from one whose device-manager connection is unavailable and reports each model's loaded state and context usage/capacity when available

### Requirement: Active inference visibility
The device manager's `GET /v1/device/status` endpoint SHALL expose `active_inference` as `null` when no inference is executing, or an object containing the active request ID, model ID when supplied, model type, and start time. The status endpoint SHALL report this snapshot while inference is active without waiting behind the inference queue. The manager SHALL retain the active status through generation and context cleanup, then clear it when the operation completes or fails.

#### Scenario: Observe an active streamed completion
- **WHEN** a client requests device status while an LLM stream is generating
- **THEN** the status response includes the stream's request ID, model ID, model type, and start time, along with the current queue depth

#### Scenario: Observe idle device state after completion
- **WHEN** the streamed or non-streamed inference and its context cleanup have completed
- **THEN** a device status response reports `active_inference` as `null`

#### Scenario: Read status while inference occupies the worker
- **WHEN** a client requests `/v1/device/status` while the manager is generating an inference response
- **THEN** the status request returns a current snapshot without being queued behind that inference

### Requirement: Shared Hailo runtime ownership
LLM model loading and inference SHALL be performed through the Hailo device manager's exclusive device ownership and serialized request path. The LLM service SHALL NOT create its own Hailo VDevice. A chat request SHALL select a registered model, loading it through the manager on demand if necessary. LLM requests SHALL not run concurrently with other device-manager inference operations.

#### Scenario: Load the selected LLM on demand
- **WHEN** a chat request selects a configured model that is not currently loaded
- **THEN** the manager loads that model using its own VDevice before inference and retains exclusive device ownership

#### Scenario: Switch between configured models
- **WHEN** successive chat requests select different configured model IDs
- **THEN** each request is run against its selected model, with model load/unload or reuse handled through the manager's model registry and device-resource policy

#### Scenario: Share the NPU with another managed service
- **WHEN** LLM inference and another managed service's inference are requested at overlapping times
- **THEN** the manager serializes the operations and neither service opens an independent Hailo device context

#### Scenario: Device manager is unavailable
- **WHEN** the LLM service cannot connect to the device manager
- **THEN** its readiness endpoint reports unavailable and chat requests return a structured service-unavailable error

### Requirement: Conversation and runtime context isolation
Each chat request SHALL use the message history supplied in that request as its conversation context. By default, residual runtime context from an earlier request SHALL be cleared or otherwise prevented from affecting the next request for the selected model, so one caller's conversation is not implicitly exposed to another. The service SHALL expose `POST /v1/context/clear` to explicitly clear runtime context for a selected registered model; if no model is supplied, the service SHALL clear the configured default model's context.

#### Scenario: Keep independent requests isolated
- **WHEN** two chat requests are submitted successively without repeating the first request's message history
- **THEN** the second request is not conditioned on the first request's messages or generated response

#### Scenario: Preserve a multi-turn conversation
- **WHEN** a client resubmits prior user and assistant messages as part of a later chat request
- **THEN** those supplied messages are included as the later request's conversation context

#### Scenario: Explicitly clear runtime context
- **WHEN** a client posts to `/v1/context/clear` with a registered model ID, or without one
- **THEN** the service clears that model's runtime context through the manager (the configured default model when omitted) and returns success, or reports a structured unknown-model, unavailable, or busy error

### Requirement: Function tool compatibility
The service SHALL accept OpenAI-style function tool declarations for configured models that support tool calling, and SHALL return recognized model tool calls in the OpenAI chat completion message format. The service SHALL NOT execute client tools; clients remain responsible for execution and may submit tool results in subsequent messages.

#### Scenario: Return a function tool call
- **WHEN** a tool-capable model responds with a valid call matching a declared function
- **THEN** the completion contains an OpenAI-style `tool_calls` entry and uses `tool_calls` as the finish reason

#### Scenario: Continue after a client tool result
- **WHEN** a client submits the assistant tool-call message and corresponding tool result in a subsequent request
- **THEN** the service includes those supplied messages in the next model input

#### Scenario: Model does not support tools
- **WHEN** a request supplies tools to a configured model that does not support tool calling
- **THEN** the service returns a structured unsupported-request error instead of silently claiming that a tool was called

### Requirement: Configurable model registry and generation defaults
The service SHALL allow operators to configure a registry of unique public model IDs, each mapped to a local model artifact and optional LoRA and generation defaults, plus a default model ID, bind address, and port. The configured model artifacts SHALL be selected only by registry ID; chat requests SHALL NOT supply arbitrary model paths or trigger remote downloads. A chat request with no model ID SHALL use the configured default model. Per-request values for supported OpenAI generation options SHALL override the selected model's configured defaults without changing settings for later requests.

#### Scenario: Configure multiple model IDs
- **WHEN** an operator configures two or more unique model IDs mapped to local artifacts
- **THEN** each ID is discoverable through `GET /v1/models` and can be selected independently in a chat request

#### Scenario: Apply selected model defaults
- **WHEN** a valid completion request omits its model ID or optional generation settings
- **THEN** the service uses the configured default model when needed and applies the selected model's configured generation defaults

#### Scenario: Override generation settings for one request
- **WHEN** a valid request supplies supported generation settings such as temperature or maximum generated tokens
- **THEN** those values apply only to that request and subsequent requests continue to use their own values or the selected model's configured defaults

#### Scenario: Reject an unregistered model ID
- **WHEN** a chat request or context-clear request names a model ID not present in the configured registry
- **THEN** the service returns a structured unknown-model client error and does not load an artifact

#### Scenario: Report a selected model load failure
- **WHEN** a registered model cannot be loaded by the manager because its artifact is unavailable or device resources are insufficient
- **THEN** the service returns a structured service-unavailable error for that request and remains able to serve other registered models when the manager remains healthy
