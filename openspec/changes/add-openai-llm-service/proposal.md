## Why

The project has no LLM service that participates in the shared Hailo device-manager architecture: `hailo-ollama` owns the device directly, while the other APIs use the manager. A managed OpenAI-compatible chat API will let standard clients run local Hailo LLMs alongside the other services without creating a competing VDevice owner.

## What Changes

- Add a configurable Hailo LLM API service based on the `hailo-llm-server` reference, exposing OpenAI-style chat completions and model discovery for a YAML-configured registry of model IDs and HEF artifacts.
- Support ordinary and streamed chat responses, chat message history, generation options, and function-tool declarations where supported by the configured model.
- Add an LLM inference path to the device manager so it owns and loads selected registered LLMs on demand, and serializes requests with existing services.
- Provide health/context visibility and an explicit way to clear the LLM's hardware inference context; keep request history/API behavior isolated between callers.
- Expose active inference metadata through device-manager status so operators can identify a running stream and the requests waiting behind it.
- Document installation, model configuration, API usage, and device-manager requirements.

## Capabilities

### New Capabilities
- `openai-llm-api`: OpenAI-compatible, configurable LLM chat service integrated with the shared Hailo device manager.

### Modified Capabilities

## Impact

- `device_manager/hailo_device_manager.py` and `device_manager/device_client.py` for managed LLM loading, inference, context operations, and active-request status.
- A new `system_services/hailo-llm` service, its OpenAI-compatible HTTP API, deployment/configuration files, and documentation.
- The runtime dependency/configuration contract for Hailo GenAI LLM models; the existing `hailo-ollama` service remains independently available.
- API consumers gain a standard `/v1/chat/completions` surface and `/v1/models` discovery, including optional Server-Sent Events streaming.
