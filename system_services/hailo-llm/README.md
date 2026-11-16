# Hailo LLM Service

`hailo-llm` serves configured Hailo GenAI LLMs through an OpenAI-compatible chat API. It uses the shared Hailo device manager for model loading, inference, and serialization with other Hailo services. The service does not open its own VDevice.

## Requirements and installation

- Raspberry Pi 5 with Hailo-10H and a supported HailoRT GenAI installation.
- `hailo-device-manager` installed and running.
- A compiled `.hef` for each configured model, and any corresponding LoRA files available to the manager runtime.

Install the device manager first, then install the LLM API:

```bash
cd system_services/hailo-llm
sudo ./install.sh
```

The installer creates a service account, virtual environment, systemd unit, and YAML-to-JSON configuration. It does not download model artifacts. Put each HEF at the path configured in `/etc/hailo/hailo-llm.yaml`; use `/var/lib/hailo-llm/models` rather than `/home/pi`, which the service sandbox cannot access.

```bash
sudo systemctl status hailo-llm
sudo journalctl -u hailo-llm -f
curl http://localhost:11441/health
```

## Configure models

Edit `/etc/hailo/hailo-llm.yaml`. Each registry key is the public model ID used by OpenAI clients; each entry maps it to a local HEF and optional LoRA adapter.

```yaml
server:
  host: 0.0.0.0
  port: 11441

default_model: qwen2.5-1.5b

generation:
  max_tokens: 128
  temperature: 0.0

models:
  qwen2.5-1.5b:
    hef_path: /var/lib/hailo-llm/models/qwen2.5-1.5b.hef
    lora_name: null
    supports_tools: false
  qwen2.5-coder-1.5b:
    hef_path: /var/lib/hailo-llm/models/qwen2.5-coder-1.5b.hef
    lora_name: coder_adapter
    supports_tools: true
    generation:
      max_tokens: 256
      temperature: 0.2

device_manager:
  status_url: http://127.0.0.1:5099/v1/device/status
  timeout_s: 120
```

The global `NPU_MAX_GENERATED_TOKENS` environment variable overrides the global YAML `generation.max_tokens` default; model-specific settings override that default, and request fields override model defaults. The limit is validated against the model's runtime context capacity together with the estimated prompt size. A positive temperature enables sampling in the GenAI LLM API; zero disables it.

After editing YAML, restart the service. Its systemd unit re-renders JSON from the YAML before each start:

```bash
sudo systemctl restart hailo-llm
```

Models load on demand through the manager. HailoRT 5.3 allows only one GenAI KV-cache holder at a time, so switching between LLM, Whisper, or VLM models unloads the previously cached GenAI model before the new one is loaded. This adds model-load latency; the previous service reloads its model on its next inference. If the KV-cache remains unavailable after eviction, the manager rebuilds its VDevice runtime in-process and retries without restarting the manager service; this unloads all cached models, which reload on demand.

## OpenAI API

### List configured models

```bash
curl http://localhost:11441/v1/models
```

### Non-streaming chat

```bash
curl http://localhost:11441/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-1.5b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is a Hailo NPU?"}
    ],
    "max_tokens": 128
  }'
```

If `model` is omitted, the configured `default_model` is used. A later turn sends its history in the `messages` array; the service does not keep an implicit conversation between requests.

### Streaming chat

```bash
curl -N http://localhost:11441/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-1.5b",
    "messages": [{"role": "user", "content": "Count from one to five."}],
    "stream": true
  }'
```

Text completions are sent as OpenAI-compatible SSE delta chunks, followed by a finish-reason chunk and `data: [DONE]`. Inference is serialized by the device manager; another service's request waits until the active stream finishes. Tool-call output is parsed and emitted after generation completes.

### Function tools

Function declarations are accepted only for models configured with `supports_tools: true`. The model may return a standard `tool_calls` completion; the service does not execute tools. The client sends tool results as `tool` messages in a subsequent request.

## Context and status

```bash
# Clear the default model's runtime context
curl -X POST http://localhost:11441/v1/context/clear

# Clear a selected model's runtime context
curl -X POST http://localhost:11441/v1/context/clear \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder-1.5b"}'

# Inspect the active manager request and queue
curl http://127.0.0.1:5099/v1/device/status
```

Context clearing is serialized with inference and does not unload model weights. The manager status endpoint reports `active_inference` with request/model IDs and start time while generation is running. It is `null` after generation and context cleanup finish.

## Errors and limitations

- Unknown model IDs and unsupported request fields return structured 4xx errors.
- Context overflow returns HTTP 400 with `context_length_exceeded`.
- Device-manager, model-load, and inference failures return structured 503 errors.
- Supported message content is text. Supported generation fields are `max_tokens`/`max_completion_tokens` and `temperature`; other OpenAI request fields are rejected.
- Only one model inference runs at a time across all manager clients. Requests queue; model switching may trigger LRU eviction and reload.

See [API_SPEC.md](API_SPEC.md) for the endpoint contract.

## Uninstall

```bash
sudo ./uninstall.sh
```

Use `--purge-data` to remove model files and configuration, and `--remove-user` to remove the service account/group.
