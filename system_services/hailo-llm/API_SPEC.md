# Hailo LLM API Specification

Base URL: `http://localhost:11441`

The service provides OpenAI-compatible text chat for a YAML-configured registry of local Hailo GenAI `.hef` models. Requests use the configured public model ID, not a filesystem path. Model loads and inference run through `hailo-device-manager`.

## GET `/v1/models`

Returns all configured model IDs, whether or not they are currently loaded.

```json
{
  "object": "list",
  "data": [
    {"id": "qwen2.5-1.5b", "object": "model", "created": 0, "owned_by": "hailo"}
  ]
}
```

## POST `/v1/chat/completions`

Accepts ordered text messages with roles `system`, `user`, `assistant`, and `tool`. `model` may be omitted to use `default_model`.

Supported request fields:

| Field | Behavior |
|---|---|
| `model` | Configured model ID; defaults to `default_model` |
| `messages` | Required, non-empty ordered array; message `content` must be text or `null` for assistant tool calls |
| `stream` | Optional boolean, default `false` |
| `max_tokens` | Optional output-token limit, defaulted from selected model config |
| `max_completion_tokens` | Alias for `max_tokens`; do not supply both |
| `temperature` | Optional value 0–2; positive values are passed to GenAI and enable sampling, zero disables sampling |
| `top_p` | Optional nucleus-sampling value greater than 0 and at most 1; supplying it enables sampling unless `temperature: 0` is also supplied |
| `top_k` | Optional positive integer; supplying it enables sampling unless `temperature: 0` is also supplied |
| `tools` | Optional OpenAI function definitions; model must have `supports_tools: true` |
| `tool_choice` | Optional `auto` or `none` |

Sampling parameters may also be set in global or per-model `generation` config. If `top_p` and `top_k` are omitted, the GenAI runtime's defaults apply. Unsupported fields and non-text content return `400 invalid_request_error`. Unknown model IDs return `404` with code `model_not_found`. Prompt estimate plus requested output tokens must fit the model's runtime context capacity; otherwise the service returns `400 context_length_exceeded`.

### Non-streaming response

Successful requests return an OpenAI-shaped `chat.completion` object with one assistant choice, `finish_reason` (`stop` or `tool_calls`), and estimated usage counts.

### Streaming response

For `stream: true`, response content type is `text/event-stream`. Text is emitted as `chat.completion.chunk` events, followed by a finish-reason chunk and `data: [DONE]`. When tools are requested, the service buffers model output for tool-call parsing and emits any recognized call after generation.

Device-manager errors after the SSE response begins are emitted as an SSE `data` error object followed by `[DONE]`.

## GET `/health`

Returns HTTP 200 when the service can query the device manager, otherwise 503. The JSON includes configured models and their manager-loaded state, last known context usage/capacity, manager queue depth, and `active_inference` when available.

```json
{
  "status": "ok",
  "model_count": 2,
  "models": [
    {"id": "qwen2.5-1.5b", "loaded": true, "context_used": 0, "context_max": 2048}
  ],
  "active_inference": null,
  "queue_depth": 0
}
```

## POST `/v1/context/clear`

Clears one registered model's runtime context. The optional JSON body selects a model; when omitted, the service uses `default_model`.

```bash
curl -X POST http://localhost:11441/v1/context/clear \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b"}'
```

Returns `204 No Content` on success. Clearing an unloaded model is a successful no-op. The operation is queued behind active inference and does not unload the model or erase message history held by the client.

## Device manager status

`GET http://127.0.0.1:5099/v1/device/status` exposes `active_inference` as `null` while idle or as an object containing `request_id`, `model_id` (when supplied), `model_type`, `model_path`, and Unix `started_at` seconds during inference. The snapshot is read out-of-band from the serialized inference queue.

## Error responses

Errors use the OpenAI shape:

```json
{"error":{"message":"model not found: missing","type":"invalid_request_error","code":"model_not_found"}}
```

- `400` invalid request or context overflow
- `404` unknown model ID
- `503` model loading, device manager, or inference unavailable
