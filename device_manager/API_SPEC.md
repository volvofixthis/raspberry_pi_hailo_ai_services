# Hailo Device Manager API

Base transport: Unix socket at `/run/hailo/device.sock`

Authentication: none (local socket permissions control access).

Protocol: length-prefixed JSON messages over a Unix socket. Ordinary actions use one request and one response frame. `infer_stream` uses one request followed by token event frames and a terminal final/error frame; each frame is a JSON object carrying the same `request_id`.

Optional HTTP endpoint (if enabled via `HAILO_DEVICE_HTTP_BIND`, default `127.0.0.1:5099`):
- `GET /v1/device/status` returns a live status snapshot, including the active inference if any, without waiting behind the inference queue.

## Request Envelope

All requests are JSON objects with an `action` field.

Common fields:
- `action` (string, required)
- `request_id` (string, optional) - echoed back in the response

Example:
```json
{
  "action": "ping",
  "request_id": "abc-123"
}
```

## Response Envelope

Common fields:
- `status` (string) - typically `ok` for success
- `error` (string or structured object) - present on failure
- `request_id` (string) - echoed back if provided in the request

Example success:
```json
{
  "status": "ok",
  "uptime_seconds": 12.3,
  "request_id": "abc-123"
}
```

Example error:
```json
{
  "error": "Model file not found: /path/to/model.hef",
  "request_id": "abc-123"
}
```

## Actions

### ping
Check connection and uptime.

Request:
```json
{
  "action": "ping",
  "request_id": "..."
}
```

Response (200 OK):
```json
{
  "status": "ok",
  "device_id": "0001:01:00.0",
  "loaded_models": [],
  "uptime_seconds": 123.4,
  "socket_path": "/run/hailo/device.sock",
  "queue_depth": 0,
  "total_lru_evictions": 0,
  "request_id": "..."
}
```

### status
Get device status and loaded model list.

Request:
```json
{
  "action": "status",
  "request_id": "..."
}
```

Response (200 OK):
```json
{
  "status": "ok",
  "device_id": "0001:01:00.0",
  "loaded_models": [
    {
      "model_type": "vlm",
      "model_path": "/path/to/model.hef",
      "model_params": {},
      "loaded_at": 1700000000.0,
      "last_used": 1700000001.5
    }
  ],
  "uptime_seconds": 123.4,
  "socket_path": "/run/hailo/device.sock",
  "queue_depth": 1,
  "active_inference": null,
  "total_lru_evictions": 0,
  "request_id": "..."
}
```

### device_status
Get detailed device status (firmware, temperature, loaded networks).

Request:
```json
{
  "action": "device_status",
  "request_id": "..."
}
```

Response (200 OK):
```json
{
  "status": "ok",
  "device": {
    "device_id": "0001:01:00.0",
    "architecture": "HAILO10H",
    "fw_version": "5.5.0 (release,app)",
    "temperature_celsius": 53.5
  },
  "networks": {
    "status": "ok",
    "source": "device_manager",
    "network_count": 1,
    "networks": [
      {
        "name": "yolov8n.hef",
        "model_type": "vision",
        "model_path": "/path/to/yolov8n.hef",
        "model_params": {},
        "loaded_at": 1700000000.0,
        "last_used": 1700000001.5
      }
    ]
  },
  "uptime_seconds": 123.4,
  "queue_depth": 0,
  "active_inference": {
    "request_id": "...",
    "model_id": "qwen2.5-1.5b",
    "model_type": "llm",
    "model_path": "/var/lib/hailo-llm/models/qwen.hef",
    "started_at": 1700000001.5
  },
  "total_lru_evictions": 0,
  "request_id": "..."
}
```

### load_model
Load a model into device memory.

Request:
```json
{
  "action": "load_model",
  "model_path": "/path/to/model.hef",
  "model_type": "llm",
  "model_params": {},
  "request_id": "..."
}
```

Response (200 OK):
```json
{
  "status": "ok",
  "model_path": "/path/to/model.hef",
  "model_type": "llm",
  "message": "Model loaded",
  "model_info": {
    "prompt_template": "<model prompt template>",
    "context_capacity": 2048
  },
  "request_id": "..."
}
```

### infer
Run a complete inference. If the model is not loaded, it is loaded first using the provided `model_params`. LLM requests may include optional `model_id` for active status reporting.

Request:
```json
{
  "action": "infer",
  "model_path": "/path/to/model.hef",
  "model_type": "llm",
  "model_params": {},
  "input_data": {"prompt": "Hello", "max_generated_tokens": 128},
  "model_id": "qwen2.5-1.5b",
  "request_id": "..."
}
```

Response (200 OK):
```json
{
  "status": "ok",
  "result": {
    "text": "Hello!",
    "context_used": 12,
    "context_capacity": 2048
  },
  "inference_time_ms": 42,
  "request_id": "..."
}
```

### infer_stream
Run token-streaming inference. This action currently supports `model_type: "llm"`. It occupies the serialized manager worker until generation and context cleanup finish. The manager sends framed events in order:

```json
{"event":"token","token":"Hello","request_id":"..."}
{"event":"final","status":"ok","completion_tokens":1,"inference_time_ms":123,"request_id":"..."}
```

On failure, the terminal frame has `event: "error"` and an `error` value. Other actions continue to use the existing one-response protocol.

### model_info
Read metadata for an already-loaded LLM. This action does not load an unloaded model.

```json
{"action":"model_info","model_type":"llm","model_path":"/path/to/model.hef","model_params":{}}
```

The response includes `model_info.prompt_template`, `context_capacity`, and current `context_used`.
Response example:
```json
{
  "status": "ok",
  "model_type": "llm",
  "model_path": "/path/to/model.hef",
  "model_info": {
    "prompt_template": "<model prompt template>",
    "context_capacity": 2048,
    "context_used": 12
  },
  "request_id": "..."
}
```

### clear_context
Clear an already-loaded LLM's context without unloading its model. Clearing an unloaded model succeeds as a no-op. The action is serialized behind active inference.

```json
{"action":"clear_context","model_type":"llm","model_path":"/path/to/model.hef","model_params":{}}
```
Response (200 OK):
```json
{
  "status": "ok",
  "loaded": true,
  "model_path": "/path/to/model.hef",
  "request_id": "..."
}
```

### unload_model
Unload a model from device memory.

Request:
```json
{
  "action": "unload_model",
  "model_path": "/path/to/model.hef",
  "model_type": "vlm",
  "request_id": "..."
}
```

Response (200 OK):
```json
{
  "status": "ok",
  "message": "Model unloaded",
  "request_id": "..."
}
```

## Model Types

Supported `model_type` values:
- `vlm` - Hailo GenAI VLM via `hailo_platform.genai.VLM`
- `vlm_chat` - VLM chat using `VLM.generate_all`
- `clip` - CLIP image/text encoders using `create_infer_model`
- `whisper` - Speech-to-text transcription via `hailo_platform.genai.Speech2Text`
- `ocr` - Optical character recognition (detection + recognition) with device manager serialization
- `depth` - Monocular depth estimation (e.g., scdepthv3)
  - Input: Preprocessed image tensor (float32 or uint8, [1,3,H,W] in NCHW format)
  - Output: Depth map tensor (float32, [1,1,H,W] in NCHW format)
- `llm` - Hailo GenAI LLM, chat text generation and streaming

## Tensor Payload Format

Some model types (for example `vlm_chat` and `clip`) send tensors in `input_data`.
The `tensor` model type accepts one input under `input`, or multiple named
inputs under `inputs` for HEFs with more than one input. Single-input requests
remain backward compatible.

Tensor object:
```json
{
  "dtype": "uint8",
  "shape": [1, 224, 224, 3],
  "data_b64": "..."
}
```

For a multi-input HEF, use its exact input names:
```json
{
  "inputs": {
    "model/input_layer1": {"dtype": "uint8", "shape": [1, 128, 128, 3], "data_b64": "..."},
    "model/input_layer2": {"dtype": "uint8", "shape": [1, 320, 320, 3], "data_b64": "..."}
  }
}
```

- `dtype`: NumPy dtype string
- `shape`: array shape
- `data_b64`: base64-encoded raw tensor bytes

## Error Handling

Errors are returned in a JSON object with an `error` field.

Common errors:
- `model_path required`
- `model_path and input_data required`
- `Unsupported model_type: <value>`
- `tensor must include dtype, shape, and data_b64`
- `Message too large: <bytes> bytes`

HailoRT shutdown and communication-closed errors during model loading or
inference trigger a Device Manager runtime reset. Model loading is retried once
after a successful reset; if the retry also fails, the request returns an
`error` and the manager resets the context again so later requests can recover.
If the reset itself fails, `device_status` reports `status: "error"` and
includes the failure in `runtime_broken_reason`. If the runtime remains
unavailable, the manager finishes the current response and requests a
supervised restart; systemd restarts the manager, and clients should reconnect
and retry the failed inference. If graceful shutdown remains blocked, the
manager forces its own exit after five seconds so systemd can complete restart.

The bundled async client retries once after a socket transport failure, which
can occur while the manager restarts. It does not retry manager error responses
or request timeouts.

## Limits

- Maximum message size: `HAILO_DEVICE_MAX_MESSAGE_BYTES` (default 8 MiB)
- Socket path: `HAILO_DEVICE_SOCKET` (default `/run/hailo/device.sock`)

## Example Client Usage

Python async client example:
```python
from device_client import HailoDeviceClient

async with HailoDeviceClient() as client:
    await client.load_model("/path/to/model.hef", model_type="vlm")
    result = await client.infer("/path/to/model.hef", {"prompt": "Hello"}, model_type="vlm")
```
