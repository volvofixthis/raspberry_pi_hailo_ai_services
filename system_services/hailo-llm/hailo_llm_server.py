#!/usr/bin/env python3
"""OpenAI-compatible Hailo LLM HTTP API backed by hailo-device-manager."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from jinja2 import Environment

from config import ModelConfig, ServiceConfig, load_json
try:
    from device_client import DeviceManagerError, HailoDeviceClient
except ImportError:
    from device_manager.device_client import DeviceManagerError, HailoDeviceClient

logger = logging.getLogger("hailo-llm")
XDG_CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", "/etc/xdg"))
CONFIG_JSON = XDG_CONFIG_HOME / "hailo-llm" / "hailo-llm.json"


class APIError(Exception):
    def __init__(self, status: int, message: str, error_type: str, code: str | None = None):
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        super().__init__(message)

    def as_response(self) -> web.Response:
        error: dict[str, Any] = {"message": self.message, "type": self.error_type}
        if self.code:
            error["code"] = self.code
        return web.json_response({"error": error}, status=self.status)


def _error_from_manager(exc: Exception) -> APIError:
    payload = exc.payload if isinstance(exc, DeviceManagerError) else str(exc)
    if isinstance(payload, dict) and payload.get("type") == "context_overflow":
        detail = (
            "The prompt and requested completion exceed the model context capacity "
            f"({payload.get('estimated_prompt_tokens', 0)} estimated prompt tokens + "
            f"{payload.get('max_generated_tokens', 0)} completion tokens > "
            f"{payload.get('context_max', 0)} context tokens)."
        )
        return APIError(400, detail, "invalid_request_error", "context_length_exceeded")
    return APIError(503, f"LLM inference unavailable: {payload}", "server_error", "device_unavailable")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise APIError(400, "messages must be a non-empty array", "invalid_request_error")
    messages: list[dict[str, Any]] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise APIError(400, f"messages[{index}] must be an object", "invalid_request_error")
        role = raw_message.get("role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise APIError(400, f"messages[{index}].role is unsupported", "invalid_request_error")
        content = raw_message.get("content")
        if content is not None and not isinstance(content, str):
            raise APIError(400, f"messages[{index}].content must be text or null", "invalid_request_error")
        raw_tool_calls = raw_message.get("tool_calls")
        if raw_tool_calls is not None:
            if role != "assistant" or not isinstance(raw_tool_calls, list) or any(
                not isinstance(call, dict) for call in raw_tool_calls
            ):
                raise APIError(400, f"messages[{index}].tool_calls is invalid", "invalid_request_error")
        if content is None and role != "assistant" and not raw_tool_calls:
            raise APIError(400, f"messages[{index}].content is required", "invalid_request_error")
        message: dict[str, Any] = {"role": role, "content": content}
        for key in ("name", "tool_call_id", "tool_calls"):
            if key in raw_message:
                message[key] = raw_message[key]
        messages.append(message)
    return messages


def _normalize_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise APIError(400, "tools must be an array", "invalid_request_error")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError(400, f"tools[{index}] must be a function tool", "invalid_request_error")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise APIError(400, f"tools[{index}].function.name is required", "invalid_request_error")
        name = function["name"]
        if name in names:
            raise APIError(400, f"duplicate tool name: {name}", "invalid_request_error")
        names.add(name)
        if not isinstance(function.get("parameters", {}), dict):
            raise APIError(400, f"tools[{index}].function.parameters must be an object", "invalid_request_error")
        tools.append(tool)
    return tools


def _tool_call_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.S))
    for tag in ("tool_call", "function_call"):
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
        )
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index : index + end])
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _extract_tool_specs(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [spec for item in payload for spec in _extract_tool_specs(item)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("tool_calls"), list):
        return [spec for item in payload["tool_calls"] for spec in _extract_tool_specs(item)]
    if isinstance(payload.get("function_call"), dict):
        return [payload["function_call"]]
    function = payload.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return [function]
    if isinstance(payload.get("name"), str) and "arguments" in payload:
        return [{"name": payload["name"], "arguments": payload["arguments"]}]
    return []


def _parse_tool_calls(text: str, declared_names: set[str]) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in _tool_call_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        for spec in _extract_tool_specs(payload):
            name = spec.get("name")
            if not isinstance(name, str) or name not in declared_names:
                continue
            arguments = spec.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    json.loads(arguments)
                    arguments_json = arguments
                except json.JSONDecodeError:
                    arguments_json = json.dumps(arguments, ensure_ascii=False)
            elif isinstance(arguments, dict):
                arguments_json = json.dumps(arguments, ensure_ascii=False)
            else:
                continue
            key = (name, arguments_json)
            if key in seen:
                continue
            seen.add(key)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_json},
                }
            )
    return calls or None


class LLMApiService:
    def __init__(
        self,
        config: ServiceConfig,
        client: HailoDeviceClient | None = None,
        status_provider=None,
    ) -> None:
        self.config = config
        self.client = client or HailoDeviceClient(timeout=config.manager_timeout_s)
        self.status_provider = status_provider
        self._manager_session: aiohttp.ClientSession | None = None
        self._model_locks = {model_id: asyncio.Lock() for model_id in config.models}
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._last_context_used: dict[str, int] = {model_id: 0 for model_id in config.models}
        self._jinja = Environment(autoescape=False)
        self._jinja.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)

    async def close(self) -> None:
        await self.client.disconnect()
        if self._manager_session is not None:
            await self._manager_session.close()
            self._manager_session = None

    def resolve_model(self, value: Any) -> ModelConfig:
        model_id = value if value is not None else self.config.default_model
        if not isinstance(model_id, str) or model_id not in self.config.models:
            raise APIError(404, f"model not found: {model_id}", "invalid_request_error", "model_not_found")
        return self.config.models[model_id]

    async def _ensure_model(self, model: ModelConfig) -> dict[str, Any]:
        async with self._model_locks[model.model_id]:
            response = await self.client.load_model(
                str(model.hef_path),
                model_type="llm",
                model_params={"lora_name": model.lora_name} if model.lora_name else None,
            )
            if response.get("error"):
                raise APIError(503, f"model could not be loaded: {response['error']}", "server_error", "model_load_failed")
            metadata = response.get("model_info")
            if not isinstance(metadata, dict) or not isinstance(metadata.get("prompt_template"), str):
                raise APIError(503, "device manager did not provide LLM prompt metadata", "server_error", "model_metadata_unavailable")
            if not isinstance(metadata.get("context_capacity"), int):
                raise APIError(503, "device manager did not provide LLM context capacity", "server_error", "model_metadata_unavailable")
            self._model_metadata[model.model_id] = metadata
            return metadata

    def _render_prompt(
        self,
        template: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str:
        kwargs: dict[str, Any] = {"messages": messages, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return self._jinja.from_string(template).render(**kwargs)
        except Exception as exc:
            raise APIError(400, f"could not render the model prompt template: {exc}", "invalid_request_error") from exc

    def _prepare_request(self, payload: Any) -> tuple[ModelConfig, list[dict[str, Any]], list[dict[str, Any]], str, int, float, bool]:
        if not isinstance(payload, dict):
            raise APIError(400, "request body must be a JSON object", "invalid_request_error")
        supported = {
            "model", "messages", "stream", "max_tokens", "max_completion_tokens",
            "temperature", "tools", "tool_choice",
        }
        unknown = set(payload) - supported
        if unknown:
            raise APIError(400, f"unsupported request fields: {', '.join(sorted(unknown))}", "invalid_request_error")

        model = self.resolve_model(payload.get("model"))
        messages = _normalize_messages(payload.get("messages"))
        tools = _normalize_tools(payload.get("tools"))
        if tools and not model.supports_tools:
            raise APIError(400, f"model {model.model_id} does not support tool calling", "invalid_request_error", "tools_not_supported")
        tool_choice = payload.get("tool_choice", "auto")
        if not isinstance(tool_choice, str) or tool_choice not in {"auto", "none"}:
            raise APIError(400, "only tool_choice 'auto' and 'none' are supported", "invalid_request_error")
        if tool_choice == "none":
            tools = []

        raw_max = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if "max_tokens" in payload and "max_completion_tokens" in payload:
            raise APIError(400, "use either max_tokens or max_completion_tokens, not both", "invalid_request_error")
        max_tokens = model.generation["max_tokens"] if raw_max is None else raw_max
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
            raise APIError(400, "max_tokens must be an integer from 1 to 2048", "invalid_request_error")
        temperature = payload.get("temperature", model.generation.get("temperature", 0.0))
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
            raise APIError(400, "temperature must be a number from 0 to 2", "invalid_request_error")
        do_sample = model.generation.get("do_sample", float(temperature) > 0)
        if "temperature" in payload:
            do_sample = float(temperature) > 0
        if not isinstance(payload.get("stream", False), bool):
            raise APIError(400, "stream must be a boolean", "invalid_request_error")

        metadata = self._model_metadata.get(model.model_id)
        if metadata is None:
            raise APIError(503, "model metadata is unavailable", "server_error", "model_metadata_unavailable")
        prompt = self._render_prompt(metadata["prompt_template"], messages, tools)
        prompt_tokens = _estimate_tokens(prompt)
        context_capacity = int(metadata["context_capacity"])
        if prompt_tokens + max_tokens > context_capacity:
            raise APIError(
                400,
                f"prompt and completion exceed model context capacity ({prompt_tokens} + {max_tokens} > {context_capacity})",
                "invalid_request_error",
                "context_length_exceeded",
            )
        return model, messages, tools, prompt, max_tokens, float(temperature), bool(do_sample)

    async def chat(
        self, payload: Any, http_request: web.Request | None = None
    ) -> web.StreamResponse | web.Response:
        if not isinstance(payload, dict):
            raise APIError(400, "request body must be a JSON object", "invalid_request_error")
        model = self.resolve_model(payload.get("model"))
        messages = _normalize_messages(payload.get("messages"))
        tools = _normalize_tools(payload.get("tools"))
        if tools and not model.supports_tools:
            raise APIError(400, f"model {model.model_id} does not support tool calling", "invalid_request_error", "tools_not_supported")
        try:
            await self._ensure_model(model)
        except (DeviceManagerError, FileNotFoundError, OSError, asyncio.TimeoutError, RuntimeError) as exc:
            raise _error_from_manager(exc) from exc
        # Resolve all request values after loading metadata needed by the HEF template.
        try:
            prepared_model, messages, tools, prompt, max_tokens, temperature, do_sample = self._prepare_request(payload)
        except APIError as exc:
            if exc.code == "model_metadata_unavailable":
                raise APIError(503, "model metadata is unavailable", "server_error", "model_metadata_unavailable") from exc
            raise
        model_id = prepared_model.model_id
        model_params = {"lora_name": prepared_model.lora_name} if prepared_model.lora_name else None
        input_data = {
            "prompt": prompt,
            "max_generated_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }
        tools_requested = bool(tools)
        declared_names = {
            tool["function"]["name"] for tool in tools if isinstance(tool.get("function"), dict)
        }
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        if payload.get("stream", False):
            return await self._stream_chat(
                model,
                model_params,
                input_data,
                declared_names,
                tools_requested,
                completion_id,
                created,
                http_request,
            )

        try:
            manager_response = await self.client.infer(
                str(model.hef_path),
                input_data,
                model_type="llm",
                model_params=model_params,
                model_id=model_id,
            )
        except (DeviceManagerError, OSError, asyncio.TimeoutError, RuntimeError) as exc:
            raise _error_from_manager(exc) from exc
        result = manager_response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise APIError(503, "device manager returned an invalid LLM result", "server_error", "invalid_manager_response")
        text = result["text"]
        tool_calls = _parse_tool_calls(text, declared_names) if tools_requested else None
        content = None if tool_calls else text
        finish_reason = "tool_calls" if tool_calls else "stop"
        if tool_calls is None and tools_requested:
            content = text
        self._last_context_used[model_id] = 0
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(text) if text else 0
        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            **({"tool_calls": tool_calls} if tool_calls else {}),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    async def _stream_chat(
        self,
        model: ModelConfig,
        model_params: dict[str, Any] | None,
        input_data: dict[str, Any],
        declared_names: set[str],
        tools_requested: bool,
        completion_id: str,
        created: int,
        http_request: web.Request | None,
    ) -> web.StreamResponse:
        if http_request is None:
            raise APIError(500, "streaming requires an HTTP request context", "server_error")
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        await response.prepare(http_request)

        async def send_data(value: Any) -> None:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            await response.write(f"data: {encoded}\n\n".encode("utf-8"))

        async def send_done() -> None:
            await response.write(b"data: [DONE]\n\n")

        def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model.model_id,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }

        await send_data(chunk({"role": "assistant"}))
        output_parts: list[str] = []
        finish_reason = "stop"
        manager_stream = self.client.infer_stream(
            str(model.hef_path),
            input_data,
            model_type="llm",
            model_params=model_params,
            model_id=model.model_id,
        )
        try:
            async for event in manager_stream:
                if event.get("event") == "token":
                    token = event.get("token")
                    if not isinstance(token, str):
                        continue
                    output_parts.append(token)
                    if not tools_requested:
                        await send_data(chunk({"content": token}))
                elif event.get("event") == "final":
                    break

            output_text = "".join(output_parts)
            tool_calls = _parse_tool_calls(output_text, declared_names) if tools_requested else None
            if tool_calls:
                finish_reason = "tool_calls"
                for index, tool_call in enumerate(tool_calls):
                    await send_data(
                        chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": tool_call["function"],
                                    }
                                ]
                            }
                        )
                    )
            elif tools_requested and output_text:
                await send_data(chunk({"content": output_text}))
            await send_data(chunk({}, finish_reason))
            await send_done()
            self._last_context_used[model.model_id] = 0
            await response.write_eof()
        except (DeviceManagerError, OSError, asyncio.TimeoutError, RuntimeError) as exc:
            api_error = _error_from_manager(exc)
            try:
                await send_data({"error": {"message": api_error.message, "type": api_error.error_type, "code": api_error.code}})
                await send_done()
                await response.write_eof()
            except (ConnectionResetError, OSError):
                logger.debug("Client disconnected while reporting streamed inference failure")
        finally:
            self._last_context_used[model.model_id] = 0
            await manager_stream.aclose()
        return response

    async def manager_status(self) -> dict[str, Any] | None:
        if self.status_provider is not None:
            try:
                return await self.status_provider()
            except Exception:
                logger.debug("Injected manager status provider failed", exc_info=True)
                return None
        if self._manager_session is None:
            timeout = aiohttp.ClientTimeout(total=0.75)
            self._manager_session = aiohttp.ClientSession(timeout=timeout)
        try:
            async with self._manager_session.get(self.config.manager_status_url) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                return data if isinstance(data, dict) else None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None

    async def health(self) -> web.Response:
        status = await self.manager_status()
        ready = isinstance(status, dict) and status.get("status") == "ok"
        loaded_paths: set[str] = set()
        loaded_specs: set[tuple[str, str | None]] = set()
        if status:
            networks = status.get("networks", {})
            for model in networks.get("networks", []) if isinstance(networks, dict) else []:
                if model.get("model_type") == "llm" and isinstance(model.get("model_path"), str):
                    loaded_paths.add(model["model_path"])
                    model_params = model.get("model_params") or {}
                    loaded_specs.add((model["model_path"], model_params.get("lora_name")))
        models = []
        for model_id, model in self.config.models.items():
            metadata = self._model_metadata.get(model_id, {})
            models.append(
                {
                    "id": model_id,
                    "loaded": (
                        (str(model.hef_path), model.lora_name) in loaded_specs
                        if loaded_specs
                        else str(model.hef_path) in loaded_paths
                    ),
                    "context_used": self._last_context_used.get(model_id, 0),
                    "context_max": metadata.get("context_capacity"),
                }
            )
        body = {
            "status": "ok" if ready else "unavailable",
            "model_count": len(models),
            "models": models,
            "active_inference": status.get("active_inference") if status else None,
            "queue_depth": status.get("queue_depth") if status else None,
        }
        return web.json_response(body, status=200 if ready else 503)

    async def list_models(self) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "hailo",
                    }
                    for model_id in self.config.models
                ],
            }
        )

    async def clear_context(self, payload: Any) -> web.Response:
        if payload is not None and not isinstance(payload, dict):
            raise APIError(400, "request body must be a JSON object", "invalid_request_error")
        model = self.resolve_model(payload.get("model") if payload else None)
        try:
            await self.client.clear_context(
                str(model.hef_path),
                model_type="llm",
                model_params={"lora_name": model.lora_name} if model.lora_name else None,
            )
        except (DeviceManagerError, OSError, asyncio.TimeoutError, RuntimeError) as exc:
            raise _error_from_manager(exc) from exc
        self._last_context_used[model.model_id] = 0
        return web.Response(status=204)


SERVICE_APP_KEY = web.AppKey("service", LLMApiService)


async def _read_json_request(request: web.Request, *, optional: bool = False) -> Any:
    if optional and not request.can_read_body:
        return None
    try:
        return await request.json()
    except (json.JSONDecodeError, aiohttp.ContentTypeError, UnicodeDecodeError) as exc:
        raise APIError(400, "request body must be valid JSON", "invalid_request_error") from exc


async def _chat_handler(request: web.Request) -> web.StreamResponse:
    service = request.app[SERVICE_APP_KEY]
    try:
        payload = await _read_json_request(request)
        return await service.chat(payload, request)
    except APIError as exc:
        return exc.as_response()


async def _models_handler(request: web.Request) -> web.Response:
    service = request.app[SERVICE_APP_KEY]
    return await service.list_models()


async def _health_handler(request: web.Request) -> web.Response:
    service = request.app[SERVICE_APP_KEY]
    return await service.health()


async def _clear_context_handler(request: web.Request) -> web.Response:
    service = request.app[SERVICE_APP_KEY]
    try:
        payload = await _read_json_request(request, optional=True)
        return await service.clear_context(payload)
    except APIError as exc:
        return exc.as_response()


def create_app(config: ServiceConfig, service: LLMApiService | None = None) -> web.Application:
    app = web.Application()
    app[SERVICE_APP_KEY] = service or LLMApiService(config)
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/v1/models", _models_handler)
    app.router.add_post("/v1/chat/completions", _chat_handler)
    app.router.add_post("/v1/context/clear", _clear_context_handler)

    async def close_service(application: web.Application) -> None:
        await application[SERVICE_APP_KEY].close()

    app.on_cleanup.append(close_service)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_json(CONFIG_JSON)
    web.run_app(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
