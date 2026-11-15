"""Configuration loading and validation for hailo-llm."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of silently replacing them."""


def _construct_unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    hef_path: Path
    lora_name: str | None
    supports_tools: bool
    generation: dict[str, Any]


@dataclass(frozen=True)
class ServiceConfig:
    host: str
    port: int
    default_model: str
    models: dict[str, ModelConfig]
    generation: dict[str, Any]
    manager_status_url: str
    manager_timeout_s: float


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_generation(value: Any, name: str) -> dict[str, Any]:
    generation = _require_mapping(value, name)
    result = dict(generation)
    max_tokens = result.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
            raise ValueError(f"{name}.max_tokens must be an integer from 1 to 2048")
    temperature = result.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(f"{name}.temperature must be a number from 0 to 2")
        if not 0 <= float(temperature) <= 2:
            raise ValueError(f"{name}.temperature must be a number from 0 to 2")
        result["temperature"] = float(temperature)
    do_sample = result.get("do_sample")
    if do_sample is not None and not isinstance(do_sample, bool):
        raise ValueError(f"{name}.do_sample must be a boolean")
    unsupported = set(result) - {"max_tokens", "temperature", "do_sample"}
    if unsupported:
        raise ValueError(f"{name} contains unsupported settings: {', '.join(sorted(unsupported))}")
    return result


def validate_config(raw: Any) -> ServiceConfig:
    config = _require_mapping(raw, "configuration")
    server = _require_mapping(config.get("server"), "server")
    host = server.get("host", "0.0.0.0")
    port = server.get("port", 11441)
    if not isinstance(host, str) or not host.strip():
        raise ValueError("server.host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("server.port must be an integer from 1 to 65535")

    raw_models = _require_mapping(config.get("models"), "models")
    if not raw_models:
        raise ValueError("models must contain at least one model")

    global_generation = _validate_generation(config.get("generation", {}), "generation")
    env_max_tokens = os.environ.get("NPU_MAX_GENERATED_TOKENS")
    if env_max_tokens is not None:
        try:
            env_value = int(env_max_tokens)
        except ValueError as exc:
            raise ValueError("NPU_MAX_GENERATED_TOKENS must be an integer from 1 to 2048") from exc
        if not 1 <= env_value <= 2048:
            raise ValueError("NPU_MAX_GENERATED_TOKENS must be an integer from 1 to 2048")
        global_generation["max_tokens"] = env_value
    global_generation.setdefault("max_tokens", 512)
    global_generation.setdefault("temperature", 0.0)
    global_generation.setdefault("do_sample", global_generation["temperature"] > 0)

    models: dict[str, ModelConfig] = {}
    for model_id, raw_model in raw_models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("models entries must use non-empty string IDs")
        model = _require_mapping(raw_model, f"models.{model_id}")
        hef_path_text = model.get("hef_path")
        if not isinstance(hef_path_text, str) or not hef_path_text.strip():
            raise ValueError(f"models.{model_id}.hef_path must be a non-empty absolute .hef path")
        hef_path = Path(hef_path_text)
        if not hef_path.is_absolute() or hef_path.suffix.lower() != ".hef":
            raise ValueError(f"models.{model_id}.hef_path must be an absolute .hef path")
        lora_name = model.get("lora_name")
        if lora_name is not None and not isinstance(lora_name, str):
            raise ValueError(f"models.{model_id}.lora_name must be a string or null")
        supports_tools = model.get("supports_tools", False)
        if not isinstance(supports_tools, bool):
            raise ValueError(f"models.{model_id}.supports_tools must be a boolean")

        model_generation = _validate_generation(
            model.get("generation", {}), f"models.{model_id}.generation"
        )
        generation = dict(global_generation)
        generation.update(model_generation)
        if "temperature" in model_generation and "do_sample" not in model_generation:
            generation["do_sample"] = generation["temperature"] > 0
        models[model_id] = ModelConfig(
            model_id=model_id,
            hef_path=hef_path,
            lora_name=lora_name or None,
            supports_tools=supports_tools,
            generation=generation,
        )

    default_model = config.get("default_model")
    if not isinstance(default_model, str) or default_model not in models:
        raise ValueError("default_model must name an entry in models")

    manager = _require_mapping(config.get("device_manager", {}), "device_manager")
    status_url = manager.get("status_url", "http://127.0.0.1:5099/v1/device/status")
    if not isinstance(status_url, str) or not status_url.startswith(("http://", "https://")):
        raise ValueError("device_manager.status_url must be an HTTP(S) URL")
    timeout = manager.get("timeout_s", 120.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("device_manager.timeout_s must be a positive number")

    return ServiceConfig(
        host=host,
        port=port,
        default_model=default_model,
        models=models,
        generation=global_generation,
        manager_status_url=status_url,
        manager_timeout_s=float(timeout),
    )


def load_yaml(path: Path) -> ServiceConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=UniqueKeyLoader)
    return validate_config(raw)


def load_json(path: Path) -> ServiceConfig:
    with path.open("r", encoding="utf-8") as handle:
        return validate_config(json.load(handle))
