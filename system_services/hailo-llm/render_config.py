#!/usr/bin/env python3
"""Validate operator YAML and render the runtime JSON configuration."""

import argparse
import json
import sys
from pathlib import Path

from config import load_yaml


def render_config(input_path: Path, output_path: Path) -> None:
    config = load_yaml(input_path)
    data = {
        "server": {"host": config.host, "port": config.port},
        "default_model": config.default_model,
        "generation": config.generation,
        "models": {
            model_id: {
                "hef_path": str(model.hef_path),
                "lora_name": model.lora_name,
                "supports_tools": model.supports_tools,
                "generation": model.generation,
            }
            for model_id, model in config.models.items()
        },
        "device_manager": {
            "status_url": config.manager_status_url,
            "timeout_s": config.manager_timeout_s,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        render_config(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    print(f"Configuration rendered to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
