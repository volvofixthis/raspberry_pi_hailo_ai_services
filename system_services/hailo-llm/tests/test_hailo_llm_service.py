import json
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from config import ServiceConfig, load_yaml, validate_config  # noqa: E402
from hailo_llm_server import LLMApiService, create_app  # noqa: E402
from render_config import render_config  # noqa: E402


def raw_config():
    return {
        "server": {"host": "127.0.0.1", "port": 11441},
        "default_model": "small",
        "generation": {"max_tokens": 32, "temperature": 0.0},
        "models": {
            "small": {
                "hef_path": "/models/small.hef",
                "supports_tools": False,
            },
            "tools": {
                "hef_path": "/models/tools.hef",
                "lora_name": "tools-adapter",
                "supports_tools": True,
                "generation": {"max_tokens": 64, "temperature": 0.2},
            },
        },
    }


class FakeManagerClient:
    def __init__(
        self,
        *,
        inference_text="response",
        fail_load=None,
        fail_infer=None,
        fail_clear=None,
        context_capacity=2048,
    ):
        self.inference_text = inference_text
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.fail_clear = fail_clear
        self.context_capacity = context_capacity
        self.loads = []
        self.inferences = []
        self.streams = []
        self.clears = []
        self.disconnected = False

    async def load_model(self, model_path, model_type="llm", model_params=None):
        self.loads.append((model_path, model_type, model_params))
        if self.fail_load:
            raise self.fail_load
        return {
            "status": "ok",
            "model_info": {
                "prompt_template": "{% for m in messages %}{{ m.role }}={{ m.content or '' }};{% endfor %}assistant=",
                "context_capacity": self.context_capacity,
            },
        }

    async def infer(self, model_path, input_data, model_type="llm", model_params=None, model_id=None):
        self.inferences.append((model_path, input_data, model_type, model_params, model_id))
        if self.fail_infer:
            raise self.fail_infer
        return {
            "status": "ok",
            "result": {
                "text": self.inference_text,
                "context_used": 14,
                "context_capacity": 2048,
            },
        }

    async def infer_stream(self, model_path, input_data, model_type="llm", model_params=None, model_id=None):
        self.streams.append((model_path, input_data, model_type, model_params, model_id))
        if self.fail_infer:
            raise self.fail_infer
        for token in self.inference_text.split(" "):
            yield {"event": "token", "token": token + " "}
        yield {"event": "final", "status": "ok", "completion_tokens": 2}

    async def clear_context(self, model_path, model_type="llm", model_params=None):
        self.clears.append((model_path, model_type, model_params))
        if self.fail_clear:
            raise self.fail_clear
        return {"status": "ok", "loaded": True}

    async def disconnect(self):
        self.disconnected = True


class LLMServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = validate_config(raw_config())
        self.manager = FakeManagerClient()
        self.status = {
            "status": "ok",
            "queue_depth": 1,
            "active_inference": {
                "request_id": "request-1",
                "model_id": "small",
                "model_type": "llm",
                "started_at": 123.0,
            },
            "networks": {
                "networks": [
                    {
                        "model_type": "llm",
                        "model_path": "/models/small.hef",
                        "model_params": {},
                    }
                ]
            },
        }
        self.service = LLMApiService(
            self.config,
            client=self.manager,
            status_provider=lambda: self._status(),
        )
        self.client = TestClient(TestServer(create_app(self.config, self.service)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def _status(self):
        return self.status

    async def test_model_discovery_and_health_include_active_manager_state(self):
        models = await self.client.get("/v1/models")
        self.assertEqual([item["id"] for item in (await models.json())["data"]], ["small", "tools"])

        health = await self.client.get("/health")
        body = await health.json()
        self.assertEqual(health.status, 200)
        self.assertTrue(body["models"][0]["loaded"])
        self.assertFalse(body["models"][1]["loaded"])
        self.assertEqual(body["active_inference"]["request_id"], "request-1")
        self.assertEqual(body["queue_depth"], 1)

    async def test_health_reports_manager_unavailable(self):
        self.status = None
        response = await self.client.get("/health")
        self.assertEqual(response.status, 503)
        self.assertEqual((await response.json())["status"], "unavailable")

    async def test_chat_uses_default_model_and_openai_response_shape(self):
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        body = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "small")
        self.assertEqual(body["choices"][0]["message"]["content"], "response")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.manager.loads[0][0], "/models/small.hef")
        self.assertEqual(self.manager.inferences[0][4], "small")

    async def test_model_selection_uses_per_model_lora_and_generation_defaults(self):
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "tools",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 9,
                "temperature": 0.8,
                "top_p": 0.85,
                "top_k": 40,
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(self.manager.loads[0][0], "/models/tools.hef")
        self.assertEqual(self.manager.loads[0][2], {"lora_name": "tools-adapter"})
        input_data = self.manager.inferences[0][1]
        self.assertEqual(input_data["max_generated_tokens"], 9)
        self.assertTrue(input_data["do_sample"])
        self.assertEqual(input_data["temperature"], 0.8)
        self.assertEqual(input_data["top_p"], 0.85)
        self.assertEqual(input_data["top_k"], 40)

    async def test_top_p_or_top_k_request_enables_sampling_by_default(self):
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "top_k": 20,
            },
        )
        self.assertEqual(response.status, 200)
        input_data = self.manager.inferences[0][1]
        self.assertTrue(input_data["do_sample"])
        self.assertEqual(input_data["top_k"], 20)

        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "top_p": 0.9,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status, 200)
        self.assertFalse(self.manager.inferences[1][1]["do_sample"])

    async def test_sampling_parameters_are_validated(self):
        for setting in (
            {"top_p": 0}, {"top_p": 1.1}, {"top_p": True}, {"top_p": None},
            {"top_k": 0}, {"top_k": 1.5}, {"top_k": None},
        ):
            with self.subTest(setting=setting):
                response = await self.client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}], **setting},
                )
                self.assertEqual(response.status, 400)

    async def test_unknown_model_and_unsupported_content_are_client_errors(self):
        unknown = await self.client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(unknown.status, 404)

        unsupported = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        )
        self.assertEqual(unsupported.status, 400)
        self.assertEqual(self.manager.loads, [])

    async def test_tool_calls_are_returned_in_openai_shape_and_not_executed(self):
        self.manager.inference_text = '{"name":"lookup","arguments":{"q":"x"}}'
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "tools",
                "messages": [{"role": "user", "content": "lookup x"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        body = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        call = body["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "lookup")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"q": "x"})

        follow_up = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "tools",
                "messages": [
                    {"role": "user", "content": "lookup x"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": call["function"],
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": call["id"], "content": "result"},
                ],
            },
        )
        self.assertEqual(follow_up.status, 200)
        self.assertIn("tool=result", self.manager.inferences[-1][1]["prompt"])

    async def test_streaming_returns_sse_deltas_and_done_sentinel(self):
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        body = await response.text()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "text/event-stream")
        self.assertIn('"delta":{"content":"response "}', body)
        self.assertIn('"finish_reason":"stop"', body)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))

    async def test_streaming_reports_manager_error_as_sse_error(self):
        self.manager.fail_infer = RuntimeError("manager unavailable")
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        )
        body = await response.text()
        self.assertEqual(response.status, 200)
        self.assertIn('"type":"server_error"', body)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))

    async def test_context_clear_defaults_to_configured_model_and_accepts_selection(self):
        default = await self.client.post("/v1/context/clear")
        selected = await self.client.post("/v1/context/clear", json={"model": "tools"})
        unknown = await self.client.post("/v1/context/clear", json={"model": "missing"})

        self.assertEqual(default.status, 204)
        self.assertEqual(selected.status, 204)
        self.assertEqual(unknown.status, 404)
        self.assertEqual(self.manager.clears[0][0], "/models/small.hef")
        self.assertEqual(self.manager.clears[1][0], "/models/tools.hef")
        self.assertEqual(self.manager.clears[1][2], {"lora_name": "tools-adapter"})

    async def test_context_clear_reports_manager_failure(self):
        self.manager.fail_clear = RuntimeError("manager unavailable")
        response = await self.client.post("/v1/context/clear")
        self.assertEqual(response.status, 503)

    async def test_context_overflow_is_rejected_before_inference(self):
        self.manager.context_capacity = 8
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "a longer prompt"}]},
        )
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "context_length_exceeded")
        self.assertEqual(self.manager.inferences, [])

    async def test_manager_load_and_inference_errors_are_structured(self):
        self.manager.fail_load = RuntimeError("manager offline")
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status, 503)
        self.manager.fail_load = None
        self.manager.fail_infer = RuntimeError("manager offline")
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status, 503)


class ConfigTests(unittest.TestCase):
    def test_generation_config_accepts_and_validates_sampling_options(self):
        configured = raw_config()
        configured["generation"].update({"top_p": 0.9, "top_k": 50})
        parsed = validate_config(configured)
        self.assertEqual(parsed.models["small"].generation["top_p"], 0.9)
        self.assertEqual(parsed.models["small"].generation["top_k"], 50)
        self.assertTrue(parsed.models["small"].generation["do_sample"])

        for setting in ({"top_p": 0}, {"top_p": 1.1}, {"top_k": 0}, {"top_k": True}):
            with self.subTest(setting=setting):
                invalid = raw_config()
                invalid["generation"].update(setting)
                with self.assertRaises(ValueError):
                    validate_config(invalid)

    def test_rejects_duplicate_model_keys_and_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "duplicate.yaml"
            yaml_path.write_text(
                "default_model: x\nmodels:\n  x:\n    hef_path: /x.hef\n  x:\n    hef_path: /y.hef\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                load_yaml(yaml_path)

        missing_default = raw_config()
        missing_default["default_model"] = "missing"
        with self.assertRaisesRegex(ValueError, "default_model"):
            validate_config(missing_default)

        invalid = raw_config()
        invalid["models"]["small"]["hef_path"] = "relative.hef"
        with self.assertRaisesRegex(ValueError, "absolute .hef path"):
            validate_config(invalid)
        invalid["models"]["small"]["hef_path"] = "/models/not-a-hef.bin"
        with self.assertRaisesRegex(ValueError, "absolute .hef path"):
            validate_config(invalid)

    def test_renderer_validates_and_writes_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "config.yaml"
            output = Path(temp_dir) / "xdg" / "hailo-llm.json"
            source.write_text(
                "server:\n  host: 127.0.0.1\n  port: 11441\n"
                "default_model: small\nmodels:\n  small:\n    hef_path: /models/small.hef\n",
                encoding="utf-8",
            )
            render_config(source, output)
            config = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(config["default_model"], "small")
            self.assertEqual(config["models"]["small"]["hef_path"], "/models/small.hef")


if __name__ == "__main__":
    unittest.main()
