import importlib.util
import asyncio
import json
import sys
import struct
import tempfile
import threading
import types
import unittest
import numpy  # noqa: F401 - keep the manager's runtime dependency loaded outside the mocked import context
from urllib.request import urlopen
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


class FakeLLM:
    instances = []

    def __init__(self, vdevice, model_path, **kwargs):
        self.vdevice = vdevice
        self.model_path = model_path
        self.kwargs = kwargs
        self.clear_count = 0
        self.released = False
        self.calls = []
        self.instances.append(self)

    def prompt_template(self):
        return "{{ messages }}"

    def max_context_capacity(self):
        return 2048

    def get_context_usage_size(self):
        return 17

    def clear_context(self):
        self.clear_count += 1

    def generate_all(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "generated text"

    @contextmanager
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        yield iter(["hello", "<|im_end|>", "world"])

    def release(self):
        self.released = True


class FakeGenAIModel:
    pass


class BlockingFakeLLM(FakeLLM):
    def __init__(self, vdevice, model_path, **kwargs):
        super().__init__(vdevice, model_path, **kwargs)
        self.started = threading.Event()
        self.release_generation = threading.Event()

    @contextmanager
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))

        def tokens():
            yield "first"
            self.started.set()
            if not self.release_generation.wait(3):
                raise TimeoutError("test generation release timed out")
            yield "second"

        yield tokens()


class FailingFakeLLM(FakeLLM):
    @contextmanager
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))

        def tokens():
            yield "partial"
            raise RuntimeError("generation failed")

        yield tokens()


def load_manager_module():
    if "hailo_device_manager_under_test" in sys.modules:
        return sys.modules["hailo_device_manager_under_test"]

    platform = types.ModuleType("hailo_platform")
    platform.__path__ = []
    platform.FormatType = types.SimpleNamespace(
        UINT8="uint8", UINT16="uint16", FLOAT32="float32"
    )
    platform.Device = type("Device", (), {})
    platform.VDevice = type("VDevice", (), {})

    genai = types.ModuleType("hailo_platform.genai")
    genai.VLM = FakeGenAIModel
    genai.LLM = FakeLLM
    genai.Speech2Text = FakeGenAIModel
    genai.Speech2TextTask = types.SimpleNamespace(TRANSCRIBE="transcribe", TRANSLATE="translate")
    platform.genai = genai

    source = Path(__file__).with_name("hailo_device_manager.py")
    spec = importlib.util.spec_from_file_location("hailo_device_manager_under_test", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"hailo_platform": platform, "hailo_platform.genai": genai}):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module


class LlmHandlerTests(unittest.TestCase):
    def setUp(self):
        FakeLLM.instances.clear()
        self.manager_module = load_manager_module()
        self.handler = self.manager_module.LlmHandler()

    def test_handler_loads_with_optional_lora_and_exposes_metadata(self):
        vdevice = object()
        llm = self.handler.load(vdevice, "/models/qwen.hef", {"lora_name": "adapter"})

        self.assertEqual(llm.vdevice, vdevice)
        self.assertEqual(llm.kwargs, {"lora_name": "adapter"})
        self.assertEqual(
            self.handler.metadata(llm),
            {"prompt_template": "{{ messages }}", "context_capacity": 2048},
        )

    def test_inference_clears_context_and_returns_usage_metadata(self):
        llm = FakeLLM(object(), "/models/qwen.hef")

        result = self.handler.infer(
            llm,
            {
                "prompt": "rendered prompt",
                "max_generated_tokens": 128,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.85,
                "top_k": 40,
            },
        )

        self.assertEqual(result, {"text": "generated text", "context_used": 17, "context_capacity": 2048})
        self.assertEqual(llm.clear_count, 2)
        self.assertEqual(
            llm.calls,
            [(
                "rendered prompt",
                {
                    "max_generated_tokens": 128,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "top_k": 40,
                },
            )],
        )

    def test_zero_temperature_disables_sampling_without_passing_zero_temperature(self):
        llm = FakeLLM(object(), "/models/qwen.hef")
        self.handler.infer(
            llm,
            {"prompt": "prompt", "temperature": 0.0, "do_sample": False},
        )
        self.assertEqual(llm.calls[0][1], {"do_sample": False})

    def test_stream_filters_stop_tokens_and_clears_context(self):
        llm = FakeLLM(object(), "/models/qwen.hef")

        self.assertEqual(list(self.handler.stream(llm, {"prompt": "prompt"})), ["hello", "world"])
        self.assertEqual(llm.clear_count, 2)

    def test_manager_keeps_distinct_llm_artifacts_and_unloads_selected_model(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager.vdevice = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.hef"
            second_path = Path(temp_dir) / "second.hef"
            first_path.touch()
            second_path.touch()

            first = manager._load_model({"model_type": "llm", "model_path": str(first_path)})
            second = manager._load_model({"model_type": "llm", "model_path": str(second_path)})

            self.assertEqual(first["model_info"]["context_capacity"], 2048)
            self.assertEqual(second["model_info"]["prompt_template"], "{{ messages }}")
            self.assertEqual(len(manager.loaded_models), 2)

            manager._unload_model({"model_type": "llm", "model_path": str(first_path)})

            self.assertEqual(len(manager.loaded_models), 1)
            self.assertTrue(FakeLLM.instances[-2].released)
            self.assertFalse(FakeLLM.instances[-1].released)

    def test_same_artifact_with_distinct_loras_has_separate_runtime_entries(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager.vdevice = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "shared.hef"
            model_path.touch()
            manager._load_model(
                {"model_type": "llm", "model_path": str(model_path), "model_params": {"lora_name": "first"}}
            )
            manager._load_model(
                {"model_type": "llm", "model_path": str(model_path), "model_params": {"lora_name": "second"}}
            )

            self.assertEqual(len(manager.loaded_models), 2)
            self.assertEqual(FakeLLM.instances[-2].kwargs, {"lora_name": "first"})
            self.assertEqual(FakeLLM.instances[-1].kwargs, {"lora_name": "second"})

    def test_llm_load_evicts_cached_whisper_after_kv_cache_conflict(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager.vdevice = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            whisper_path = Path(temp_dir) / "Whisper-Base.hef"
            llm_path = Path(temp_dir) / "Qwen.hef"
            whisper_path.touch()
            llm_path.touch()
            whisper = FakeLLM(manager.vdevice, str(whisper_path))
            whisper_key = manager._model_key("whisper", str(whisper_path))
            manager.loaded_models[whisper_key] = self.manager_module.ModelEntry(
                model_type="whisper",
                model_path=str(whisper_path),
                model=whisper,
                loaded_at=1.0,
                last_used=1.0,
            )

            attempts = 0
            original_llm = self.manager_module.LLM

            def load_llm(vdevice, model_path, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("HAILO_INVALID_OPERATION(6): KV-Cache unavailable")
                return FakeLLM(vdevice, model_path, **kwargs)

            self.manager_module.LLM = load_llm
            try:
                response = manager._load_model(
                    {"model_type": "llm", "model_path": str(llm_path)}
                )
            finally:
                self.manager_module.LLM = original_llm

            self.assertEqual(response["status"], "ok")
            self.assertEqual(attempts, 2)
            self.assertTrue(whisper.released)
            self.assertNotIn(whisper_key, manager.loaded_models)
            self.assertIn(manager._model_key("llm", str(llm_path)), manager.loaded_models)

    def test_internal_failure_after_kv_cache_conflict_resets_runtime_then_retries(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager.vdevice = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            whisper_path = Path(temp_dir) / "Whisper-Base.hef"
            llm_path = Path(temp_dir) / "Qwen.hef"
            whisper_path.touch()
            llm_path.touch()
            whisper = FakeLLM(manager.vdevice, str(whisper_path))
            manager.loaded_models[manager._model_key("whisper", str(whisper_path))] = self.manager_module.ModelEntry(
                model_type="whisper",
                model_path=str(whisper_path),
                model=whisper,
                loaded_at=1.0,
                last_used=1.0,
            )

            attempts = 0
            reset_count = 0
            original_llm = self.manager_module.LLM

            def load_llm(vdevice, model_path, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("HAILO_INVALID_OPERATION(6): KV-Cache unavailable")
                if attempts == 2:
                    raise RuntimeError("HAILO_INTERNAL_FAILURE(8)")
                return FakeLLM(vdevice, model_path, **kwargs)

            def reset_runtime():
                nonlocal reset_count
                reset_count += 1
                manager.loaded_models.clear()

            self.manager_module.LLM = load_llm
            manager._reset_runtime_context = reset_runtime
            try:
                response = manager._load_model(
                    {"model_type": "llm", "model_path": str(llm_path)}
                )
            finally:
                self.manager_module.LLM = original_llm

            self.assertEqual(response["status"], "ok")
            self.assertEqual(attempts, 3)
            self.assertEqual(reset_count, 1)
            self.assertEqual(len(manager.loaded_models), 1)
            self.assertIn(manager._model_key("llm", str(llm_path)), manager.loaded_models)

    def test_manager_metadata_and_context_actions(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        model_path = "/models/qwen.hef"
        llm = FakeLLM(object(), model_path)
        key = manager._model_key("llm", model_path)
        manager.loaded_models[key] = self.manager_module.ModelEntry(
            model_type="llm",
            model_path=model_path,
            model=llm,
            loaded_at=1.0,
            last_used=1.0,
        )

        info = manager._handle_request_sync(
            {"action": "model_info", "model_type": "llm", "model_path": model_path}
        )
        cleared = manager._handle_request_sync(
            {"action": "clear_context", "model_type": "llm", "model_path": model_path}
        )
        not_loaded = manager._handle_request_sync(
            {"action": "clear_context", "model_type": "llm", "model_path": "/models/not-loaded.hef"}
        )

        self.assertEqual(info["model_info"]["context_used"], 17)
        self.assertEqual(cleared["status"], "ok")
        self.assertTrue(cleared["loaded"])
        self.assertFalse(not_loaded["loaded"])
        self.assertEqual(llm.clear_count, 1)

    def test_manager_keeps_runtime_contexts_isolated_per_model(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager.vdevice = object()
        models = {}
        for model_id in ("first", "second"):
            model_path = f"/models/{model_id}.hef"
            llm = FakeLLM(manager.vdevice, model_path)
            models[model_id] = llm
            manager.loaded_models[manager._model_key("llm", model_path)] = self.manager_module.ModelEntry(
                model_type="llm",
                model_path=model_path,
                model=llm,
                loaded_at=1.0,
                last_used=1.0,
            )

        first = manager._infer(
            {
                "model_type": "llm",
                "model_path": "/models/first.hef",
                "model_id": "first",
                "input_data": {"prompt": "first history", "max_generated_tokens": 64},
            }
        )
        second = manager._infer(
            {
                "model_type": "llm",
                "model_path": "/models/second.hef",
                "model_id": "second",
                "input_data": {"prompt": "second history", "max_generated_tokens": 64},
            }
        )

        self.assertEqual(first["result"]["text"], "generated text")
        self.assertEqual(second["result"]["text"], "generated text")
        self.assertEqual(models["first"].calls[0][0], "first history")
        self.assertEqual(models["second"].calls[0][0], "second history")
        self.assertEqual(models["first"].clear_count, 2)
        self.assertEqual(models["second"].clear_count, 2)
        self.assertIsNone(manager._active_inference_snapshot())

    def test_manager_reports_context_overflow_without_generating(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        model_path = "/models/qwen.hef"
        llm = FakeLLM(object(), model_path)
        manager.loaded_models[manager._model_key("llm", model_path)] = self.manager_module.ModelEntry(
            model_type="llm",
            model_path=model_path,
            model=llm,
            loaded_at=1.0,
            last_used=1.0,
        )

        response = manager._infer(
            {
                "model_type": "llm",
                "model_path": model_path,
                "input_data": {"prompt": "x" * 9000, "max_generated_tokens": 128},
            }
        )

        self.assertEqual(response["error"]["type"], "context_overflow")
        self.assertEqual(response["error"]["estimated_prompt_tokens"], 2250)
        self.assertEqual(response["error"]["context_max"], 2048)
        self.assertEqual(llm.calls, [])
        self.assertEqual(llm.clear_count, 2)
        self.assertIsNone(manager._active_inference_snapshot())

    def test_device_status_exposes_active_inference_and_returns_to_idle(self):
        manager = self.manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager._queue_depth = 3
        manager._begin_inference(
            {
                "request_id": "req-123",
                "model_id": "qwen-test",
                "model_type": "llm",
                "model_path": "/models/qwen.hef",
            }
        )

        active = manager._device_status_payload()
        manager._end_inference()
        idle = manager._device_status_payload()

        self.assertEqual(active["active_inference"]["request_id"], "req-123")
        self.assertEqual(active["active_inference"]["model_id"], "qwen-test")
        self.assertEqual(active["active_inference"]["model_type"], "llm")
        self.assertIsInstance(active["active_inference"]["started_at"], float)
        self.assertEqual(active["queue_depth"], 3)
        self.assertIsNone(idle["active_inference"])


class StreamingManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_holds_worker_and_reports_active_status_until_cleanup(self):
        manager_module = load_manager_module()
        manager = manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        manager.vdevice = object()
        manager.http_bind = ("127.0.0.1", 0)
        manager._start_http_server()

        model_path = "/models/streaming.hef"
        llm = BlockingFakeLLM(manager.vdevice, model_path)
        manager.loaded_models[manager._model_key("llm", model_path)] = manager_module.ModelEntry(
            model_type="llm",
            model_path=model_path,
            model=llm,
            loaded_at=1.0,
            last_used=1.0,
        )

        class FakeWriter:
            def __init__(self):
                self.frames = []

            def write(self, payload):
                self.frames.append(payload)

            async def drain(self):
                return None

        writer = FakeWriter()
        worker = asyncio.create_task(manager._worker_loop())
        request = {
            "action": "infer_stream",
            "model_type": "llm",
            "model_path": model_path,
            "model_id": "streaming-model",
            "request_id": "stream-request",
            "input_data": {"prompt": "prompt", "max_generated_tokens": 8},
        }
        stream = asyncio.create_task(manager.process_stream_request(request, writer))
        queued_ping = None

        try:
            self.assertTrue(await asyncio.to_thread(llm.started.wait, 2))
            status_url = f"http://127.0.0.1:{manager._http_server.server_address[1]}/v1/device/status"
            with await asyncio.to_thread(urlopen, status_url, timeout=2) as response:
                status = json.loads(response.read().decode("utf-8"))
            active = status["active_inference"]
            self.assertEqual(active["request_id"], "stream-request")
            self.assertEqual(active["model_id"], "streaming-model")
            queued_ping = asyncio.create_task(manager._enqueue_request({"action": "ping"}))
            await asyncio.sleep(0.01)
            self.assertFalse(queued_ping.done())

            llm.release_generation.set()
            await asyncio.wait_for(stream, timeout=3)
            ping_result = await asyncio.wait_for(queued_ping, timeout=3)
        finally:
            llm.release_generation.set()
            if not stream.done():
                await stream
            await manager._request_queue.put(None)
            await asyncio.wait_for(worker, timeout=3)
            manager._stop_http_server()

        events = []
        for frame in writer.frames:
            length = struct.unpack(">I", frame[:4])[0]
            events.append(json.loads(frame[4 : 4 + length]))

        self.assertEqual([event["event"] for event in events], ["token", "token", "final"])
        self.assertEqual([event["token"] for event in events[:2]], ["first", "second"])
        self.assertTrue(all(event["request_id"] == "stream-request" for event in events))
        self.assertIsNone(manager._device_status_payload()["active_inference"])
        self.assertEqual(llm.clear_count, 2)
        self.assertEqual(ping_result["status"], "ok")

    async def test_stream_error_clears_active_status_and_sends_error_event(self):
        manager_module = load_manager_module()
        manager = manager_module.HailoDeviceManager("/tmp/test-device.sock")
        manager.device = types.SimpleNamespace(device_id="fake-device")
        model_path = "/models/failing.hef"
        llm = FailingFakeLLM(object(), model_path)
        manager.loaded_models[manager._model_key("llm", model_path)] = manager_module.ModelEntry(
            model_type="llm",
            model_path=model_path,
            model=llm,
            loaded_at=1.0,
            last_used=1.0,
        )
        queue = asyncio.Queue()
        request = {
            "model_type": "llm",
            "model_path": model_path,
            "model_id": "failing-model",
            "request_id": "failing-request",
            "input_data": {"prompt": "prompt"},
        }

        result = await asyncio.to_thread(
            manager._handle_stream_request_sync,
            request,
            asyncio.get_running_loop(),
            queue,
        )
        events = [await queue.get(), await queue.get()]

        self.assertEqual(result["error"], "generation failed")
        self.assertEqual([event["event"] for event in events], ["token", "error"])
        self.assertEqual(events[-1]["error"], "generation failed")
        self.assertIsNone(manager._device_status_payload()["active_inference"])
        self.assertEqual(llm.clear_count, 2)


if __name__ == "__main__":
    unittest.main()
