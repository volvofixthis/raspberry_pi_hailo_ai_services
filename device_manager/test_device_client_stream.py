import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path

from device_manager.device_client import HailoDeviceClient


async def read_frame(reader):
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    return json.loads(await reader.readexactly(length))


async def write_frame(writer, payload):
    body = json.dumps(payload).encode("utf-8")
    writer.write(struct.pack(">I", len(body)) + body)
    await writer.drain()


class DeviceClientStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_events_preserve_order_and_nonstream_request_still_works(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = str(Path(temp_dir) / "device.sock")
            seen = []

            async def handle(reader, writer):
                stream_request = await read_frame(reader)
                seen.append(stream_request)
                request_id = stream_request["request_id"]
                for event in (
                    {"event": "token", "token": "one"},
                    {"event": "token", "token": "two"},
                    {"event": "final", "status": "ok"},
                ):
                    await write_frame(writer, {**event, "request_id": request_id})

                ping = await read_frame(reader)
                seen.append(ping)
                await write_frame(
                    writer,
                    {"request_id": ping["request_id"], "status": "ok"},
                )
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle, path=socket_path)
            client = HailoDeviceClient(socket_path=socket_path, timeout=1.0)
            try:
                events = [
                    event
                    async for event in client.infer_stream(
                        "/models/qwen.hef",
                        {"prompt": "hello"},
                        model_id="qwen-public",
                    )
                ]
                ping = await client.ping()
                self.assertEqual([event.get("token") for event in events[:2]], ["one", "two"])
                self.assertEqual(events[-1]["event"], "final")
                self.assertEqual(seen[0]["action"], "infer_stream")
                self.assertEqual(seen[0]["model_id"], "qwen-public")
                self.assertEqual(ping["status"], "ok")
            finally:
                await client.disconnect()
                server.close()
                await server.wait_closed()

    async def test_stream_error_is_reported_to_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = str(Path(temp_dir) / "device.sock")

            async def handle(reader, writer):
                request = await read_frame(reader)
                await write_frame(
                    writer,
                    {
                        "event": "error",
                        "error": "Hailo runtime unavailable",
                        "request_id": request["request_id"],
                    },
                )
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle, path=socket_path)
            client = HailoDeviceClient(socket_path=socket_path, timeout=1.0)
            try:
                with self.assertRaisesRegex(RuntimeError, "Hailo runtime unavailable"):
                    async for _ in client.infer_stream("/models/qwen.hef", {"prompt": "hello"}):
                        pass
            finally:
                await client.disconnect()
                server.close()
                await server.wait_closed()

    async def test_stream_timeout_disconnects_stale_socket(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = str(Path(temp_dir) / "device.sock")
            accepted = asyncio.Event()

            async def handle(reader, writer):
                await read_frame(reader)
                accepted.set()
                await asyncio.sleep(0.2)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle, path=socket_path)
            client = HailoDeviceClient(socket_path=socket_path, timeout=0.05)
            try:
                with self.assertRaises(asyncio.TimeoutError):
                    async for _ in client.infer_stream("/models/qwen.hef", {"prompt": "hello"}):
                        pass
                self.assertTrue(accepted.is_set())
                self.assertIsNone(client.writer)
            finally:
                await client.disconnect()
                server.close()
                await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
