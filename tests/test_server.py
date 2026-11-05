"""HTTP and operation lifecycle checks. No model imports or model execution."""
import http.client
import json
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from qwen_one_two.server import Access, App, Conflict, Handler


class AccessTests(unittest.TestCase):
    def test_exact_origin_and_duplicate_headers(self):
        access = Access(3211, "https://operation-room.dimensionlab.net")
        self.assertTrue(access.allows(["operation-room.dimensionlab.net"],
                                      ["https://operation-room.dimensionlab.net"]))
        self.assertTrue(access.allows(["operation-room.dimensionlab.net:443"], []))
        self.assertTrue(access.allows(["127.0.0.1:3211"], ["http://127.0.0.1:3211"]))
        for hosts, origins in [
            (["evil.example"], []),
            (["operation-room.dimensionlab.net"], ["https://evil.example"]),
            (["operation-room.dimensionlab.net"], ["http://operation-room.dimensionlab.net"]),
            (["127.0.0.1:3211"], ["https://operation-room.dimensionlab.net"]),
            (["127.0.0.1:3211", "evil.example"], []),
            (["127.0.0.1:3211"], ["http://127.0.0.1:3211"] * 2),
        ]:
            with self.subTest(hosts=hosts, origins=origins):
                self.assertFalse(access.allows(hosts, origins))

    def test_invalid_public_origins(self):
        for origin in ("http://example.com", "https://example.com/", "https://example.com/path",
                       "https://user@example.com", "https://example.com?q=1"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                Access(3211, origin)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = App(self.root / "data", self.root / "cache", "cuda:0")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.app = self.app
        self.server.access = Access(self.server.server_port)
        self.server.assets = self.root
        (self.root / "index.html").write_text("<title>test interface</title>")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_assets_state_and_http_origin_rejection(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"test interface", body)
        self.assertEqual(headers["Cache-Control"], "no-store")
        status, _, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["loaded"])
        self.assertEqual(self.request("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/decide", body="{}", headers={
            "Content-Type": "application/json", "Origin": "https://evil.example"})[0], 403)

    def test_invalid_requests_never_start_model_work(self):
        self.assertEqual(self.request("POST", "/api/decide", body="{}",
                                      headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/decide", body="[",
                                      headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("GET", "/api/runs/../../other")[0], 400)
        self.assertIsNone(self.app.runtime)
        self.assertFalse(self.app.state()["busy"])

    def test_separate_decision_endpoint_is_retired_without_model_work(self):
        status, _, body = self.request("POST", "/api/decide", body="{}",
                                       headers={"Content-Type": "application/json"})
        self.assertEqual(status, 410)
        self.assertIn("inside the generation run", json.loads(body)["error"])
        self.assertIsNone(self.app.runtime)
        self.assertFalse(self.app.state()["busy"])

    def test_operation_lock_and_targeted_stop_remain_responsive(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def job():
            entered.set()
            release.wait(timeout=3)
            finished.set()

        self.app.execute("generating", "a" * 32, job, background=True)
        try:
            self.assertTrue(entered.wait(timeout=3))
            with self.assertRaises(Conflict):
                self.app.execute("deciding", "b" * 32, lambda: None)
            with self.assertRaises(Conflict):
                self.app.stop({"operation_id": "b" * 32})
            self.assertFalse((self.app.data / ".cancel").exists())
            self.assertTrue(json.loads(self.request("GET", "/api/state")[2])["busy"])
            status, _, _ = self.request("POST", "/api/stop", body=json.dumps({"operation_id": "a" * 32}),
                                         headers={"Content-Type": "application/json"})
            self.assertEqual(status, 200)
            self.assertTrue((self.app.data / ".cancel").exists())
        finally:
            release.set()
            self.assertTrue(finished.wait(timeout=3))

    def test_disconnected_generation_releases_operation_slot(self):
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()

        def generate(_args):
            entered.set()
            release.wait(timeout=3)
            completed.set()
            return {"answer": "normal answer"}

        self.app.runtime = SimpleNamespace(thinking=SimpleNamespace(generate=generate), gpu_name="scripted", active_adapter="router")
        stream = socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3)
        body = json.dumps({"text": "prompt", "run_id": "c" * 32})
        payload = (f"POST /api/generate HTTP/1.1\r\nHost: 127.0.0.1:{self.server.server_port}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}")
        stream.sendall(payload.encode())
        try:
            self.assertTrue(entered.wait(timeout=3))
            stream.shutdown(socket.SHUT_RDWR)
            stream.close()
        finally:
            release.set()
        self.assertTrue(completed.wait(timeout=3))
        for _ in range(100):
            if not self.app.state()["busy"]:
                break
            time.sleep(.01)
        self.assertFalse(self.app.state()["busy"])
        self.assertIsNone(self.app.state()["active_id"])


if __name__ == "__main__":
    unittest.main()
