"""Small loopback HTTP server. HTTP threads remain responsive during GPU work.

Exactly one model operation owns the slot; cancellation and progress use separate
HTTP threads. No user-provided code, shell commands or external model calls.
"""
import argparse
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .policy import HEAD, MAX_THINKING_TOKENS, SCORE_CURVE, TOTAL_CONTEXT_TOKENS
RUN_ID = re.compile(r"[0-9a-f]{32}\Z")


class Conflict(ValueError):
    pass


class Access:
    def __init__(self, port, public_origin=None):
        self.local = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.public = None
        if public_origin:
            url = urlsplit(public_origin)
            if (url.scheme != "https" or not url.hostname or url.username or url.password or
                    url.path or url.query or url.fragment or not re.fullmatch(r"[a-zA-Z0-9.-]+", url.hostname)):
                raise ValueError("QOT_PUBLIC_ORIGIN must be an exact HTTPS origin, with no path or credentials")
            self.public = url.hostname.lower() + (f":{url.port}" if url.port and url.port != 443 else "")

    def allows(self, hosts, origins):
        if len(hosts) != 1 or len(origins) > 1:
            return False
        host = hosts[0].lower()
        public_host = host[:-4] if host.endswith(":443") else host
        local, public = host in self.local, self.public is not None and public_host == self.public
        if not local and not public:
            return False
        if not origins:
            return True
        return (local and origins[0] == f"http://{host}") or (public and origins[0] in {f"https://{self.public}", f"https://{self.public}:443"})


class App:
    def __init__(self, data, cache, device):
        self.data, self.cache, self.device = Path(data), Path(cache), device
        self.data.mkdir(parents=True, exist_ok=True)
        self.runtime, self.head = None, None
        self.lock = threading.Lock()
        self.busy, self.phase, self.active, self.error = False, "unloaded", None, None
        self.started = None

    def state(self):
        with self.lock:
            return {"loaded": self.runtime is not None, "ready": self.runtime is not None and self.head is not None,
                    "busy": self.busy, "phase": self.phase, "active_id": self.active, "error": self.error,
                    "started_at": self.started, "head": self.head,
                    "gpu": self.runtime.gpu_name if self.runtime else None,
                    "model": "Qwen/Qwen3.5-0.8B", "output_type": "continuous_score", "score_range": [0, 1],
                    "max_thinking_tokens": MAX_THINKING_TOKENS, "total_context_tokens": TOTAL_CONTEXT_TOKENS,
                    "score_curve": SCORE_CURVE, "decision_operations": ["noul", "choice", "score"],
                    "adapter": self.runtime.active_adapter if self.runtime else None,
                    "cache_policy": "mixed-adapter cache reuse is experimental"}

    def claim(self, phase, operation_id):
        with self.lock:
            if self.busy:
                raise Conflict("Another operation is running. Wait or stop that run first.")
            self.busy, self.phase, self.active, self.error = True, phase, operation_id, None
            self.started = time.time()
            (self.data / ".cancel").unlink(missing_ok=True)

    def execute(self, phase, operation_id, fn, background=False):
        self.claim(phase, operation_id)

        def work():
            try:
                return fn()
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)[:1000]
                if phase == "generating":
                    path = self.data / "runs" / f"{operation_id}.json"
                    if not path.exists():
                        path.parent.mkdir(exist_ok=True)
                        tmp = path.with_suffix(".tmp")
                        tmp.write_text(json.dumps({"run_id": operation_id, "phase": "error", "error": str(exc)[:1000]}))
                        tmp.replace(path)
                if not background:
                    raise
            finally:
                with self.lock:
                    self.busy, self.active, self.started = False, None, None
                    self.phase = "error" if self.error else "ready" if self.head else "needs_head" if self.runtime else "unloaded"
                    (self.data / ".cancel").unlink(missing_ok=True)

        if background:
            threading.Thread(target=work, name=f"qwen-{phase}", daemon=True).start()
            return {"operation_id": operation_id}
        return work()

    def load(self):
        def run():
            if self.runtime is None:
                from .model import Runtime
                self.runtime = Runtime(self.data, self.cache, self.device)
            self.head = None
            if self.runtime.training_info is not None:
                self.head = self.runtime.thinking.head()[0]
            return {"ok": True}
        return self.execute("loading", uuid.uuid4().hex, run, True)

    def require_runtime(self):
        if self.runtime is None:
            raise ValueError("Load the GPU model first.")
        return self.runtime

    def build(self, args):
        def run():
            self.head = self.require_runtime().thinking.build(args)
            return self.head
        return self.execute("preparing", uuid.uuid4().hex, run, True)

    def generate(self, args):
        rid = args.get("run_id", "")
        if not isinstance(rid, str) or not RUN_ID.fullmatch(rid):
            raise ValueError("Invalid run id")
        if (self.data / "runs" / f"{rid}.json").exists():
            raise Conflict("Run id already used")
        return self.execute("generating", rid, lambda: self.require_runtime().thinking.generate(args), True)

    def stop(self, args):
        with self.lock:
            if not self.busy or args.get("operation_id") != self.active:
                raise Conflict("This operation is no longer active.")
            (self.data / ".cancel").touch()
        return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    server_version = "QwenOneTwo"

    def send(self, code, body, mime="application/json"):
        if mime == "application/json":
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A client disconnect does not interrupt operation cleanup.

    def allowed(self):
        if not self.server.access.allows(self.headers.get_all("Host", []), self.headers.get_all("Origin", [])):
            self.send(403, {"error": "Allowed Host and same-origin requests only"})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/state":
            return self.send(200, self.server.app.state())
        if path.startswith("/api/runs/"):
            rid = path.removeprefix("/api/runs/")
            if not RUN_ID.fullmatch(rid):
                return self.send(400, {"error": "Invalid run id"})
            file = self.server.app.data / "runs" / f"{rid}.json"
            try:
                value = json.loads(file.read_text())
            except FileNotFoundError:
                value = {"run_id": rid, "phase": "queued"}
            return self.send(200, value)
        assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in assets:
            filename, mime = assets[path]
            return self.send(200, (self.server.assets / filename).read_bytes(), mime)
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.allowed():
            return
        try:
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "JSON required"})
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                return self.send(413, {"error": "Request must be 1–16384 bytes"})
            args = json.loads(self.rfile.read(length))
            if not isinstance(args, dict):
                raise ValueError("JSON object required")
            app = self.server.app
            path = urlsplit(self.path).path
            if path == "/api/decide":
                return self.send(410, {"error": "Continuous scoring is now inside the generation run. Reload the page and send your prompt."})
            routes = {"/api/load": lambda: app.load(), "/api/prepare": lambda: app.build(args),
                      "/api/generate": lambda: app.generate(args),
                      "/api/stop": lambda: app.stop(args)}
            if path not in routes:
                return self.send(404, {"error": "Not found"})
            self.send(202 if path in ("/api/load", "/api/prepare", "/api/generate") else 200, routes[path]())
        except Conflict as exc:
            self.send(409, {"error": str(exc)})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            self.send(400, {"error": str(exc)[:1000]})
        except Exception as exc:
            self.send(500, {"error": str(exc)[:1000]})

    def log_message(self, format, *args):
        # Quiet polling. Errors remain visible in the browser and application state.
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="QwenOneTwo · GPU-only decision-first inference")
    parser.add_argument("--port", type=int, default=int(os.getenv("QOT_PORT", "3211")))
    parser.add_argument("--no-autoload", action="store_true", help="Serve UI without loading a model until requested")
    args = parser.parse_args()
    data = Path(os.getenv("QOT_DATA_DIR", "data")).resolve()
    cache = Path(os.getenv("QOT_HF_CACHE", str(data / "huggingface"))).resolve()
    app = App(data, cache, os.getenv("QOT_DEVICE", "cuda:0"))
    access = Access(args.port, os.getenv("QOT_PUBLIC_ORIGIN") or os.getenv("QWEN_ONE_TWO_PUBLIC_ORIGIN"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.app, server.access = app, access
    packaged_assets = Path(__file__).parent / "static"
    server.assets = packaged_assets if packaged_assets.is_dir() else Path(__file__).resolve().parent.parent / "static"
    if not args.no_autoload:
        app.load()
    print(f"QwenOneTwo · http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        (data / ".cancel").touch()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
