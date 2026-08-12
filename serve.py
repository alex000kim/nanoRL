"""The disaggregation transport, in stdlib HTTP: rollout workers submit batches out of
lockstep with training, and the trainer publishes new adapter weights. Nothing about the
loss changes; only the importance ratio stops being 1."""
from __future__ import annotations

import io
import os
import threading
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from core import Batch, Trajectory

# The wire format must not be a code-execution vector: weights_only=True restricts
# unpickling to tensors + these two allowlisted types. NANORL_TOKEN (same value on every
# role) additionally gates all requests.
torch.serialization.add_safe_globals([Batch, Trajectory])
_TOKEN = os.environ.get("NANORL_TOKEN", "")


# --------------------------------------------------------------------------- #
# what actually crosses the wire
# --------------------------------------------------------------------------- #
def _dump(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _load(raw: bytes):
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def adapter_state(policy) -> dict:
    """Only the TRAINABLE tensors; under LoRA that makes per-step sync cheap."""
    # .clone() is load-bearing: .cpu() on a CPU tensor returns the SAME storage, and an
    # aliased snapshot would silently track every optimizer step
    return {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}


def adapter_tarball(policy) -> bytes:
    """The same adapter in PEFT's on-disk format, tarred (vLLM only loads LoRA from a path;
    PEFT serializes itself rather than us hand-reconstructing its format)."""
    import io as _io
    import tarfile
    import tempfile
    inner = getattr(policy, "model", policy)
    if not hasattr(inner, "save_pretrained"):
        return b""
    with tempfile.TemporaryDirectory() as d:
        inner.save_pretrained(d)
        buf = _io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            tf.add(d, arcname=".")
        return buf.getvalue()


def extract_tarball(raw: bytes, dest: str) -> str:
    import io as _io
    import tarfile
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=_io.BytesIO(raw), mode="r") as tf:
        try:
            tf.extractall(dest, filter="data")     # blocks ../ traversal, symlinks, devices
        except TypeError:                          # Python without the PEP-706 backport
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in os.path.normpath(m.name).split(os.sep):
                    raise ValueError(f"unsafe tar member {m.name!r}")
            tf.extractall(dest)
    return dest


# --------------------------------------------------------------------------- #
# trainer side: serve weights, collect rollouts
# --------------------------------------------------------------------------- #
class TrainerServer:
    """Current weights + a bounded queue of incoming batches. The queue bound and the
    version check in pop() are the entire staleness policy."""

    def __init__(self, port: int = 8000, queue_max: int = 8):
        self.version = 0
        self.blob = _dump({})                 # serialized weights for the current version
        self.q: deque = deque(maxlen=queue_max)
        self.lock = threading.Lock()
        self.dropped = 0                      # batches evicted by the bound (a real metric)
        self.received = 0
        self.tar = b""                        # same adapter in PEFT dir format, for vLLM
        server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(self))
        server.daemon_threads = True
        self.server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()

    # ---- called by the training loop ---- #
    def publish(self, policy) -> None:
        """Make the current parameters visible to workers as version+1 (state_dict for HF
        workers, PEFT tarball for vLLM ones)."""
        blob = _dump(adapter_state(policy))   # already detached onto CPU
        # The PEFT tarball only exists for vLLM workers, which train.py restricts to LoRA.
        # For anything else save_pretrained would serialize the FULL model every step.
        tar = adapter_tarball(policy) if getattr(policy, "is_lora", False) else b""
        with self.lock:
            self.version += 1
            self.blob, self.tar = blob, tar

    def pop(self, max_staleness: int, timeout: float = 1800.0):
        """Block until a fresh-enough batch arrives. Returns (batch, staleness).

        `timeout` must exceed ONE worker's full generation cycle, not the mean arrival gap:
        workers stay in phase, so batches arrive in bursts with long troughs between.
        """
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                while self.q:
                    ver, batch = self.q.popleft()
                    lag = self.version - ver
                    if lag <= max_staleness:
                        return batch, lag
                    self.dropped += 1        # too old: drop, do not train on it
            time.sleep(0.05)
        raise TimeoutError(f"no fresh rollouts within {timeout}s — are workers running?")

    def stats(self) -> dict:
        with self.lock:
            return {"queued": len(self.q), "dropped": self.dropped, "received": self.received}

    def close(self) -> None:
        self.server.shutdown()


def _make_handler(store: "TrainerServer"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):            # keep the training log readable
            pass

        def _send(self, code: int, body: bytes = b"", ctype="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _authed(self) -> bool:
            return not _TOKEN or self.headers.get("X-Token", "") == _TOKEN

        def do_GET(self):
            if not self._authed():
                return self._send(403)
            if self.path.startswith("/version"):
                self._send(200, str(store.version).encode())
            elif self.path.startswith("/adapter"):
                with store.lock:
                    ver, tar = store.version, store.tar
                self.send_response(200)
                self.send_header("X-Version", str(ver))
                self.send_header("Content-Length", str(len(tar)))
                self.end_headers()
                self.wfile.write(tar)
            elif self.path.startswith("/weights"):
                with store.lock:
                    ver, blob = store.version, store.blob
                self.send_response(200)
                self.send_header("X-Version", str(ver))
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
            else:
                self._send(404)

        def do_POST(self):
            if not self._authed():
                return self._send(403)
            if not self.path.startswith("/rollouts"):
                return self._send(404)
            n = int(self.headers.get("Content-Length", 0))
            ver = int(self.headers.get("X-Version", 0))
            batch = _load(self.rfile.read(n))
            with store.lock:
                if len(store.q) == store.q.maxlen:
                    store.dropped += 1        # deque eviction is silent; count it
                store.q.append((ver, batch))
                store.received += 1
            self._send(200, b"ok")

    return Handler


# --------------------------------------------------------------------------- #
# rollout side
# --------------------------------------------------------------------------- #
class RolloutClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _get(self, path: str, timeout: float):
        req = urllib.request.Request(f"{self.url}{path}", headers={"X-Token": _TOKEN})
        return urllib.request.urlopen(req, timeout=timeout)

    def wait_for_trainer(self, timeout: float = 600.0) -> None:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._get("/version", timeout=5).read()
                return
            except (urllib.error.URLError, OSError):
                time.sleep(2.0)
        raise TimeoutError(f"trainer at {self.url} never came up")

    def pull_weights(self, have: int):
        """Return (version, state_dict) if the trainer is ahead of `have`, else (have, None)."""
        with self._get("/weights", timeout=120) as r:
            ver = int(r.headers.get("X-Version", 0))
            if ver <= have:
                r.read()
                return have, None
            return ver, _load(r.read())

    def pull_adapter(self, have: int, dest_root: str):
        """PEFT-format adapter for vLLM. Returns (version, path) or (have, None)."""
        with self._get("/adapter", timeout=300) as r:
            ver = int(r.headers.get("X-Version", 0))
            raw = r.read()
            if ver <= have or not raw:
                return have, None
        path = extract_tarball(raw, os.path.join(dest_root, f"v{ver}"))
        # GC old versions or a long run fills /tmp (~150MB x hundreds of steps). Keep the
        # previous one too, in case vLLM still holds a request pointing at it.
        import shutil
        for d in os.listdir(dest_root):
            if d.startswith("v") and d[1:].isdigit() and int(d[1:]) < ver - 1:
                shutil.rmtree(os.path.join(dest_root, d), ignore_errors=True)
        return ver, path

    def submit(self, version: int, batch) -> None:
        req = urllib.request.Request(
            f"{self.url}/rollouts", data=_dump(batch), method="POST",
            headers={"X-Version": str(version), "X-Token": _TOKEN,
                     "Content-Type": "application/octet-stream"})
        urllib.request.urlopen(req, timeout=300).read()
