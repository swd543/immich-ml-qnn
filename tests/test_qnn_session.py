"""Regression tests for immich_ml.sessions.qnn (QnnSession).

Runs with plain stdlib unittest (no pytest):

    python3 -m unittest discover -s tests -v

The NPU daemon is mocked at the urllib boundary, so most tests run anywhere
with numpy. The CPU-fallback tests additionally need onnxruntime and a real
ONNX model file (IMMICH_TEST_ONNX, or /cache/... inside the ML container) and
are skipped when those are absent.
"""

import json
import logging
import os
import sys
import types
import unittest
import urllib.error
import urllib.request

import numpy as np

# Make the package importable when run from a repo checkout or a copied dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_module(name: str, **attrs) -> types.ModuleType:
    """Register a minimal stub for a third-party module if it is missing."""
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        # Attach to parent package if dotted.
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(sys.modules.get(parent, types.ModuleType(parent)), child, mod)
        return mod


# The production image ships gunicorn/pydantic/orjson/fastapi; a bare test env
# may not. Stub just what the import graph needs so the REAL qnn.py, enums and
# routing code under test are exercised.
class _Arbiter:
    WORKER_BOOT_ERROR = 3


_stub_module("gunicorn")
_stub_module("gunicorn.arbiter", Arbiter=_Arbiter)


class _BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _Field(*a, **k):
    return None


_stub_module("pydantic", BaseModel=_BaseModel, Field=_Field)


_stub_module(
    "orjson", dumps=lambda *a, **k: json.dumps(a[0]).encode(), loads=lambda b: json.loads(b)
)


class _JSONResponse:
    def __init__(self, *a, **k):
        pass


_stub_module("fastapi")
_stub_module("fastapi.responses", JSONResponse=_JSONResponse)


class _QnnLog:
    """Stands in for immich_ml.config.log (a gunicorn-backed logger)."""

    def __init__(self):
        self._log = logging.getLogger("qnn-test")

    def info(self, msg, *a, **k):
        self._log.info(msg, *a)

    def warning(self, msg, *a, **k):
        self._log.warning(msg, *a)

    def error(self, msg, *a, **k):
        self._log.error(msg, *a)

    def debug(self, msg, *a, **k):
        self._log.debug(msg, *a)


_config_stub = types.ModuleType("immich_ml.config")
_config_stub.log = _QnnLog()
sys.modules["immich_ml.config"] = _config_stub

from immich_ml.sessions import qnn  # noqa: E402
from immich_ml.sessions.qnn import QnnSession, model_key_for, QNN_ENV_VAR  # noqa: E402
from immich_ml.schemas import ModelTask, ModelType  # noqa: E402

ARC_ONNX_CANDIDATES = [
    os.environ.get("IMMICH_TEST_ONNX", ""),
    "/cache/facial-recognition/buffalo_l/recognition/model.onnx",
    "/usr/src/app/models/facial-recognition/buffalo_l/recognition/model.onnx",
]


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc) -> None:
        return None


class _FakeDaemon:
    """Stands in for the qnn_dsp_daemon HTTP surface."""

    def __init__(self, models=("clip", "arcface", "scrfd")) -> None:
        self.models = list(models)
        self.infer_bodies: list[bytes] = []
        self.next_infer: dict = {}   # key -> callable(payload) -> bytes or exception
        self.mode = "ok"             # ok | http500 | malformed

    def urlopen(self, req, timeout=None):
        url = req.full_url
        if url.endswith("/health"):
            body = json.dumps(
                {"status": "ok", "models": [{"name": m} for m in self.models]}
            ).encode()
            return _FakeResp(body)
        if "/infer/" in url:
            key = url.rsplit("/infer/", 1)[1]
            payload = req.data or b""
            self.infer_bodies.append((key, bytes(payload)))
            if self.mode == "http500":
                raise urllib.error.HTTPError(url, 500, "boom", None, None)
            if self.mode == "malformed":
                return _FakeResp(b"\x01\x02\x03")  # not even a multiple of 4
            if key in self.next_infer:
                result = self.next_infer[key](payload)
                if isinstance(result, Exception):
                    raise result
                return _FakeResp(result)
            # default: echo a deterministic float32 buffer of the exact
            # expected output size for this model
            if key == "scrfd":
                n = sum(int(np.prod(sh)) for sh in qnn.MODELS["scrfd"]["out_shapes"])
            else:
                n = 512
            vals = (np.arange(n, dtype=np.float32) % 7) - 3
            return _FakeResp(vals.tobytes())
        raise urllib.error.HTTPError(url, 404, "nf", None, None)


def _make(key: str, onnx: str = "/nonexistent/model.onnx") -> QnnSession:
    return QnnSession(onnx, key)


def _floats(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


class QnnSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.get(QNN_ENV_VAR)
        os.environ[QNN_ENV_VAR] = "http://127.0.0.1:9"
        self.daemon = _FakeDaemon()
        self._real_urlopen = urllib.request.urlopen
        urllib.request.urlopen = self.daemon.urlopen  # type: ignore[assignment]

    def tearDown(self) -> None:
        urllib.request.urlopen = self._real_urlopen  # type: ignore[assignment]
        if self._old_env is None:
            os.environ.pop(QNN_ENV_VAR, None)
        else:
            os.environ[QNN_ENV_VAR] = self._old_env

    # ------------------------------------------------------------------ NPU
    def test_single_face_arcface_shape_and_values(self) -> None:
        s = _make("arcface")
        x = np.zeros((1, 3, 112, 112), dtype=np.float32)
        x[0, 0, 0, 0] = 42.0
        outs = s.run(None, {"input_1": x})
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].shape, (1, 512))
        self.assertEqual(len(self.daemon.infer_bodies), 1)
        key, body = self.daemon.infer_bodies[0]
        self.assertEqual(key, "arcface")
        self.assertEqual(len(body), 3 * 112 * 112 * 4)

    def test_batch_multi_face_concatenates_to_n_512(self) -> None:
        """Regression: np.stack produced (N,1,512) and broke pgvector."""
        s = _make("arcface")
        n = 3
        x = np.zeros((n, 3, 112, 112), dtype=np.float32)
        for i in range(n):
            x[i, 0, 0, 0] = float(i)

        # Each call must return a distinct per-item buffer so ordering bugs
        # are visible. (infer_bodies is appended before the responder runs, so
        # the 1st call sees len == 1.)
        def responder(payload: bytes) -> bytes:
            idx = len(self.daemon.infer_bodies) - 1
            vals = np.full(512, float(idx), dtype=np.float32)
            return vals.tobytes()

        self.daemon.next_infer["arcface"] = responder
        outs = s.run(None, {"input_1": x})
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].shape, (n, 512), "batch axis must be (N,512), not (N,1,512)")
        self.assertEqual(len(self.daemon.infer_bodies), n, "static-batch=1 graph: one call per item")
        for i in range(n):
            self.assertTrue(np.allclose(outs[0][i], float(i)), f"row {i} out of order")

    def test_scrfd_nine_outputs_split_in_graph_order(self) -> None:
        s = _make("scrfd")
        x = np.zeros((1, 3, 640, 640), dtype=np.float32)

        specs = qnn.MODELS["scrfd"]["out_shapes"]
        blocks = [
            np.full(int(np.prod(sh)), float(i), dtype=np.float32) for i, sh in enumerate(specs)
        ]
        self.daemon.next_infer["scrfd"] = lambda p: b"".join(b.tobytes() for b in blocks)
        outs = s.run(None, {"input_1": x})
        self.assertEqual(len(outs), 9)
        for i, (sh, block) in enumerate(zip(specs, blocks)):
            self.assertEqual(outs[i].shape, sh, f"output {i} shape")
            self.assertTrue(np.allclose(outs[i], float(i)), f"output {i} content")

    def test_wrong_input_shape_raises(self) -> None:
        s = _make("arcface")
        with self.assertRaises(ValueError):
            s.run(None, {"input_1": np.zeros((1, 3, 96, 112), dtype=np.float32)})

    # -------------------------------------------------------------- fallback
    def _arc_onnx(self) -> str:
        for p in ARC_ONNX_CANDIDATES:
            if p and os.path.isfile(p):
                return p
        return ""

    def test_fallback_on_http_error_uses_cpu(self) -> None:
        onnx = self._arc_onnx()
        if not onnx or not _have_ort():
            self.skipTest("onnxruntime/model unavailable for CPU fallback test")
        s = _make("arcface", onnx)
        x = np.zeros((1, 3, 112, 112), dtype=np.float32)
        # 1st: daemon 500s -> NPU path fails -> CPU fallback takes over.
        self.daemon.mode = "http500"
        outs = s.run(None, {"input_1": x})
        self.assertFalse(s._npu_ok, "NPU path must be disabled after daemon 500")
        self.assertIsNotNone(s._cpu)
        self.assertEqual(outs[0].shape, (1, 512))
        # 2nd: no more daemon calls at all.
        calls_before = len(self.daemon.infer_bodies)
        s.run(None, {"input_1": x})
        self.assertEqual(len(self.daemon.infer_bodies), calls_before,
                         "fallback must be sticky; daemon must not be queried again")

    def test_fallback_on_malformed_response(self) -> None:
        onnx = self._arc_onnx()
        if not onnx or not _have_ort():
            self.skipTest("onnxruntime/model unavailable for CPU fallback test")
        s = _make("arcface", onnx)
        x = np.zeros((1, 3, 112, 112), dtype=np.float32)
        self.daemon.mode = "malformed"
        outs = s.run(None, {"input_1": x})
        self.assertFalse(s._npu_ok, "malformed response must degrade to CPU")
        self.assertEqual(outs[0].shape, (1, 512))

    def test_daemon_unavailable_at_construction_marks_cpu(self) -> None:
        self.daemon.models = []  # /health: none of our models loaded
        s = _make("arcface")
        self.assertFalse(s._npu_ok)


def _have_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.get(QNN_ENV_VAR)

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop(QNN_ENV_VAR, None)
        else:
            os.environ[QNN_ENV_VAR] = self._old_env

    def test_routing_requires_env_and_exact_model_names(self) -> None:
        os.environ.pop(QNN_ENV_VAR, None)
        self.assertIsNone(model_key_for(ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION,
                                        "buffalo_l"))
        os.environ[QNN_ENV_VAR] = "http://127.0.0.1:9"
        self.assertEqual(model_key_for(ModelType.VISUAL, ModelTask.SEARCH, "ViT-B-32__openai"), "clip")
        self.assertEqual(
            model_key_for(ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION, "buffalo_l"),
            "arcface",
        )
        self.assertEqual(
            model_key_for(ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION, "buffalo_l"),
            "scrfd",
        )
        # Other InsightFace bundles must stay on ORT (incompatible weights).
        self.assertIsNone(
            model_key_for(ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION, "buffalo_sc"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
