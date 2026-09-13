"""QNN (Qualcomm Hexagon NPU) model session.

Drop-in replacement for ``immich_ml.sessions.ort.OrtSession`` that routes
inference to a ``qnn-dsp-daemon`` process running pre-built QNN context
binaries on the Qualcomm Hexagon cDSP (HTP backend, INT8).

The daemon owns the QNN C API, the context binaries, and the float32<->int8
quantization/dequantization (per the context encodings). This session only
transfers raw float32 tensor bytes over loopback HTTP and reshapes results.

Activation: set ``IMMICH_ML_QNN_URL`` (e.g. ``http://127.0.0.1:8089``).
When unset, this module is inert and models use the regular ORT sessions.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from immich_ml.config import log

QNN_ENV_VAR = "IMMICH_ML_QNN_URL"

# --------------------------------------------------------------------------
# Tensor ground truth (QAIRT 2.37, HTP backend, INT8 context binaries).
# Verified with qairt-dlc-info on the quantized DLCs and qnn-net-run on the
# board (QCS6490).
# --------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = {
    "clip": {
        "in_name": "image",
        "in_shape": (1, 3, 224, 224),
        "out_name": "embedding",
        "out_shape": (1, 512),
    },
    "arcface": {
        # Names from arcface37v6_6490.bin, not the source ONNX. QnnSession
        # ignores feed/output names internally today, but exposing the actual
        # context contract keeps callers and future validation correct.
        "in_name": "input_1",
        "in_shape": (1, 3, 112, 112),
        "out_name": "_683",
        "out_shape": (1, 512),
    },
    "scrfd": {
        # SCRFD-2.5G (buffalo_l detection), scrfd_6490_v2.bin. Multi-output:
        # the daemon returns one concatenated float32 buffer with all 9
        # outputs in ONNX graph-output order (score/box/kps per FPN level).
        "in_name": "input_1",
        "in_shape": (1, 3, 640, 640),
        "out_names": ["448", "471", "494", "451", "474", "497", "454", "477", "500"],
        "out_shapes": [
            (12800, 1), (3200, 1), (800, 1),        # scores (strides 8/16/32)
            (12800, 4), (3200, 4), (800, 4),        # bboxes
            (12800, 10), (3200, 10), (800, 10),     # 5-landmark kps
        ],
    },
}

# (model_type, model_task) -> (daemon model key, set of model names backed by a
# QNN context binary shipped in the daemon).
#   * clip ViT-B/32__openai : clip_visual.onnx -> clipr37_6490.bin (INT8)
#   * buffalo_l only: w600k_r50 recognition model -> arcface37v6_6490.bin
#     (INT8). Other InsightFace bundles must stay on ORT unless separately
#     converted and validated; silently substituting buffalo_l weights would
#     produce incompatible embeddings.
_QNN_ROUTES: dict[tuple[str, str], tuple[str, frozenset[str]]] = {
    ("visual", "clip"): ("clip", frozenset({"ViT-B-32__openai"})),
    ("recognition", "facial-recognition"): (
        "arcface",
        frozenset({"buffalo_l"}),
    ),
    # Face detection (SCRFD-2.5G, buffalo_l) on the NPU: scrfd_6490_v2.bin.
    ("detection", "facial-recognition"): ("scrfd", frozenset({"buffalo_l"})),
}


class _Node:
    """Minimal SessionNode (name/shape protocol)."""

    def __init__(self, name: str, shape: tuple[Any, ...]) -> None:
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QnnNode(name={self.name!r}, shape={self.shape})"


class _NpuUnavailable(Exception):
    """The NPU daemon rejected this model/request (not loaded, execute error,
    or transport failure). Triggers the per-session CPU fallback."""


class QnnSession:
    """ModelSession implementation backed by qnn-dsp-daemon over HTTP.

    Graceful degradation: if the daemon does not have this model loaded (e.g.
    the Hexagon VTCM could only host a subset of the contexts) or a request
    fails, this session falls back to a CPU ONNX Runtime session for the same
    model file and serves requests from there for the rest of its lifetime.
    """

    def __init__(self, model_path: Path | str, model_key: str) -> None:
        self.model_key = model_key
        self.model_path = Path(model_path)
        self.url = os.environ.get(QNN_ENV_VAR, "").rstrip("/")
        if not self.url:
            raise RuntimeError(f"{QNN_ENV_VAR} is not set; cannot use QNN session '{model_key}'")
        spec = MODELS[model_key]
        self._in_name: str = spec["in_name"]
        self._in_shape: tuple[int, ...] = spec["in_shape"]
        self._out_names: list[str] = (
            list(spec["out_names"]) if "out_names" in spec else [spec["out_name"]]
        )
        self._out_shapes: list[tuple[int, ...]] = (
            list(spec["out_shapes"]) if "out_shapes" in spec else [spec["out_shape"]]
        )
        assert len(self._out_names) == len(self._out_shapes), "spec names/shapes mismatch"
        # Element counts the daemon's concatenated response is split by.
        self._out_elems = [int(np.prod(sh)) for sh in self._out_shapes]
        self._npu_ok: bool = self._npu_has_model()
        self._cpu: Any = None
        if self._npu_ok:
            log.info(
                f"QnnSession: routing '{model_key}' ({self.model_path.name}) to NPU daemon at {self.url}"
            )
        else:
            log.warning(
                f"QnnSession: daemon at {self.url} has no '{model_key}' model "
                f"({self.model_path.name}) — using CPU fallback for this session"
            )

    def _npu_has_model(self) -> bool:
        """Query the daemon's /health and check whether this model is loaded."""
        try:
            import json

            raw = self._get("/health")
            models = json.loads(raw).get("models", [])
            return any(m.get("name") == self.model_key for m in models)
        except Exception as exc:  # daemon unreachable — treat as unavailable
            log.warning(f"QnnSession: cannot reach NPU daemon at {self.url}: {exc}")
            return False

    # ------------------------------------------------------------------ http
    def _get(self, path: str) -> bytes:
        req = urllib.request.Request(f"{self.url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def _post(self, path: str, payload: bytes) -> bytes:
        req = urllib.request.Request(f"{self.url}{path}", data=payload, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # Any daemon error (500 execute failure, 503 model not loaded, 4xx
            # contract mismatch) means the NPU path cannot serve this model;
            # degrade to CPU rather than breaking detection.
            raise _NpuUnavailable(f"daemon returned {exc.code} for {path}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _NpuUnavailable(f"daemon unreachable for {path}: {exc}") from exc

    # ------------------------------------------------------- session protocol
    def get_inputs(self) -> list[_Node]:
        # FaceRecognizer._load() skips rewriting the ONNX batch axis when
        # str(shape[0]) == "batch"; the QNN graph is static batch=1 and the
        # session handles batching by looping, so report a dynamic dim.
        first_dim: Any = "batch" if self.model_key == "arcface" else self._in_shape[0]
        return [_Node(self._in_name, (first_dim, *self._in_shape[1:]))]

    def get_outputs(self) -> list[_Node]:
        return [_Node(n, sh) for n, sh in zip(self._out_names, self._out_shapes)]

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        arr = next(iter(input_feed.values()))
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        # The QNN graph is static batch=1: validate the CHW dims and handle
        # the batch axis by looping (multi-face recognition requests arrive
        # as [N, 3, H, W]).
        if arr.ndim != 4 or arr.shape[1:] != self._in_shape[1:]:
            raise ValueError(
                f"QnnSession '{self.model_key}': unexpected input shape {arr.shape},"
                f" expected {self._in_shape[1:]} after the batch axis"
            )
        # _infer_one returns a list of output arrays, one per model output
        # (a single-output model returns a 1-element list). run() returns
        # that list directly — OrtSession-compatible for multi-output graphs.
        if arr.shape[0] > 1:
            # Batched input (multi-face recognition): the QNN graph is
            # static batch=1, so run per item and concatenate along the
            # batch axis. Each per-item output already carries the batch
            # axis ((1, ...) from _out_shapes); np.stack would add a
            # second one ((N, 1, ...)) and corrupt the embedding shape.
            outs = [self._infer_one(arr[i]) for i in range(arr.shape[0])]
            return [
                np.concatenate([o[j] for o in outs], axis=0)
                for j in range(len(self._out_shapes))
            ]
        return self._infer_one(arr[0])

    def _cpu_session(self) -> Any:
        """Lazily create the CPU ONNX Runtime fallback for this model file.

        Built directly on onnxruntime (not via immich_ml.sessions.ort) so the
        lazy import cannot create a package import cycle (ort.py imports
        immich_ml.models, which imports the sessions package).
        """
        if self._cpu is None:
            import onnxruntime as _ort

            log.warning(
                f"QnnSession '{self.model_key}': falling back to CPU ONNX Runtime "
                f"({self.model_path.name})"
            )
            self._cpu = _ort.InferenceSession(str(self.model_path))
        return self._cpu

    def _cpu_run(self, chw: NDArray[np.float32]) -> list:
        # The ONNX file's input name (e.g. "input.1") differs from the QNN
        # context tensor name (e.g. "input_1"); use the ONNX session's own.
        # _infer_one passes a 3-D CHW item; ORT expects the full batch shape.
        cpu_in = self._cpu.get_inputs()[0].name
        x = chw.reshape(self._in_shape) if chw.ndim == len(self._in_shape) - 1 else chw
        return list(self._cpu.run(None, {cpu_in: x}))

    def _infer_one(self, chw: NDArray[np.float32]) -> NDArray[np.float32]:
        if chw.shape != self._in_shape[1:]:
            raise ValueError(
                f"QnnSession '{self.model_key}': item shape {chw.shape} != {self._in_shape[1:]}"
            )
        if self._npu_ok:
            try:
                payload = np.ascontiguousarray(chw, dtype=np.float32).tobytes()
                raw = self._post(f"/infer/{self.model_key}", payload)
                flat = np.frombuffer(raw, dtype=np.float32)
                if flat.size != sum(self._out_elems):
                    raise _NpuUnavailable(
                        f"unexpected response size {flat.size} "
                        f"(expected {sum(self._out_elems)})"
                    )
                outs = []
                off = 0
                for elems, shape in zip(self._out_elems, self._out_shapes):
                    outs.append(flat[off : off + elems].copy().reshape(shape))
                    off += elems
                return outs
            except _NpuUnavailable as exc:
                self._npu_ok = False
                log.warning(f"QnnSession '{self.model_key}': NPU path failed: {exc}")
        self._cpu_session()  # ensure created
        return [np.asarray(o) for o in self._cpu_run(chw)]

    # ------------------------------------------------------- OrtSession-style
    @property
    def providers(self) -> list[str]:
        return ["QnnDspExecutionProvider"]

    @property
    def provider_options(self) -> list[dict[str, Any]]:
        return [{}]

    @property
    def sess_options(self) -> Any:
        return None


def qnn_enabled() -> bool:
    return bool(os.environ.get(QNN_ENV_VAR, "").strip())


def model_key_for(model_type: Any, model_task: Any, model_name: str) -> str | None:
    """Return the daemon model key if this model should route to the NPU."""
    if not qnn_enabled():
        return None
    route = _QNN_ROUTES.get((model_type.value, model_task.value))
    if route and model_name in route[1]:
        return route[0]
    return None
