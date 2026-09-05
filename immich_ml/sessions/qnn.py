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
        "in_name": "input.1",
        "in_shape": (1, 3, 112, 112),
        "out_name": "683",
        "out_shape": (1, 512),
    },
}

# (model_type, model_task) -> (daemon model key, set of model names backed by a
# QNN context binary shipped in the daemon).
#   * clip ViT-B/32__openai : clip_visual.onnx -> clipr37_6490.bin (INT8)
#   * buffalo_l / buffalo_m / antelopev2 : w600k_r50 recognition model
#     -> arcface37v6_6490.bin (INT8). (buffalo_s uses w600k_mbf: not covered.)
_QNN_ROUTES: dict[tuple[str, str], tuple[str, frozenset[str]]] = {
    ("visual", "clip"): ("clip", frozenset({"ViT-B-32__openai"})),
    ("recognition", "facial-recognition"): (
        "arcface",
        frozenset({"buffalo_l", "buffalo_m", "antelopev2"}),
    ),
}


class _Node:
    """Minimal SessionNode (name/shape protocol)."""

    def __init__(self, name: str, shape: tuple[Any, ...]) -> None:
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QnnNode(name={self.name!r}, shape={self.shape})"


class QnnSession:
    """ModelSession implementation backed by qnn-dsp-daemon over HTTP."""

    def __init__(self, model_path: Path | str, model_key: str) -> None:
        self.model_key = model_key
        self.model_path = Path(model_path)
        self.url = os.environ.get(QNN_ENV_VAR, "").rstrip("/")
        if not self.url:
            raise RuntimeError(f"{QNN_ENV_VAR} is not set; cannot use QNN session '{model_key}'")
        spec = MODELS[model_key]
        self._in_name: str = spec["in_name"]
        self._in_shape: tuple[int, ...] = spec["in_shape"]
        self._out_name: str = spec["out_name"]
        self._out_shape: tuple[int, ...] = spec["out_shape"]
        self._health = self._get("/health")
        log.info(
            f"QnnSession: routing '{model_key}' ({model_path.name}) to NPU daemon at {self.url}"
        )

    # ------------------------------------------------------------------ http
    def _get(self, path: str) -> bytes:
        req = urllib.request.Request(f"{self.url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def _post(self, path: str, payload: bytes) -> bytes:
        req = urllib.request.Request(f"{self.url}{path}", data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    # ------------------------------------------------------- session protocol
    def get_inputs(self) -> list[_Node]:
        # FaceRecognizer._load() skips rewriting the ONNX batch axis when
        # str(shape[0]) == "batch"; the QNN graph is static batch=1 and the
        # session handles batching by looping, so report a dynamic dim.
        first_dim: Any = "batch" if self.model_key == "arcface" else self._in_shape[0]
        return [_Node(self._in_name, (first_dim, *self._in_shape[1:]))]

    def get_outputs(self) -> list[_Node]:
        return [_Node(self._out_name, self._out_shape)]

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        arr = next(iter(input_feed.values()))
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.shape[:1] != self._in_shape[:1]:
            raise ValueError(
                f"QnnSession '{self.model_key}': unexpected input shape {arr.shape},"
                f" expected leading dims {self._in_shape}"
            )
        if arr.ndim == 4 and arr.shape[0] > 1:
            # Batched input (face recognition): the QNN graph is static
            # batch=1, so run per item and re-stack.
            outs = [self._infer_one(arr[i]) for i in range(arr.shape[0])]
            return [np.stack(outs, axis=0)]
        return [self._infer_one(arr[0])]

    def _infer_one(self, chw: NDArray[np.float32]) -> NDArray[np.float32]:
        if chw.shape != self._in_shape[1:]:
            raise ValueError(
                f"QnnSession '{self.model_key}': item shape {chw.shape} != {self._in_shape[1:]}"
            )
        payload = np.ascontiguousarray(chw, dtype=np.float32).tobytes()
        raw = self._post(f"/infer/{self.model_key}", payload)
        return np.frombuffer(raw, dtype=np.float32).reshape(self._out_shape)

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
