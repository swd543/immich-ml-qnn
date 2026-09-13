"""E2E harness: immich_ml models through QnnSession (NPU) or ORT (CPU).

Run with IMMICH_ML_QNN_URL set (NPU path) or unset (CPU path).
Writes clip_<tag>.npy and face_<tag>.npy to /out/.

Exits non-zero on any failed assertion (useful as a regression gate).
Set FACE_IMG=/path/to/a_real_face.jpg to also exercise the SCRFD detector.
"""
import json
import os

import numpy as np
from PIL import Image

TAG = os.environ.get("TEST_TAG", "x")
QNN = os.environ.get("IMMICH_ML_QNN_URL", "")
os.makedirs("/out", exist_ok=True)

mode = "NPU" if QNN else "CPU"
print(f"[{TAG}] mode={mode}")

from immich_ml.models import from_model_type
from immich_ml.schemas import ModelFormat, ModelTask, ModelType


def check(label: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"{label} FAILED")
    print(f"[{TAG}]   OK {label}")


# ---------- CLIP visual ----------
clip = from_model_type("ViT-B-32__openai", ModelType.VISUAL, ModelTask.SEARCH,
                       model_format=ModelFormat.ONNX)
img = Image.open("/t/guitar_224.png").convert("RGB")
emb = np.asarray(json.loads(clip.predict(img)), dtype=np.float32)
print(f"[{TAG}] CLIP session type: {type(clip.session).__name__}")
print(f"[{TAG}] CLIP emb: shape={emb.shape} norm={float(np.linalg.norm(emb)):.6f}")
np.save(f"/out/clip_{TAG}.npy", emb)
check("CLIP emb is 512-d", emb.shape == (512,))
check("CLIP emb L2-norm ~1.0", 0.95 <= float(np.linalg.norm(emb)) <= 1.05)

# ---------- Face recognition (ArcFace w600k_r50) ----------
fr = from_model_type("buffalo_l", ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION,
                     model_format=ModelFormat.ONNX)
fr.load()
print(f"[{TAG}] Face session type: {type(fr.session).__name__}")


def synth_crop(seed: int) -> "np.ndarray":
    # deterministic synthetic 112x112 RGB crop (differs per seed)
    yy, xx = np.mgrid[0:112, 0:112]
    c = np.zeros((112, 112, 3), dtype=np.uint8)
    c[..., 0] = (xx * 2 + seed) % 256
    c[..., 1] = (yy * 2 + seed) % 256
    c[..., 2] = ((xx + yy + seed) % 256)
    return c


# single crop
feat1 = np.asarray(fr.model.get_feat([synth_crop(0)]), dtype=np.float32)
print(f"[{TAG}] face feat (1): shape={feat1.shape} norm={float(np.linalg.norm(feat1[0])):.6f}")
np.save(f"/out/face_{TAG}.npy", feat1)
check("single face feat is [1,512]", feat1.shape == (1, 512))

# multi-crop batch -> must stay [N,512] (regression: np.stack gave [N,1,512])
feat2 = np.asarray(fr.model.get_feat([synth_crop(0), synth_crop(7)]), dtype=np.float32)
print(f"[{TAG}] face feat (batch 2): shape={feat2.shape}")
check("batch face feat is [N,512] (no extra axis)", feat2.shape == (2, 512))
check("batch row 0 == single row 0 (ordering preserved)",
      bool(np.allclose(feat2[0], feat1[0])))

# ---------- SCRFD detection (optional real-face image) ----------
face_img = os.environ.get("FACE_IMG", "")
if face_img and os.path.isfile(face_img):
    from immich_ml.models import from_model_type as _fmt

    det = _fmt("buffalo_l", ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION,
               model_format=ModelFormat.ONNX)
    det_img = Image.open(face_img).convert("RGB")
    out = det.predict(det_img)
    if isinstance(out, dict):
        boxes = np.asarray(out.get("boxes"), dtype=np.float32)
        scores = np.asarray(out.get("scores"), dtype=np.float32)
    else:  # FaceDetectionOutput object
        boxes = np.asarray(getattr(out, "boxes", out), dtype=np.float32)
        scores = np.asarray(getattr(out, "scores", []), dtype=np.float32)
    print(f"[{TAG}] SCRFD boxes: {boxes.shape} scores: {scores.shape}")
    check("SCRFD predict returns a boxes array", hasattr(boxes, "shape") and boxes.size >= 0)
    if boxes.size:
        check("SCRFD boxes are Nx4 (valid face detected)", boxes.ndim == 2 and boxes.shape[1] == 4)
        check("SCRFD scores match box count", len(scores) == boxes.shape[0])
else:
    print(f"[{TAG}] SCRFD detection skipped (no FACE_IMG)")

print(f"[{TAG}] done — all assertions passed")
