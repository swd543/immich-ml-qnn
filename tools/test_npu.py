"""Unit test: immich_ml models through QnnSession (NPU) or ORT (CPU).

Run with IMMICH_ML_QNN_URL set (NPU path) or unset (CPU path).
Writes clip_<tag>.npy and face_<tag>.npy to /out/.
"""
import os

import numpy as np
from PIL import Image

TAG = os.environ.get("TEST_TAG", "x")
QNN = os.environ.get("IMMICH_ML_QNN_URL", "")
os.makedirs("/out", exist_ok=True)

# ---------- CLIP visual ----------
import json

from immich_ml.models import from_model_type
from immich_ml.schemas import ModelFormat, ModelTask, ModelType

clip = from_model_type("ViT-B-32__openai", ModelType.VISUAL, ModelTask.SEARCH,
                       model_format=ModelFormat.ONNX)
img = Image.open("/t/guitar_224.png").convert("RGB")
emb = np.asarray(json.loads(clip.predict(img)), dtype=np.float32)
print(f"[{TAG}] CLIP session type: {type(clip.session).__name__}")
print(f"[{TAG}] CLIP emb: shape={emb.shape} norm={float(np.linalg.norm(emb)):.6f}")
np.save(f"/out/clip_{TAG}.npy", emb)

# ---------- Face recognition (ArcFace w600k_r50) ----------
fr = from_model_type("buffalo_l", ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION,
                     model_format=ModelFormat.ONNX)
fr.load()
print(f"[{TAG}] Face session type: {type(fr.session).__name__}")

# deterministic synthetic 112x112 RGB crop (same in both runs)
yy, xx = np.mgrid[0:112, 0:112]
crop = np.zeros((112, 112, 3), dtype=np.uint8)
crop[..., 0] = (xx * 2) % 256
crop[..., 1] = (yy * 2) % 256
crop[..., 2] = ((xx + yy) % 256)
feat = fr.model.get_feat([crop])
feat = np.asarray(feat, dtype=np.float32)
print(f"[{TAG}] face feat: shape={feat.shape} norm={float(np.linalg.norm(feat[0])):.6f}")
np.save(f"/out/face_{TAG}.npy", feat)
print(f"[{TAG}] done")
