# STATE (session continuity) — 2026-09-05

## DONE: Immich ML NPU integration (production)

The production `immich-ml` container on the Radxa Q6A now runs
`immich-ml-qnn:local`: CLIP ViT-B/32 visual + ArcFace w600k_r50 recognition
on the QCS6490 NPU (HTP INT8, via qnn-dsp-daemon), everything else on CPU ORT.
Verified live: `/predict` returns CLIP embeddings + face detection/recognition;
immich server reports the ML service healthy; all 6 containers healthy.

### Key facts
- Image: `immich-ml-qnn:local` (FROM `ghcr.io/immich-app/immich-machine-learning:release`, v3.1.0).
- Daemon binary in image = `qnn_dsp_daemon_bookworm` (built in a debian:bookworm
  container on the board; the board-glibc binary fails with GLIBC_2.38 in the
  bookworm image).
- Image needs `libyaml-0-2` + `libatomic1` (installed in Dockerfile) for the HTP stub.
- Daemon must run WITHOUT `LD_PRELOAD` (image preloads mimalloc; breaks fastrpc).
- Entrypoint: `/entrypoint-qnn.sh` (starts daemon with `env -u LD_PRELOAD`, cwd
  `/opt/qnn/runtime`, health-waits, then `exec python -m immich_ml`).
- Env gate: `IMMICH_ML_QNN_URL=http://127.0.0.1:8089` (unset = stock CPU behaviour).
- Routing hook: `immich_ml/models/base.py::_make_session` → QnnSession for
  model keys `clip` (CLIP visual) / `arcface` (face recognition).
- QnnSession (`immich_ml/sessions/qnn.py`): duck-types ort.Session
  (get_inputs/get_outputs/run); insightface ArcFaceONNX works unmodified.
- HTTP daemon bug fixed this session: body-read loop could overshoot
  Content-Length → corrupted inputs → deviant outputs. Now reads exactly
  Content-Length bytes + logs FNV-1a64 body checksum.
- Verification: CLIP NPU vs CPU cosine 0.91; face 0.95 (real face); daemon md5
  = qnn-net-run ground truth (clip `dc209b6b…`, face `90f9c96e…`).
- Production container created via `docker run` (docker-compose v1 on the board
  cannot CREATE containers with the current docker daemon: KeyError
  ContainerConfig). Compose labels set; see README "Production swap" + rollback.
- Rollback backups: `/home/buga/immich-ml-qnn/swap-backup/` (board).
- Artifacts (context bins, ground refs, raws, board runtime):
  `/home/buga/immich-ml-qnn/artifacts/` (board).

## OPEN
- Forum draft (`/home/buga/code/radxa-q6a-forum-post.md`): fold in WiFi/USB-PM
  fix, NPU model verification, CDSP; user go-ahead needed to post (account was
  anti-spam blocked earlier; user swd543 not banned).
- 22:50 reboot mystery: confirm with user if not user-triggered.
- Optional future: SCRFD detector on NPU, sidecar split, upstream QnnSession patch.
- Note: axiom `/tmp/qairt-work/` (SDKs/venv/DLCs/headers) is volatile;
  board `/tmp/` artifacts are persisted under /home/buga/immich-ml-qnn/artifacts/.

## SCRFD → NPU: BLOCKED (2026-09-05, root cause proven)

Attempted to port the buffalo_l SCRFD face detector (last heavy CPU model) to the NPU.
**Result: not possible on this board's firmware. Root cause proven with full evidence chain.**

### The 0x138d mystery — solved
`0x138d = 5005 = SNPE_ERRORCODE_QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND`
(in SDK headers: `include/SNPE/DlSystem/DlError.h`). Context registration fails because the
context references a QNN op package the device's HTP backend does not provide.

### Trigger: the ONNX `Resize` (nearest) op
| Test (all via same pipeline + soc_details="chipset:SM7325;dsp_arch:v68;soc_model:35") | Result |
|---|---|
| SCRFD 9 outputs | 0x138d |
| SCRFD 1 output | 0x138d |
| SCRFD 320x320 / vtcm 4/6 / Sigmoid→Identity | 0x138d |
| **Minimal single-op int8 Resize model (20x20→40x40)** | **0x138d** |
| **CLIP (production model) rebuilt via same pipeline** | **registers + runs** ✓ |
| SCRFD on x86 HTP sim (qnn-net-run) | runs, matches ORT (cos ~1.0) |

Conclusion: the QCS6490 HTP firmware (QAIRT v2.37.1 runtime, QNN interface cap 2.32) does
**not** ship the op package the HTP compiler needs to lower `Resize`. No such package exists
in the 2.37.1 SDK to deploy (`lib/hexagon-v68/unsigned/` has only the standard skels).
Nearest-2x upsample is **non-local**: impossible with Conv/ConvTranspose (measured
maxdiff 1.51 vs ORT for the ConvTranspose attempt). SCRFD's multi-scale anchor blending
(56-ch prior maps) requires Resize → **SCRFD cannot run on the QCS6490 NPU**.

### Toolchain facts learned
- Generator CLI `--config_file`: `graphs`/`devices`/`soc_model` sections are **silently
  ignored** (byte-identical binaries). Correct path: Python API
  `qairt.compile(model, config=CompileConfig(backend="HTP", soc_details=..., graph_custom_configs=[HtpGraphConfig(...)]))`.
- `soc_model` is an **int** SoC-family id (`QNN_SOC_MODEL_SM7325 = 35`); a *string*
  ("QCS6490") in the config path hung the CDSP (2026-09-05 board hang incident, power-cycle
  recovery). `dsp_arch` values: v66/v68/v69/v73/v75/v79/v81.
- The 2.37.1 converter's SoC support table is **empty** (`is_soc_model_supported()` is False
  for every SoC incl. QCS6490/SM7325) — `--target_soc_model` is unusable in this build.
- QCS6490 = `QNN_SOC_MODEL_SM7325 = 35` (SM7325 family), HTP v68.
- SCRFD ONNX: 9 outputs (score/bbox/kps x 3 strides 8/16/32: 12800/3200/800 anchors),
  58 Conv + 2 Resize + 3 Sigmoid + 9 Transpose/Reshape + 3 AvgPool + 1 MaxPool;
  anchor priors are content-derived (lateral convs), NOT constants.

### Decisions
- Keep SCRFD on CPU (status quo). Production NPU path (CLIP + ArcFace) unaffected.
- Alternatives (not pursued): Resize-free detector (e.g. YOLOv8-Face family — needs custom
  bridge + accuracy validation); wait for Radxa to ship a newer QAIRT runtime with
  op-package support.

### Ops notes
- All QAIRT work moved off axiom tmpfs `/tmp` → `/home/buga/qairt/` (persistent, 22 GB):
  fresh 2.37.1 SDK at `sdk-237/`, work dir at `work/` (incl. `work/scrfd/` artifacts).
- Board `/tmp` is tmpfs — wiped on reboot; test .bin files removed, old test containers
  (iml-inspect, npu-probe) removed by name.
