# STATE (session continuity) — 2026-09-13

## DONE: Immich ML NPU integration (production)

The production `immich-ml` container on the Radxa Q6A now runs
`immich-ml-qnn:local`: CLIP ViT-B/32 visual + ArcFace w600k_r50 recognition
+ **SCRFD face detection** on the QCS6490 NPU (HTP INT8, via qnn-dsp-daemon),
everything else on CPU ORT. 2026-09-13: the full ML pipeline (detection +
recognition + CLIP) is NPU-accelerated; verified live via `/predict`
(bounding boxes + embeddings whose values are exact multiples of the NPU
dequant scales).
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

## SCRFD → NPU: SUPERSEDED 2026-09-13 — "BLOCKED" was a VTCM misconfiguration, not the firmware

**Correction (2026-09-13).** The 2026-09-05 conclusion below is invalid. Re-running the
probes with a correctly pinned VTCM (graph name matched to `HtpGraphConfig`) showed the
real failure mode: the firmware rejects contexts requesting **4 MB VTCM**
(`Request feature vtcm size with value 4194304 unsupported` → surfaced as `0x138d`).
Every Sept-5 SCRFD/min-Resize context binary was built at 4 MB (the HtpGraphConfig name
didn't match the graph, so `vtcm_size_in_mb=2` was silently dropped and the compiler
defaulted to 4 MB); the only control that registered (CLIP) was built at 2 MB. Verified
by string inspection: `ctx_nine/ctx_one/ctxA_oneout/scrfd320/scrfd_dbg/scrfd_6490.bin`
all carry `vtcm_size=4194304`; `ctx_control_clip.bin` carries `2097152`.

**2026-09-13 re-test results (QAIRT 2.37.1, correct vtcm pin, board scratch daemon, port
8092, wedge-safety timeouts, board alive after every run):**
- Minimal single-op int8 `Resize` context (2 MB): **registers + loads** ✓
- Minimal single-op int8 `Gather` (static int64 indices) context (2 MB): **registers + loads** ✓
- **Full SCRFD 640×640, 9 outputs, INT8 per-channel** (10 real-photo calib, vtcm 2 MB,
  5.25 MB context `scrfd_6490_v2.bin`): **registers + loads** ✓
- Quality (x86 HTP sim of the exact context vs CPU float32 ORT, same preprocessed tensor,
  insightface decode, 4 real photos / 6 faces): IoU 0.984–0.996, score deltas ≤ 0.09,
  all faces found by both. Raw score tensors: maxdiff ≤ 0.043 on 0..1.

**Consequence: SCRFD face detection CAN run on the QCS6490 NPU.** CPU cost was 710
ms/frame; expected NPU cost ~15–25 ms (model is lighter than the 8.9 ms ArcFace R50).
Artifacts: dev host `/home/buga/qairt/work/scrfd/` (scrfd_q.dlc, scrfd_6490_v2.bin,
10 calib raws, out_scrfd_v2/ + out3/ sim ground truths, final_compare.py), board
`/home/buga/immich-ml-qnn/work/gather-probe/` (probe contexts + scrfd_6490_v2.bin).

Remaining (production integration, pending go-ahead):
1. daemon: 3rd model route `scrfd` (--scrfd-context/--scrfd-graph, input `input.1`
   [1,3,640,640] + 9 outputs; tensor ids/scales/offsets from context metadata).
2. `QnnSession` routing key `detection` in `immich_ml/models/base.py` — must return 2D
   [K,1]/[K,4]/[K,10] outputs (batch folded, as ORT does) for insightface RetinaFace.
3. VTCM budget: 3 graphs × 2 MB = 6 MB of 8 MB — verify under production load.
4. Rebuild daemon (bookworm) + image + asset staging; verify per §9 (md5 = net-run,
   /predict e2e, cosine gates); keep rollback backup.
5. Lesson: ALWAYS check the context binary's embedded `vtcm_size=` (string search) after
   compiling — a silent HtpGraphConfig name mismatch is invisible in the compile log.

---
Original 2026-09-05 record (superseded; kept for the evidence chain):

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

### Decisions (as of 2026-09-05 — superseded 2026-09-13, see correction above)
- Keep SCRFD on CPU (status quo). Production NPU path (CLIP + ArcFace) unaffected.
- Alternatives (not pursued): Resize-free detector (e.g. YOLOv8-Face family — needs custom
  bridge + accuracy validation); wait for Radxa to ship a newer QAIRT runtime with
  op-package support.

### Addendum 2026-09-13 (QAIRT 2.42 data point)
- Re-ran the ONNX→QNN conversion of the exact production detection model
  (`/cache/facial-recognition/buffalo_l/detection/model.onnx`, 158 nodes) with QAIRT
  **2.42.0** (x86_64 build host, py3.10 venv, onnx 1.15.0, `input.1` pinned
  1,3,640,640): **conversion succeeds** (incl. 2 Resize + 4 Gather nodes) — i.e. the
  graph-level op coverage is fine on the newer toolchain. The block is unchanged and
  strictly at the firmware level: 2.42 context binaries are rejected by the board's
  QNN interface cap (2.32), and the 2.37.1 path still hits the missing `Resize`
  op-package (0x138d).
- One untried rewrite remains: replace the 2 `Resize`(nearest 2×) nodes with
  `Gather` over **static int index tensors** (row-then-column replication is exact —
  no arithmetic, so no INT8 precision loss like the failed ConvTranspose attempt;
  `Gather` is already in the model's op set). Whether the HTP v68 firmware provides a
  registerable `Gather` op package is unknown — a minimal single-op int8 Gather context
  would answer that in ~1 h if pursued (same wedge-safety protocol as the Resize probe).

### Ops notes
- All QAIRT work moved off axiom tmpfs `/tmp` → `/home/buga/qairt/` (persistent, 22 GB):
  fresh 2.37.1 SDK at `sdk-237/`, work dir at `work/` (incl. `work/scrfd/` artifacts).
- Board `/tmp` is tmpfs — wiped on reboot; test .bin files removed, old test containers
  (iml-inspect, npu-probe) removed by name.

## STT/TTS NPU investigation (2026-09-06) — see docs/NPU-ASR-TTS.md

Single-op probe campaign (5 new ops) → **all registered + executed on this
firmware**: Gather, ConvTranspose(2D), Conv-3D (1-D conv), InstanceNorm, Pad.
Combined with the production-proven set, the full op set of **Whisper** (STT)
and **Piper/VITS** (TTS) is available on QCS6490 HTP v68; only Resize remains
blocked (SCRFD finding) and neither model needs it.

Key findings:
- AI Hub has **no** precompiled speech assets for QCS6490 (current + all past
  releases; CLI `find`), and current audio assets are QAIRT-2.45 builds (op
  packages above this firmware's 2.32 interface cap). → compile from source
  with QAIRT 2.37.1 only.
- Piper's official ONNX contains dynamic ops (RandomNormalLike, NonZero,
  ScatterND, dynamic Slice/Where) → must use Qualcomm's open fixed-shape
  `pipertts_en` export recipe (ai-hub-models repo) + our pipeline.
- STT first target: distil-whisper-small.en (encoder-only, single forward)
  or whisper-base; TTS first target: piper EN (e.g. en_US-lessac-medium).

Board-side load-test gotchas (reusable):
- fastrpc needs root/render → run qnn-net-run tests with sudo (RADXA_SUDO in
  vllm-setup/.env).
- rt6490/qnn-net-run is v2.42 (mismatched with the 2.37.1 Radxa runtime →
  fastrpc 0x72); use SDK `bin/aarch64-ubuntu-gcc9.4/qnn-net-run` + Radxa
  2.37.1 libQnnHtp*.so; SDK prebuilt stub/skel are unsigned →
  "unsigned PD not supported" on this board.
- Probe workspace: board `/home/buga/immich-ml-qnn/probe-asr-tts/` (22 MB),
  host `~/qairt/work/probes/`.
- CDSP stayed attached; production immich-ml daemon untouched throughout.

## 2026-09-13: SCRFD NPU deployment (COMPLETE, production)

The VTCM root-cause fix (above) led to the full SCRFD-on-NPU rollout:

1. **Context**: `scrfd_6490_v2.bin` (5.25 MB, `vtcm_size=2097152`, graph name
   `scrfd`, INT8 per-channel, calibrated on 10 real photos 640x640). The
   quantized DLC's **embedded graph name** (search the context binary for the
   string; `qairt.load(...).name` returns the FILE name, not the graph name)
   must exactly match `HtpGraphConfig(name=...)` or the VTCM pin silently
   drops to 4 MB.
2. **Quality gate**: x86 HTP sim vs CPU float32 ORT, 4 real photos / 6 faces:
   all faces found by both, **IoU 0.984-0.996**, scores within 0.09.
3. **Board E2E (real 960x720 photo, 6 faces)**: NPU IoU min 0.9721 vs CPU,
   score diffs <= 0.022, NPU steady ~62-84 ms vs CPU 2.6 s (~42x).
4. **3-model concurrency**: clip + arcface + scrfd contexts coexist in VTCM
   (1 MB + 0.95 MB + 0.95 MB); the daemon loads them in that order (newest
   last = degrades first if VTCM were ever exhausted).
5. **Graceful degradation (contract the user approved)**: `QnnSession`
   (`immich_ml/sessions/qnn.py`) falls back to a lazily-built CPU
   onnxruntime session if the daemon is unreachable / errors / reports the
   model unavailable (per-process, logged once). Detection degrades first
   (loaded last). Fallback verified: results match the CPU baseline.
6. **Daemon v1.1.0** (`daemon/qnn_dsp_daemon.cpp`): multi-output support
   (single-output models are the 1-element case; response = concatenated
   float32 in ONNX graph order), `--scrfd-context`/`--scrfd-graph` flags.
7. **Deployment**: image `immich-ml-qnn:local` (rollback:
   `immich-ml-qnn:rollback-20260913`), container recreated with the same
   `docker run` args; `/health` reports all 3 models; all 7 containers
   healthy.

### Bugs found and fixed in this rollout (lessons)
- **Dangling `dimensions` pointer**: `makeTensor` must take
  `(const uint32_t* dims, uint32_t rank)` — the backend dereferences the
  dims during `graphExecute`, so the storage must outlive the call (stable
  per-model vectors; `reserve()` before push_back). A loop-local
  `std::vector` produced `0x1774` (5007 GRAPH_EXECUTION_FAILED).
- **Unpopulated dequant tables**: refactoring the load loop dropped the
  `outScales[i]`/`outOffsets[i]` assignments; dequant then ran on
  zero-initialized vectors -> all-zero outputs despite valid int8 data in
  `outBufs`. Instrument with raw-NZ + checksum probes (`--selftest`
  pattern), never ship it.
- **QnnSession batch bug (pre-existing, latent)**: the static-batch=1 shape
  check `arr.shape[:1] != self._in_shape[:1]` raised on **batched face
  recognition** ([N,3,112,112]) BEFORE the per-item loop. Multi-face assets
  would have failed. Now validates the CHW dims after the batch axis and
  loops the batch.
- **QNN output tensor IDs** are needed for `graphExecute` (not derivable
  from names); `tools/qnn_sys_introspector.cpp` prints them (also fixed its
  constant name `QNN_SYSTEM_CONTEXT_BINARY_INFO_V3` ->
  `QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3` vs the genuine SDK headers).
- **Board E2E gotchas**: `/predict` multipart needs the `entries` form field
  (JSON pipeline spec, see e2e scripts); the response is nested per task
  (facial-recognition is a LIST of per-face entries); the production
  `docker run` mounts `/cache` **rw** (not ro); scratch daemons must be
  bound to 127.0.0.1 so `curl 127.0.0.1:port` inside the container does not
  traverse the docker bridge and hit the host.
