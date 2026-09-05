# Reproduction Guide — Immich ML on the QCS6490 NPU

End-to-end, from a clean machine to a running production container. Two
machines are involved:

| Role | Machine | OS |
|---|---|---|
| **Build host** (compiles context binaries, runs the QAIRT x86 tools) | any x86_64 Linux (used: Fedora 44) | glibc ≥ 2.34, ~40 GB free disk |
| **Board** (runs everything) | Radxa Dragon Q6A | RadxaOS (Ubuntu 24.04), kernel 7.0.11-6-qcom |

```
build host (x86_64)                          board (aarch64, Q6A)
┌─────────────────────────────┐              ┌──────────────────────────────────┐
│ QAIRT 2.37.1.250807 SDK     │              │ RadxaOS + QAIRT 2.37.1 runtime   │
│  converter → DLC            │   .bin       │  /usr/lib/... (fastrpc stack)    │
│  quantizer → INT8 DLC ──────┼──scp──▶      │  immich-ml-qnn:local image       │
│  qairt.compile → context ◀──┼── (dlc)      │   ├─ patched immich_ml (python)  │
└─────────────────────────────┘              │   ├─ qnn-dsp-daemon (C++)        │
                                             │   └─ /opt/qnn/{runtime,models}   │
                                             │        │ fastrpc                 │
                                             │        ▼                         │
                                             │   /dev/fastrpc-cdsp → HTP (NPU) │
                                             └──────────────────────────────────┘
```

---

## 1. Board facts (why these specific settings)

| Fact | Value | Consequence |
|---|---|---|
| SoC | QCS6490 = **SM7325 family** | `QNN_SOC_MODEL_SM7325 = 35` (SDK `QnnTypes.h` docs) |
| Hexagon arch | **v68** | skels are `libQnnHtpV68*.so` |
| HMX | 4 units, 8 MB VTCM total | `vtcm_mb: 2` per graph works (2 graphs resident) |
| NPU precision | **INT8 only** | per-channel INT8 quantization required |
| Firmware QNN interface cap | **2.32.0** | must use QAIRT **2.37.1** context binaries (matches the on-board v2.37.1 runtime); newer SDKs (2.42) generate contexts the firmware rejects |
| `Resize` op | **not registerable** | contexts referencing the Resize op-package fail with `0x138d` (OP_PACKAGE_NOT_FOUND) — see §9. This blocks SCRFD on the NPU. |

All commands in this guide use QAIRT **2.37.1.250807** and produce binaries
bit-compatible with the on-board Radxa runtime.

---

## 2. Toolchain setup (build host)

```sh
# --- persistent work dir (NOT /tmp: on many dev boxes /tmp is tmpfs and
# --- evicts multi-GB SDKs mid-session — this happened, twice)
mkdir -p ~/qairt && cd ~/qairt
unzip <path-to>/qairt-v2.37.1.250807.zip        # 1.35 GB zip → ~6 GB
SDK=~/qairt/qairt/2.37.1.250807

# --- Python 3.10 venv (the SDK tools are 3.10-era; system 3.13 is ABI-incompatible)
uv python install 3.10
uv venv qairt-env --python 3.10
uv pip install --python qairt-env/bin/python \
    numpy==1.26.4 onnx==1.14.1 onnxruntime paramiko
# numpy 1.26.4 is MANDATORY for qairt-converter in this SDK release.
# (the uv venv has no `pip`; use `uv pip install --python <venv>/bin/python`)

# --- environment for every tool invocation
export SDK=$HOME/qairt/qairt/2.37.1.250807
export UVLIB=$HOME/.local/share/uv/python/cpython-3.10.21-linux-x86_64-gnu/lib
export LD_LIBRARY_PATH=$UVLIB:$SDK/lib/x86_64-linux-clang
export PYTHONPATH=$SDK/lib/python
export QNN_SDK_ROOT=$SDK        # required by the Python API's save() step
PY=~/qairt/qairt-env/bin/python
```

Quick sanity check:

```sh
$PY $SDK/bin/x86_64-linux-clang/qairt-converter --help | head -3   # prints help
```

---

## 3. The exact Immich models

Immich downloads these into its ML container's cache; extract the ONNX from
the running production container (or the stock image):

```sh
# on the board (paths inside the stock immich-ml container, v3.1.0):
docker exec immich-ml sh -c 'ls /cache/clip/ViT-B-32__openai/visual /cache/facial-recognition/buffalo_l/recognition'
docker cp immich-ml:/cache/clip/ViT-B-32__openai/visual/model.onnx  clip_visual.onnx
docker cp immich-ml:/cache/facial-recognition/buffalo_l/recognition/model.onnx arcface_model.onnx
```

- **CLIP visual** = `ViT-B-32__openai` (open_clip). Input `image` [1,3,224,224]
  float32 (Immich preprocesses: resize+center-crop 224, `(x/255 - mean)/std`
  per-channel). Output `embedding` [1,512], L2-normalised downstream.
- **ArcFace** = `buffalo_l` recognition model (`w600k_r50`). Input
  `input_1` [1,3,112,112] float32. Output `_683` [1,512].
- **Why B/32, not B/16**: A/B tested on this NPU — ViT-B/16 is more sensitive
  to INT8 quantization AND slower; B/32 wins on both axes.
- The text encoder, SCRFD detector and OCR models stay on CPU (ORT) — they are
  never converted.

### 3.1 CLIP ONNX patch (required)

`ViT-B-32__openai` starts with `/visual/conv1/Conv` — a **32×32 kernel,
32×32 stride** conv (3→768 channels), i.e. the ViT patchify. The QAIRT HTP
converter cannot lower that conv layout. It is replaced by an algebraically
identical reshape+Gemm chain:

```sh
$PY tools/patch_clip_conv1.py clip_visual.onnx clip_visual_patched.onnx
```

(`tools/patch_clip_conv1.py` is in this repo; it verifies the Gemm weight
element-wise against the conv weight before saving.)

### 3.2 ArcFace ONNX

No patch needed — `arcface_model.onnx` converts directly.

---

## 4. Calibration sets

Per-channel INT8 quantization needs calibration data *in distribution*:

- **CLIP**: 26 diverse images — ILSVRC2012 val crops (nature objects,
  animals, instruments) + real photos (people, indoor scenes), resized 224×224
  with Immich's exact preprocessing, written as float32 little-endian
  `[1,3,224,224]` CHW `.raw` files, one path per line in `clip_calib.txt`.
- **ArcFace**: face crops 112×112 with the recognition preprocessing
  (`(px - 127.5) / 128`), same `.raw` format, `arcface_calib.txt`.

The lists and raws used for the production binaries are preserved:

- build host: `~/qairt/work/{calib,calib_real,real}/` + `*_calib*.txt`
- board: `/home/buga/immich-ml-qnn/artifacts/`

A few dozen in-distribution images is plenty; the calibration drives
per-channel activation scales, not weights.

---

## 5. Conversion pipeline (per model)

### 5.1 CLIP (ViT-B/32)

```sh
# 1) convert → DLC
$PY $SDK/bin/x86_64-linux-clang/qairt-converter \
  --input_network clip_visual_patched.onnx \
  --output_path clipr37.dlc --target_backend HTP
# expect: "CONVERSION_SUCCESS"

# 2) quantize → INT8 per-channel DLC
$PY $SDK/bin/x86_64-linux-clang/qairt-quantizer \
  -i clipr37.dlc -o clipr37_q.dlc \
  -l clip_calib.txt --act_bitwidth 8 --weights_bitwidth 8 \
  --use_per_channel_quantization --target_backend HTP
# expect: 0 ERROR lines

# 3) compile → HTP context binary (SoC-pinned)
$PY tools/compile_htp.py clipr37_q.dlc clipr37 2 .
# WROTE ./clipr37.bin  (~90 MB)
```

### 5.2 ArcFace (w600k_r50)

```sh
$PY $SDK/bin/x86_64-linux-clang/qairt-converter \
  --input_network arcface_model.onnx \
  --output_path arcface37v6.dlc --target_backend HTP

$PY $SDK/bin/x86_64-linux-clang/qairt-quantizer \
  -i arcface37v6.dlc -o arcface37v6_q.dlc \
  -l arcface_calib.txt --act_bitwidth 8 --weights_bitwidth 8 \
  --use_per_channel_quantization --target_backend HTP

$PY tools/compile_htp.py arcface37v6_q.dlc arcface37 2 .
# WROTE ./arcface37.bin
```

### 5.3 Ground-truth references (do this BEFORE deploying anything)

`qnn-net-run` (x86 or board) executes the context and dumps raw outputs —
these are the bit-exact references the daemon is later verified against:

```sh
$SDK/bin/x86_64-linux-clang/qnn-net-run \
  --backend $SDK/lib/x86_64-linux-clang/libQnnHtp.so \
  --retrieve_context clipr37.bin \
  --input_list clip_test_in.txt \
  --output_dir clipout/
md5sum clipout/Result_0/embedding.raw
# production reference (guitar.raw): dc209b6b350b7a2064682d321e240d25

$SDK/bin/x86_64-linux-clang/qnn-net-run \
  --backend $SDK/lib/x86_64-linux-clang/libQnnHtp.so \
  --retrieve_context arcface37.bin \
  --input_list arcface_test_in.txt \
  --output_dir faceout/
md5sum faceout/Result_0/_683.raw
# production reference (face.raw): 90f9c96e8c897058791ba84e420fc2cc
```

Notes:
- Test inputs are float32 little-endian CHW `.raw` files (same preprocessing
  as §4), listed one path per line.
- Use the **FLOAT-input path** of qnn-net-run as ground truth. (Its INT8-input
  path applies a different quantization of the input and is not what the
  daemon does.)
- The daemon quantizes the float32 input itself (scales/offsets from the
  context metadata), so it must reproduce the FLOAT-input path bit-exactly.

### 5.4 Tensor specs (for reference / sanity checks)

| Model | graph | input (id, dims) | output (id, dims) |
|---|---|---|---|
| CLIP | `clipr37` | `image` uFxp_8 [1,3,224,224] id=1, scale 0.01544376, offset −116 | `embedding` uFxp_8 [1,512] id=726, scale 0.00327797, offset −207 |
| ArcFace | `arcface37` | `input_1` uFxp_8 [1,3,112,112] id=1, scale 0.00688966, offset −136 | `_683` uFxp_8 [1,512] id=475, scale 0.01560364, offset −123 |

`dataFormat = 1032` (QNN_DEFINITION_DEFINED layout) for all tensors.

---

## 6. The daemon (`daemon/qnn_dsp_daemon.cpp`)

A single-file C++ HTTP server using the QNN C API:

- loads each context binary **once at startup** and keeps it resident
  (per-request context create/destroy exhausts the HTP SSR and can crash the
  CDSP — see §9);
- `POST /infer/<key>` with a float32 little-endian CHW body → returns
  float32 little-endian output (quantize/dequant happens inside the daemon);
- `GET /health` → JSON with the loaded models + tensor metadata;
- reads HTTP bodies **exactly** `Content-Length` bytes (an earlier version
  overshot and corrupted inputs — the body-read loop is exact by
  construction, with an FNV-1a64 body checksum logged per request).

Key API details that cost real debugging time:
- tensors are declared with `QNN_DEFINITION_DEFINED` (dataFormat 1032) using
  the exact context tensor names/ids — anything else fails `graphExecute`
  with `0x1774`;
- `sizeof(Qnn_Tensor_t) = 144` in this SDK (v1 layout = 112),
  `quantizeParams` at offset 32, `clientBuf.dataSize` at 104.

### 6.1 Building the daemon (MUST target bookworm glibc 2.36)

The stock immich image is Debian 12 (glibc 2.36); the board's glibc is 2.39.
A board-built binary fails in the image with `GLIBC_2.38 not found`. A
`-static` build **segfaults** (static-glibc `dlopen` mismatch with the
fastrpc runtime). So build inside a native aarch64 `debian:bookworm`
container **on the board**:

```sh
# on the board, from the repo checkout (~/immich-ml-qnn):
docker run --rm -v "$PWD":/src -w /src debian:bookworm bash -c '
  apt-get update && apt-get install -y --no-install-recommends g++ &&
  g++ -O2 -std=c++17 -pthread -o qnn_dsp_daemon_bookworm \
      daemon/qnn_dsp_daemon.cpp &&
  ldd qnn_dsp_daemon_bookworm'
# result → daemon/qnn_dsp_daemon_bookworm (committed to daemon/)
```

(Only the QNN C headers from the SDK are needed at build time; the daemon
links nothing QNN-specific — the backend `.so` is loaded at runtime.)

### 6.2 Runtime dependencies

- **cwd = the HTP runtime dir**: the HTP backend resolves its DSP skel
  (`libQnnHtpV68Skel.so`) from the current working directory. The
  entrypoint `cd /opt/qnn/runtime` for this reason.
- The HTP stub needs **`libyaml-0.so.2` + `libatomic.so.1`** (absent from the
  stock immich image → installed by the Dockerfile).
- The daemon must run **without `LD_PRELOAD`**: the base image preloads
  mimalloc globally, which breaks the fastrpc transport
  (`deviceCreate` fails). The entrypoint uses `env -u LD_PRELOAD` for the
  daemon only; the Python side keeps mimalloc.
- `/dev/fastrpc-cdsp` must be accessible (board node is root:render; the
  container runs as an existing uid in the `render`-equivalent group, or as
  root).

---

## 7. The Docker image

```dockerfile
# (abridged — full file in the repo root)
ARG IMMICH_ML_BASE=ghcr.io/immich-app/immich-machine-learning:release
FROM ${IMMICH_ML_BASE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libyaml-0-2 libatomic1 && rm -rf /var/lib/apt/lists/*
COPY immich_ml /usr/src/immich_ml          # patched package (base.py + sessions/qnn.py)
COPY daemon/qnn_dsp_daemon_bookworm /opt/qnn/qnn_dsp_daemon
COPY daemon/runtime/ /opt/qnn/runtime/     # Radxa v2.37.1 HTP runtime (libQnnHtp.so, skels, stub)
COPY daemon/models/ /opt/qnn/models/       # clipr37_6490.bin + arcface37v6_6490.bin
COPY docker/entrypoint-qnn.sh /entrypoint-qnn.sh
ENV IMMICH_ML_QNN_PORT=8089
ENTRYPOINT ["tini", "--", "/entrypoint-qnn.sh"]
```

```sh
# build on the board (the image pulls from ghcr.io; buildkit required):
cd ~/immich-ml-qnn && docker build -t immich-ml-qnn:local .
```

### 7.1 The Python patch (upstream-mergeable, 201-line diff)

Only two files differ from stock `immich-ml` v3.1.0
(`upstream-diff.patch` in the repo root):

- `immich_ml/models/base.py` — `_make_session()` returns a `QnnSession` for
  model keys `clip`/`arcface` **only when `IMMICH_ML_QNN_URL` is set**
  (read via `os.environ`, not pydantic, so unset = bit-identical stock
  behaviour);
- `immich_ml/sessions/qnn.py` (new) — duck-types `ort.Session`
  (`get_inputs()` / `get_outputs()` / `run()`), so Immich's model code —
  including insightface's `ArcFaceONNX` wrapper — works **unmodified**.

Routing keys: `clip` = CLIP visual encoder, `arcface` = face recognition.
SCRFD detection and everything else stay on the ONNX Runtime path.

---

## 8. Deployment (board)

### 8.1 Container run (production)

The board runs `docker-compose` **v1 (Python)**, which **cannot create
containers** with the current docker daemon (`KeyError: 'ContainerConfig'`
in `get_container_data_volumes`) — it can still *read* running ones. The
production `immich-ml` container is therefore created with `docker run`,
carrying the standard compose labels so it stays part of the `immich`
project:

```sh
docker run -d \
  --name immich-ml \
  --restart unless-stopped \
  --network immich_default \
  --network-alias immich-ml \
  -e COMPOSE_PROJECT_NAME=immich -e COMPOSE_SERVICE=immich-ml \
  -e COMPOSE_VERSION=3.8 \
  -e IMMICH_ML_QNN_URL=http://127.0.0.1:8089 \
  --device /dev/fastrpc-cdsp \
  -v /proc/device-tree:/proc/device-tree:ro \
  -v /usr/lib/dsp:/usr/lib/dsp:ro \
  -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:/usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:ro \
  -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so:/usr/lib/aarch64-linux-gnu/libcdsprpc.so:ro \
  -v /home/buga/FFclone/immich/media/cache:/cache:ro \
  immich-ml-qnn:local
```

Required mounts/devices, and why:

| Mount / device | Why |
|---|---|
| `--device /dev/fastrpc-cdsp` | the CDSP ioctl node (the NPU) |
| `/proc/device-tree` (ro) | HTP backend reads SoC topology (HMX count, VTCM) |
| `/usr/lib/dsp` (ro) | fastrpc daemon dirs (adsp/cdsp) |
| `libcdsprpc.so{,.1}` (ro) | userspace fastrpc client lib |
| `…/media/cache` → `/cache` (ro) | Immich model cache (CLIP/ArcFace ONNX for the CPU models + config) |

### 8.2 Compose / env wiring (ffclone repo, `qnn` branch)

- `infra/immich/docker-compose.yml` — the `immich-ml` service gained
  `IMMICH_ML_QNN_URL` + the four mounts + `devices: [/dev/fastrpc-cdsp]`.
- `stack.env` (gitignored; lives at `/home/buga/FFclone/immich/config/`):
  `IMMICH_ML_IMAGE=immich-ml-qnn:local` (was the stock digest-pinned tag).

After (re)creating the container, the Immich **server** re-registers the ML
service on its next health check; `immich` reports `immich-ml: healthy`.

### 8.3 Rollback

Pre-swap backups of `stack.env` + `docker-compose.yml` are on the board at
`/home/buga/immich-ml-qnn/swap-backup/`:

```sh
cd /home/buga/immich-ml-qnn/swap-backup
cp stack.env /home/buga/FFclone/immich/config/stack.env
cp docker-compose.yml /home/buga/code/ffclone/infra/immich/docker-compose.yml
docker rm -f immich-ml          # named container only — never blanket docker rm
# recreate with the stock image via the docker run command above minus the
# QNN env/mounts (or restore the original compose service)
```

---

## 9. Verification checklist

Run after any change (binaries, daemon, image, deployment):

1. **Daemon unit (scratch container)** — load + one inference per model,
   compare md5 with §5.3 ground truth (must be **bit-exact**):
   ```sh
   docker run --rm --device /dev/fastrpc-cdsp \
     -v /proc/device-tree:/proc/device-tree:ro -v /usr/lib/dsp:/usr/lib/dsp:ro \
     -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:/usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:ro \
     -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so:/usr/lib/aarch64-linux-gnu/libcdsprpc.so:ro \
     -v <repo>:/opt/qnn:ro -v /tmp:/h -w /opt/qnn/runtime \
     qnptest:local /opt/qnn/qnn_dsp_daemon_bookworm --backend ./libQnnHtp.so \
       --port 8092 --bind 127.0.0.1 --clip-context /h/clipr37_6490.bin
   # then: curl -X POST --data-binary @guitar.raw http://127.0.0.1:8092/infer/clip | md5sum
   # clip → dc209b6b350b7a2064682d321e240d25 ; arcface → 90f9c96e8c897058791ba84e420fc2cc
   ```
   Always `timeout 25` around these; always ping+ssh the board afterwards
   (a bad context can wedge the DSP — see §10).
2. **Python session (scratch container with the image)** — `QnnSession` vs
   CPU ORT on the same input: CLIP cosine ≥ 0.90, ArcFace ≥ 0.95 **on a real
   face** (synthetic images are out-of-distribution for the calibration set
   and score lower — expected, not a bug).
3. **End-to-end** — `POST /predict` (multipart: `image` + `entries` JSON on
   port 3003) returns a 512-d CLIP embedding (norm ≈ 1.0) + face detection +
   512-d ArcFace embedding, with the daemon log showing the `/infer` calls.
4. **Production** — all 6 Immich containers healthy; `cat
   /sys/class/remoteproc/remoteproc1/state` = `attached`; daemon log shows
   `listening; models: 2`.
5. **Performance reference** (production, 951×1385 photo, 30 sequential
   `/predict`): ~1.8 req/s, ~543 ms/req end-to-end (PIL decode + SCRFD on
   CPU dominate); NPU share ≈ 17 ms (CLIP ~10.8 ms + ArcFace ~6.7 ms);
   daemon ≈ 0% CPU.

---

## 10. Failure modes & lessons (read before debugging)

| Symptom | Cause | Fix / prevention |
|---|---|---|
| `contextCreateFromBinary ... 0x138d` | **OP_PACKAGE_NOT_FOUND** (5005): the context references an op package the device firmware lacks. Proven trigger on this board: the `Resize` op (a minimal single-op int8 Resize model fails identically; the same pipeline on CLIP succeeds) | Do not use `Resize`/`Upsample` in NPU models on this firmware; see §11 |
| Board hangs (ping ok, SSH dead), CDSP `attached` but silent | A malformed context/config wedges the DSP inside a fastrpc ioctl (no timeout in the driver); the stuck thread (D-state) then wedges docker/sshd | Power-cycle the board. Never ship experimental config values (e.g. the `soc_model` string `"QCS6490"` did exactly this — the config wants the **int** family id 35). Always `timeout` device tests + liveness check after each |
| CDSP `crashed` after load/unload loops | Repeated `contextCreateFromBinary`/`free` cycles exhaust the HTP SSR | Daemon loads contexts once at startup, keeps them resident |
| `GLIBC_2.38 not found` in the image | daemon built against board glibc 2.39 | build in `debian:bookworm` (§6.1) |
| daemon segfaults with `-static` | static glibc + fastrpc `dlopen` mismatch | dynamic bookworm build only |
| fastrpc `deviceCreate` fails in daemon | image-global `LD_PRELOAD` (mimalloc) | `env -u LD_PRELOAD` for the daemon only |
| `graphExecute` `0x1774` | tensors not declared as `QNN_DEFINITION_DEFINED` (dataFormat 1032) with exact context names/ids | use the specs from §5.4 |
| Deviant (but stable) outputs | daemon HTTP body-read loop overshot `Content-Length` | fixed: exact-length read + FNV-1a64 body checksum in the daemon log |
| SDK files vanishing mid-session | build-host `/tmp` is tmpfs | keep the SDK + work dir on persistent storage (`~/qairt`) |
| `is_soc_model_supported` always False / `--target_soc_model` rejected | the 2.37.1 converter's SoC table is empty in this build | use the Python API `CompileConfig(soc_details=...)` instead |
| generator `--config_file` changes produce identical binaries | the CLI ignores the `graphs`/`devices` sections | Python API `qairt.compile(...)` (see `tools/compile_htp.py`) |

---

## 11. What does NOT work here (with evidence)

- **SCRFD face detector on the NPU — blocked by firmware.** SCRFD's
  multi-scale anchor blending requires `Resize` (nearest 2× of content-derived
  56-ch prior maps). The QCS6490 firmware (v2.37.1 runtime, QNN interface cap
  2.32) does not provide the op package the compiler needs; `0x138d`
  (OP_PACKAGE_NOT_FOUND) is returned for *any* context containing a Resize —
  including a minimal single-op int8 Resize model. Nearest-2× upsample is
  non-local and cannot be expressed with Conv/ConvTranspose (ConvTranspose
  attempt measured maxdiff 1.51 vs ORT). Full bisection evidence and the
  toolchain findings are in `STATE.md` ("SCRFD → NPU: BLOCKED").
  Consequence: SCRFD stays on CPU (ORT).
- **QAIRT 2.42** context binaries — the firmware's QNN interface cap (2.32)
  predates them; stick to 2.37.1.
- **FLOAT contexts** — HTP on this SoC is INT8-only.

---

## 12. Where everything lives

### Repo (this one)
```
Dockerfile, docker/entrypoint-qnn.sh    image + startup
daemon/qnn_dsp_daemon.cpp               daemon source
daemon/qnn_dsp_daemon_bookworm          built aarch64 binary (bookworm glibc)
daemon/runtime/                         Radxa v2.37.1 HTP runtime (libQnnHtp.so, skels, stub)
daemon/models/*.bin                     INT8 context binaries
immich_ml/                              full package (only base.py + sessions/qnn.py differ)
upstream-diff.patch                     the 201-line stock diff (mergeable)
tools/patch_clip_conv1.py               CLIP conv1 → Gemm patch
tools/compile_htp.py                    DLC → SoC-pinned HTP context binary
tools/test_npu.py, qnn_*_test.cpp       unit/integration harnesses
docs/REPRODUCTION.md                    this file
STATE.md                                session continuity + SCRFD-block evidence
README.md                               overview
```

### Board (`/home/buga/`)
```
immich-ml-qnn/                          repo checkout (build + deployment dir)
immich-ml-qnn/artifacts/                context bins, ground refs, raws, board runtime
immich-ml-qnn/swap-backup/               pre-swap stack.env + docker-compose.yml
FFclone/immich/                         deployed Immich (compose project)
FFclone/immich/config/stack.env          IMMICH_ML_IMAGE=immich-ml-qnn:local
code/ffclone/                           ffclone git repo (branches: main = pre-QNN, qnn = NPU wiring)
```

### Build host
```
~/qairt/                                QAIRT 2.37.1 SDK (fresh) + work dir
~/qairt/work/                           DLCs, calib sets, refs, intermediate ONNX
```

### Durability
- ffclone: GitHub remote `git@github.com:swd543/ffclone.git` (main + qnn).
- this repo: GitHub remote `git@github.com:swd543/immich-ml-qnn.git`.
- a git bundle of this repo also lives on the board at
  `/home/buga/immich-ml-qnn/immich-ml-qnn.bundle`.
- Large regenerable binaries (models ~128 MB, runtime ~11 MB) are gitignored;
  they live in `daemon/{runtime,models}` on disk and in
  `/home/buga/immich-ml-qnn/artifacts/` on the board.
