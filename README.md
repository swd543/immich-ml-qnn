# immich-ml-qnn — Immich ML on the Qualcomm QCS6490 NPU

Runs Immich's machine-learning models (CLIP ViT-B/32 image embeddings + ArcFace
w600k_r50 face recognition) on the Radxa Dragon Q6A's Hexagon NPU (HTP), while
everything else (text encoder, SCRFD face detection, OCR) stays on the CPU
(ONNX Runtime) — the same split as stock Immich, with the two heaviest
per-image models offloaded to the NPU.

## Quickstart (full guide)

**`docs/REPRODUCTION.md`** is the end-to-end reproduction guide: toolchain
setup, exact model conversion (QAIRT 2.37.1 → INT8 → SoC-pinned context
binaries, with the CLIP conv1→Gemm patch), daemon build (bookworm glibc),
image build, production deployment (incl. the compose-v1 workaround),
verification checklist, and every failure mode with its fix.

## Architecture

```
immich (server)
   │  POST /predict  (multipart: image + entries JSON)
   ▼
immich-ml  (patched, this image)
   ├─ QnnSession (sessions/qnn.py) ── HTTP ──▶ qnn-dsp-daemon (C++, QNN C API)
   │      image tower (CLIP visual)                │  load: context binaries
   │      face recognition (ArcFace)               ▼
   │                                    /dev/fastrpc-cdsp → Hexagon DSP (HTP)
   └─ OrtSession (unchanged) ── CPU ── text tower / SCRFD detection / OCR
```

- **qnn-dsp-daemon**: small C++ HTTP server (single file, `daemon/`) that loads
  QNN context binaries once at startup and serves `POST /infer/<key>` (and
  `/infer/<key>/int8`). It does the float32→int8 quantization of inputs and
  int8→float32 dequantization of outputs (scales/offsets come from the context
  binary metadata), so the Python side only ever moves plain float32 bytes.
  `GET /health` reports which models are loaded.
- **QnnSession** (`immich_ml/sessions/qnn.py`): drop-in replacement for the
  ONNX Runtime session object. It exposes `get_inputs()` / `get_outputs()` /
  `run()` (duck-typed the same as `ort.Session`), so Immich's model code —
  including insightface's `ArcFaceONNX` wrapper — works unchanged.
- **Routing** (`immich_ml/models/base.py`): `_make_session()` returns a
  `QnnSession` **only** when the env var `IMMICH_ML_QNN_URL` is set **and**
  the model key is one of `clip` / `arcface`. With the env var unset the image
  behaves exactly like stock immich-ml (all models on ORT/CPU) — the fork is
  inert by default.

## What is NPU vs CPU

| Model                          | Default (stock) | This image (env set)   |
|--------------------------------|-----------------|------------------------|
| CLIP ViT-B/32 **visual**       | CPU (ORT)       | **NPU (HTP, INT8)**    |
| ArcFace w600k_r50 **recognition** | CPU (ORT)    | **NPU (HTP, INT8)**    |
| CLIP textual / SCRFD / OCR     | CPU (ORT)       | CPU (ORT), unchanged   |

## Files

```
Dockerfile                  image build (pinned Immich ML v3.1.0 digest)
daemon/qnn_dsp_daemon.cpp   daemon source (single file)
docker/entrypoint-qnn.sh    starts the daemon (if env set) then the ML server
immich_ml/                  patched package (only models/base.py + sessions/qnn.py differ)
tools/                      conversion, staging, verification and test tools

generated / gitignored image inputs (not shipped by this public repo):
build-headers/QNN/          QAIRT headers for the bookworm daemon build
daemon/qnn_dsp_daemon_bookworm  aarch64 binary built against Debian 12 glibc
daemon/runtime/             Qualcomm HTP runtime (3 files)
daemon/models/              generated INT8 contexts (clipr37_6490.bin, arcface37v6_6490.bin)
```

## Build (on the board)

A clean clone needs generated/proprietary image inputs staged before a Docker
build. After producing the two contexts, stage the headers/runtime/models on
the **build host** and copy them to the board exactly as documented in
**`docs/REPRODUCTION.md` §7.1**. Then, on the board:

```sh
cd /home/buga/immich-ml-qnn
docker run --rm -v "$PWD":/src -w /src debian:bookworm bash -c '
  apt-get update && apt-get install -y --no-install-recommends g++ &&
  g++ -O2 -std=c++17 -Wall -Ibuild-headers daemon/qnn_dsp_daemon.cpp \
    -o daemon/qnn_dsp_daemon_bookworm -ldl -pthread'
tools/verify_image_assets.sh
docker build -t immich-ml-qnn:local .
```

The daemon is built against Debian 12 (bookworm) glibc 2.36 so it runs inside
the pinned stock immich base image (also bookworm). The HTP stub needs
`libyaml-0-2` and `libatomic1`, which the Dockerfile installs. The staging
and verification scripts check the untracked header/runtime/context files
against `docs/ARTIFACT_MANIFEST.sha256`; they never download or commit
Qualcomm SDK artifacts.

## Run

The container needs the DSP device node, the fastrpc userspace libraries, and
the board's device tree. The HTP backend resolves its DSP skel from the
**current working directory**, which the entrypoint sets to `/opt/qnn/runtime`.

```sh
docker run -d --name immich-ml \
  -e IMMICH_ML_QNN_URL=http://127.0.0.1:8089 \
  --device /dev/fastrpc-cdsp \
  -v /proc/device-tree:/proc/device-tree:ro \
  -v /usr/lib/dsp:/usr/lib/dsp:ro \
  -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:/usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:ro \
  -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so:/usr/lib/aarch64-linux-gnu/libcdsprpc.so:ro \
  -v /home/buga/FFclone/immich/media/cache:/cache:ro \
  -p 3003:3003 immich-ml-qnn:local
```

- **Without** `IMMICH_ML_QNN_URL` the daemon is not started and every model
  runs on the CPU (stock behaviour).
- **With** it, the daemon starts, loads both context binaries (~1-2 s), and
  the CLIP-visual and ArcFace-recognition models route to the NPU.

## Verification (done on the board)

- **CLIP (NPU)**: cosine similarity vs CPU ORT = **0.91**; embedding norm
  1.000 (L2-normalised as expected).
- **ArcFace (NPU)**: cosine similarity vs CPU ORT = **0.95** on a real face.
  (The synthetic-gradient test image gave 0.77 because it is far outside the
  calibration data distribution — a real face is in-distribution and scores 0.95.)
- **End-to-end** `POST /predict` on a real photo returns a 512-d CLIP embedding
  and a face detection + 512-d ArcFace embedding, all produced with the NPU
  path active.
- Bit-exactness: the daemon's output md5 matches the ground-truth references
  produced by `qnn-net-run` on the board (CLIP `dc209b6b…`, ArcFace
  `90f9c96e…`).

## Notes / gotchas

- **LD_PRELOAD**: the base image sets `LD_PRELOAD=/usr/lib/libmimalloc.so.2`
  (a global allocator). That breaks the fastrpc transport, so the entrypoint
  starts the daemon with `env -u LD_PRELOAD`. The Python side keeps mimalloc.
- **glibc**: the daemon must be built against bookworm glibc (2.36). A binary
  built on the board (glibc 2.39) fails with `GLIBC_2.38 not found`.
- **HTP INT8 only**: the QCS6490 HTP backend only supports INT8. The models are
  quantised to INT8 with per-channel calibration (26 diverse calibration
  images). CLIP ViT-B/16 is *more* sensitive to INT8 quantisation than B/32,
  so B/32 is the right model for the NPU on this board.
- **CDSP stability**: repeatedly creating/destroying QNN contexts can exhaust
  the HTP SSR and crash the CDSP. The daemon loads each context once at
  startup and keeps it resident, so there is no context churn at request time.

## Production swap (board) — what was done

1. `docker build -t immich-ml-qnn:local .` (on the board).
2. Backups of `docker-compose.yml` + `stack.env` saved to
   `swap-backup/` in the same directory on the board.
3. `docker-compose.yml`: the `immich-ml` service gained
   `IMMICH_ML_QNN_URL`, the four fastrpc/device-tree mounts, and
   `devices: [/dev/fastrpc-cdsp]`.
4. `stack.env`: `IMMICH_ML_IMAGE=immich-ml-qnn:local`.
5. The service container was recreated.

### Rollback

```sh
# from /home/buga/immich-ml-qnn/swap-backup
cp stack.env /home/buga/FFclone/immich/config/stack.env
cp docker-compose.yml /home/buga/code/ffclone/infra/immich/docker-compose.yml
docker rm -f immich-ml
cd /home/buga/code/ffclone/infra/immich && ~/bin/immichctl compose up -d immich-ml
```

### Note: docker-compose v1 on this board

This board runs `docker-compose` v1 (Python). The current docker daemon no
longer returns a `ContainerConfig` field in `docker inspect <image>`, so
compose v1 **crashes when it tries to *create* a container**
(`KeyError: 'ContainerConfig'` in `get_container_data_volumes`). It still reads
running containers fine. The `immich-ml` container was therefore created with
`docker run` directly (with the standard compose labels so it stays part of the
`immich` project). If you need to recreate it later, use the same `docker run`
command as above, not `compose up`.
