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
