# Immich ML with Qualcomm NPU (QCS6490) support.
#
# Base: the stock immich-machine-learning image (exact running build on the
# board: v3.1.0, image tag "release", Debian bookworm). We overlay:
#   * patched immich_ml package (QnnSession + routing hook in models/base.py)
#   * qnn-dsp-daemon (QNN C API inference server, aarch64) — COMPILED IN-IMAGE
#     from daemon/qnn_dsp_daemon.cpp against bookworm glibc 2.36 (staged QNN
#     headers) so the binary in the image always matches the committed source
#   * Qualcomm HTP runtime (libQnnHtp.so + skel + stub, QAIRT 2.37 generation)
#   * INT8 context binaries (CLIP ViT-B/32 + ArcFace w600k_r50 + SCRFD-2.5G detection)
#
# Build (on the board, one command — BuildKit builds the daemon stage first):
#   docker build --build-arg QNN_COMMIT="$(git rev-parse HEAD)" -t immich-ml-qnn:local .
#
# The daemon stage needs the staged (gitignored) QNN headers: build-headers/QNN
# (tools/stage_image_assets.sh). A standalone board-test binary
# (daemon/qnn_dsp_daemon_bookworm) is no longer required for the image build —
# only for running the daemon outside a container (probe contexts, port 8092).

# Verified production base: Immich ML v3.1.0. Pin the digest so a rebuild does
# not silently pick up a different moving `release` image.
# Declared before the first FROM (BuildKit requires it for FROM resolution).
ARG IMMICH_ML_BASE=ghcr.io/immich-app/immich-machine-learning@sha256:5a0839dc5303cd7215bcd2180a26aed3af41675aefb3e75e5157e9f10ad16e6e

# ---------- stage: compile the NPU daemon (bookworm glibc) ------------------
# Pinned bookworm digest (same glibc 2.36 as the base image below); the g++
# apt layer is BuildKit-cached across rebuilds.
FROM debian:bookworm@sha256:6ebd97fa83deb272194a2cf015b3d26a4d538e9ad3a7a79d544c8af5b0a01443 AS daemon-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /out
WORKDIR /src
COPY daemon/qnn_dsp_daemon.cpp ./qnn_dsp_daemon.cpp
COPY build-headers/QNN ./build-headers/QNN
RUN g++ -O2 -std=c++17 -Wall -Wextra -Ibuild-headers qnn_dsp_daemon.cpp \
    -o /out/qnn_dsp_daemon -ldl -pthread

# ---------- final: production image -----------------------------------------
FROM ${IMMICH_ML_BASE}

# Libraries the QNN HTP stub requires that the stock image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libyaml-0-2 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Patched immich_ml package (only base.py + sessions/qnn.py differ from stock).
COPY immich_ml /usr/src/immich_ml

# NPU daemon (compiled in the daemon-build stage above) + HTP runtime +
# context binaries.
COPY --from=daemon-build /out/qnn_dsp_daemon /opt/qnn/qnn_dsp_daemon
COPY daemon/runtime/ /opt/qnn/runtime/
COPY daemon/models/ /opt/qnn/models/
COPY docker/entrypoint-qnn.sh /entrypoint-qnn.sh

RUN chmod +x /entrypoint-qnn.sh /opt/qnn/qnn_dsp_daemon \
    && ln -sf /opt/qnn/qnn_dsp_daemon /usr/local/bin/qnn-dsp-daemon

ENV IMMICH_ML_QNN_PORT=8089

# Provenance: build with `--build-arg QNN_COMMIT=<git-rev>` so the running
# image can be mapped back to the source tag/commit.
ARG QNN_COMMIT=unknown
LABEL org.opencontainers.image.revision="$QNN_COMMIT" \
      org.opencontainers.image.version="immich-ml-qnn-3models" \
      immich-ml-qnn.npu="clip,arcface,scrfd"

ENTRYPOINT ["tini", "--", "/entrypoint-qnn.sh"]
