# Immich ML with Qualcomm NPU (QCS6490) support.
#
# Base: the stock immich-machine-learning image (exact running build on the
# board: v3.1.0, image tag "release", Debian bookworm). We overlay:
#   * patched immich_ml package (QnnSession + routing hook in models/base.py)
#   * qnn-dsp-daemon (QNN C API inference server, aarch64, built against
#     bookworm glibc 2.36 so it runs in this image)
#   * Qualcomm HTP runtime (libQnnHtp.so + skel + stub, QAIRT 2.37 generation)
#   * INT8 context binaries (CLIP ViT-B/32 + ArcFace w600k_r50)
#
# Build (on the board):
#   docker build -t immich-ml-qnn:local .

ARG IMMICH_ML_BASE=ghcr.io/immich-app/immich-machine-learning:release
FROM ${IMMICH_ML_BASE}

# Libraries the QNN HTP stub requires that the stock image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libyaml-0-2 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Patched immich_ml package (only base.py + sessions/qnn.py differ from stock).
COPY immich_ml /usr/src/immich_ml

# NPU daemon (bookworm build) + HTP runtime + context binaries.
COPY daemon/qnn_dsp_daemon_bookworm /opt/qnn/qnn_dsp_daemon
COPY daemon/runtime/ /opt/qnn/runtime/
COPY daemon/models/ /opt/qnn/models/
COPY docker/entrypoint-qnn.sh /entrypoint-qnn.sh

RUN chmod +x /entrypoint-qnn.sh /opt/qnn/qnn_dsp_daemon \
    && ln -sf /opt/qnn/qnn_dsp_daemon /usr/local/bin/qnn-dsp-daemon

ENV IMMICH_ML_QNN_PORT=8089

ENTRYPOINT ["tini", "--", "/entrypoint-qnn.sh"]
