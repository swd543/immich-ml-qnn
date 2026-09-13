#!/usr/bin/env bash
# Fail early with actionable messages before Dockerfile COPY errors.
set -euo pipefail

# Required for the IMAGE build (the Dockerfile compiles the daemon itself
# from daemon/qnn_dsp_daemon.cpp + these staged headers).
required=(
  daemon/qnn_dsp_daemon.cpp
  build-headers/QNN/QnnInterface.h
  daemon/runtime/libQnnHtp.so
  daemon/runtime/libQnnHtpV68Stub.so
  daemon/runtime/libQnnHtpV68Skel.so
  daemon/models/clipr37_6490.bin
  daemon/models/arcface37v6_6490.bin
  daemon/models/scrfd_6490_v2.bin
)
# Optional: standalone board-test binary (daemon outside a container, port
# 8092 probes). Not needed for the image build.
optional=(
  daemon/qnn_dsp_daemon_bookworm
)
missing=0
for path in "${required[@]}"; do
  if [[ ! -f $path ]]; then
    printf 'missing: %s\n' "$path" >&2
    missing=1
  fi
done
for path in "${optional[@]}"; do
  [[ -f $path ]] || printf 'optional (board probes only): %s absent\n' "$path" >&2
done
if ((missing)); then
  cat >&2 <<'EOF'

Run tools/stage_image_assets.sh first (stages the QNN headers, HTP runtime
and context binaries). The daemon binary itself is compiled in-image by the
Dockerfile's daemon-build stage; a standalone daemon/qnn_dsp_daemon_bookworm
(for board-side probes) can be built with:
  docker run --rm -v "$PWD":/src -w /src debian:bookworm bash -c \
    'apt-get update && apt-get install -y --no-install-recommends g++ && \
     g++ -O2 -std=c++17 -Wall -Ibuild-headers daemon/qnn_dsp_daemon.cpp \
       -o daemon/qnn_dsp_daemon_bookworm -ldl -pthread'
EOF
  exit 1
fi

sha256sum --check --strict docs/ARTIFACT_MANIFEST.sha256
printf 'image assets ready:\n'
for path in "${required[@]}"; do
  printf '  %s\n' "$path"
done