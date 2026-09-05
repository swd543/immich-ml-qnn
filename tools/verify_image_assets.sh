#!/usr/bin/env bash
# Fail early with actionable messages before Dockerfile COPY errors.
set -euo pipefail

required=(
  build-headers/QNN/QnnInterface.h
  daemon/qnn_dsp_daemon_bookworm
  daemon/runtime/libQnnHtp.so
  daemon/runtime/libQnnHtpV68Stub.so
  daemon/runtime/libQnnHtpV68Skel.so
  daemon/models/clipr37_6490.bin
  daemon/models/arcface37v6_6490.bin
)
missing=0
for path in "${required[@]}"; do
  if [[ ! -f $path ]]; then
    printf 'missing: %s\n' "$path" >&2
    missing=1
  fi
done
if ((missing)); then
  cat >&2 <<'EOF'

Run tools/stage_image_assets.sh first, then build the bookworm daemon:
  docker run --rm -v "$PWD":/src -w /src debian:bookworm bash -c \
    'apt-get update && apt-get install -y --no-install-recommends g++ && \
     g++ -O2 -std=c++17 -Wall -Ibuild-headers daemon/qnn_dsp_daemon.cpp \
       -o daemon/qnn_dsp_daemon_bookworm -ldl -pthread'
EOF
  exit 1
fi

printf 'image assets ready:\n'
for path in "${required[@]}"; do
  printf '  %s\n' "$path"
done
