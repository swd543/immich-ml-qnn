#!/usr/bin/env bash
# Stage generated/proprietary inputs required by Dockerfile. They intentionally
# are not tracked in this public repository: the Qualcomm runtime is governed
# by its SDK license and the context binaries are large, regenerable outputs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/stage_image_assets.sh --sdk <QAIRT-2.37.1.250807-root> \
    --clip <clipr37_6490.bin> --arcface <arcface37v6_6490.bin> \
    [--runtime-root <board-runtime-root>]

Stages the exact Docker build inputs:
  build-headers/QNN/                 QNN headers for the bookworm daemon build
  daemon/runtime/{libQnnHtp.so,libQnnHtpV68Stub.so,libQnnHtpV68Skel.so}
  daemon/models/{clipr37_6490.bin,arcface37v6_6490.bin}

--sdk supplies headers and, by default, the runtime. It must be QAIRT
2.37.1.250807. --runtime-root optionally replaces the runtime source with the
board-copied SDK layout, for example:
  /home/buga/immich-ml-qnn/artifacts/board-runtime

The script never downloads, commits, or redistributes proprietary artifacts.
Run tools/verify_image_assets.sh before docker build.
EOF
}

sdk= clip= arcface= runtime_root=
while (($#)); do
  case "$1" in
    --sdk) sdk=${2:?}; shift 2 ;;
    --clip) clip=${2:?}; shift 2 ;;
    --arcface) arcface=${2:?}; shift 2 ;;
    --runtime-root) runtime_root=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $sdk && -n $clip && -n $arcface ]] || { usage >&2; exit 2; }
[[ -d "$sdk/include/QNN" ]] || { echo "missing SDK headers: $sdk/include/QNN" >&2; exit 1; }
[[ -f $clip ]] || { echo "missing CLIP context: $clip" >&2; exit 1; }
[[ -f $arcface ]] || { echo "missing ArcFace context: $arcface" >&2; exit 1; }
runtime_root=${runtime_root:-$sdk}
host_lib="$runtime_root/lib/aarch64-ubuntu-gcc9.4"
dsp_lib="$runtime_root/lib/hexagon-v68/unsigned"

for f in "$host_lib/libQnnHtp.so" "$host_lib/libQnnHtpV68Stub.so" "$dsp_lib/libQnnHtpV68Skel.so"; do
  [[ -f $f ]] || { echo "missing runtime component: $f" >&2; exit 1; }
done

rm -rf build-headers/QNN
mkdir -p build-headers daemon/runtime daemon/models
cp -a "$sdk/include/QNN" build-headers/QNN
install -m 0644 "$host_lib/libQnnHtp.so" daemon/runtime/
install -m 0644 "$host_lib/libQnnHtpV68Stub.so" daemon/runtime/
install -m 0644 "$dsp_lib/libQnnHtpV68Skel.so" daemon/runtime/
install -m 0644 "$clip" daemon/models/clipr37_6490.bin
install -m 0644 "$arcface" daemon/models/arcface37v6_6490.bin

printf '%s\n' 'staged headers, HTP runtime, and context binaries.'
printf '%s\n' 'Next: build daemon/qnn_dsp_daemon_bookworm, then run tools/verify_image_assets.sh.'
