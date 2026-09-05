# Entrypoint for immich-ml-qnn: optionally starts the QNN NPU daemon, then
# runs the standard immich-ml server.
#
# Set IMMICH_ML_QNN_URL (e.g. http://127.0.0.1:8089) to activate the NPU path.
# Without it, this image behaves exactly like stock immich-ml (ORT/CPU).

set -e

QNN_PORT="${IMMICH_ML_QNN_PORT:-8089}"

if [ -n "${IMMICH_ML_QNN_URL:-}" ]; then
  # The HTP backend resolves the DSP skel (libQnnHtpV68Skel.so) from the
  # current working directory — the runtime directory must be cwd.
  cd /opt/qnn/runtime
  # LD_PRELOAD (mimalloc, set by the base image) breaks the fastrpc
  # transport; start the daemon without it.
  env -u LD_PRELOAD /opt/qnn/qnn_dsp_daemon \
    --backend ./libQnnHtp.so \
    --port "$QNN_PORT" \
    --bind 127.0.0.1 \
    --clip-context /opt/qnn/models/clipr37_6490.bin \
    --arcface-context /opt/qnn/models/arcface37v6_6490.bin &
  DAEMON_PID=$!
  trap 'kill "$DAEMON_PID" 2>/dev/null || true' TERM INT

  # Wait for the daemon (context binaries load in ~1-2 s).
  i=0
  while [ $i -lt 60 ]; do
    if python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:${QNN_PORT}/health', timeout=1)" 2>/dev/null; then
      echo "qnn-dsp-daemon ready (pid $DAEMON_PID)"
      break
    fi
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
      echo "qnn-dsp-daemon died during startup" >&2
      exit 1
    fi
    i=$((i + 1))
    sleep 0.5
  done
  if [ $i -ge 60 ]; then
    echo "qnn-dsp-daemon failed health check" >&2
    exit 1
  fi
fi

exec python -m immich_ml
