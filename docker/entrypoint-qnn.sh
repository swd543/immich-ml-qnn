# Entrypoint for immich-ml-qnn: optionally starts the QNN NPU daemon, then
# runs the standard immich-ml server.
#
# Set IMMICH_ML_QNN_URL (e.g. http://127.0.0.1:8089) to activate the NPU path.
# Without it, this image behaves exactly like stock immich-ml (ORT/CPU).

set -e

QNN_PORT="${IMMICH_ML_QNN_PORT:-8089}"
DAEMON_PID=""
PY_PID=""

# Forward termination to whichever children exist, wait for the server, and
# always stop the daemon before exiting. Without this, the daemon survives as
# an orphan (and on SIGKILL-style stops is torn down mid-fastrpc-call — the
# documented suspect for CDSP wedges that only a power cycle clears).
cleanup() {
  rc=$1
  if [ -n "$PY_PID" ]; then kill "$PY_PID" 2>/dev/null || true; fi
  if [ -n "$DAEMON_PID" ]; then kill "$DAEMON_PID" 2>/dev/null || true; fi
  # Give the daemon its graceful-teardown window (context/device/backend free).
  if [ -n "$DAEMON_PID" ]; then wait "$DAEMON_PID" 2>/dev/null || true; fi
  exit "$rc"
}
trap 'cleanup 143' TERM
trap 'cleanup 130' INT

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
    --arcface-context /opt/qnn/models/arcface37v6_6490.bin \
    --scrfd-context /opt/qnn/models/scrfd_6490_v2.bin &
  DAEMON_PID=$!

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

# Do NOT exec: this shell must stay alive to forward signals and stop the
# daemon. Exit status = the ML server's.
python -m immich_ml &
PY_PID=$!
wait "$PY_PID"
rc=$?
if [ -n "$DAEMON_PID" ]; then
  kill "$DAEMON_PID" 2>/dev/null || true
  wait "$DAEMON_PID" 2>/dev/null || true
fi
exit "$rc"
