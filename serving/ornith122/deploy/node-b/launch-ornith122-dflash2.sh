#!/usr/bin/env bash
# Ornith-1.5-122B EXL3 on Node B. v5 serving fixes, 2026-11-02.
# Preserves native tool definitions and payload bytes, reads model sampling
# defaults, honors fleet controls, and cancels abandoned prefill at a chunk boundary.
# AR is the quality default; PROFILE=MTP2 / SPEC=mtp2 enables faster speculation.
# All modes use FP16 router projection and FP32 recurrent state by default.
# Usage: start | restart | stop | kill | status | logs | dry-run.
set -u
NAME=${NAME:-ornith122df}
PORT=${PORT:-18161}
HOST=${HOST:-0.0.0.0}
SPEC_FILE=${SPEC_FILE:-$HOME/.ornith122-spec}
# Priority: SPEC env > llmfleet PROFILE > ~/.ornith122-spec > ar
if [ -z "${SPEC:-}" ] && [ -n "${PROFILE:-}" ]; then
  case "$PROFILE" in
    DFlash2|dflash2|DFLASH2) SPEC=dflash2 ;;
    MTP2|mtp2)               SPEC=mtp2 ;;
    AR|ar)                   SPEC=ar ;;
  esac
fi
SPEC=${SPEC:-$([ -f "$SPEC_FILE" ] && cat "$SPEC_FILE" || echo ar)}
CTX=${CTX:-262144}
# DFlash2 proposes a fixed block; NDT caps the target verification window through
# EXL3_DRAFT_TOKENS. Matched 2k/192-token runs: depth 15 median 24.0 tok/s,
# depth 2 median 30.6 tok/s. MTP2 uses its own --draft argument.
NDT=${NDT:-15}
export PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export RECURRENT_STATE_DTYPE=${RECURRENT_STATE_DTYPE:-fp32}
# Repetition guard, fleet default for requests that never mention the field (see the BUNDLE
# note above): 1 = the engine defaults, shadow = detect and log only, 0 = off (v2 behaviour),
# or a JSON object of options. Item 22 runs guarded; a client can still override per request
# with repetition_guard, including an explicit false to switch it off for that call.
GUARD=${GUARD:-1}
SERVED_NAME=${SERVED_NAME:-ornith-1.5-122b-exl3}
ROOT=${ROOT:-/home/hycris/x3-exp/b-baseline-delivery-20261026/dflash2-opt-20261027-h1aSJ2/claude-fix-test-v1}
SRC=${SRC:-$ROOT/source}
ART=${ART:-$ROOT/artifacts/repguard-build1}
BUNDLE=${BUNDLE:-$ROOT/serving-quality-v5/bundle}
ENV_SCRIPT=${ENV_SCRIPT:-/home/hycris/x3-exp/b-native-candidate-prep-20261026-kqFr1i/bundle/env-b.sh}
ENV_SHA=${ENV_SHA:-a0311ff48b90739958dbbb67f7ae7a473fa1c6b3d16e78adc3347732388b54eb}
MODEL_DIR=${MODEL_DIR:-/home/hycris/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw}
DRAFT_DIR=${DRAFT_DIR:-/home/hycris/models/Ornith-1.5-122B-DFlash2}
LOG=${LOG:-$HOME/ornith122-dflash2.log}
PIDFILE=${PIDFILE:-$HOME/ornith122-dflash2.pid}
port_busy() { ss -ltn 2>/dev/null | grep -q ":$PORT "; }
managed_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid args
  pid=$(cat "$PIDFILE")
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  args=$(ps -ww -p "$pid" -o args= 2>/dev/null) || return 1
  [[ "$args" == *"ornith_serve.py"* && "$args" == *"--port $PORT "* ]] || return 1
  printf '%s' "$pid"
}
case "${1:-start}" in
  restart)
    "$0" stop && "$0" start
    ;;
  start)
    [ -d "$MODEL_DIR" ] || { echo "missing target: $MODEL_DIR" >&2; exit 2; }
    if [ "$SPEC" = dflash2 ]; then
      [ -f "$DRAFT_DIR/config.json" ] || { echo "missing drafter: $DRAFT_DIR" >&2; exit 2; }
      [[ "$NDT" =~ ^([1-9]|1[0-5])$ ]] || { echo "NDT must be 1..15" >&2; exit 4; }
    fi
    [ "$(sha256sum "$ENV_SCRIPT" | cut -d' ' -f1)" = "$ENV_SHA" ] || { echo "env script sha drift" >&2; exit 6; }
    port_busy && { echo "port $PORT already in use" >&2; exit 3; }
    case "$SPEC" in
      dflash2) MODE_ARGS="--draft-model $DRAFT_DIR" ;;
      mtp2)    MODE_ARGS="--draft ${MTP_DRAFT:-2}" ;;
      ar)      MODE_ARGS="--draft 0" ;;
      *) echo "unknown SPEC '$SPEC' (dflash2|mtp2|ar)" >&2; exit 4 ;;
    esac
    ROUTER_DET=0
    setsid nohup bash -c '
      set -eu
      . "$1"
      export EXL3_ROUTER_DET="'"$ROUTER_DET"'"
      export REPETITION_GUARD="'"$GUARD"'"
      if [ "'"$SPEC"'" = dflash2 ]; then
        export EXL3_DRAFT_TOKENS="'"$NDT"'"
      else
        unset EXL3_DRAFT_TOKENS
      fi
      export PYTHONDONTWRITEBYTECODE=1
      export FORK="'"$SRC"'"
      export PYTHONPATH="$FORK:'"$ART"':'"$BUNDLE"'"
      PY=$(command -v python)
      exec "$PY" -P "'"$BUNDLE"'/ornith_serve.py" \
        --model "'"$MODEL_DIR"'" '"$MODE_ARGS"' \
        --ctx '"$CTX"' --chunk-size "$PREFILL_CHUNK" --port '"$PORT"' --host '"$HOST"' \
        --served-name "'"$SERVED_NAME"'"
    ' _ "$ENV_SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started $NAME on :$PORT (pid $(cat "$PIDFILE")) ctx=$CTX spec=$SPEC router_det=$ROUTER_DET log=$LOG"
    ;;
  stop)
    if pid=$(managed_pid); then
      kill -TERM "$pid" 2>/dev/null
    elif port_busy; then
      echo "port $PORT is owned by an unmanaged process; left it alone" >&2; exit 5
    fi
    for _ in $(seq 1 30); do port_busy || break; sleep 1; done
    rm -f "$PIDFILE"; echo "stopped $NAME" ;;
  kill)
    if pid=$(managed_pid); then
      kill -KILL "$pid" 2>/dev/null
    elif port_busy; then
      echo "port $PORT is owned by an unmanaged process; left it alone" >&2; exit 5
    fi
    rm -f "$PIDFILE"; echo "killed $NAME" ;;
  status)
    printf '%s :%s ' "$NAME" "$PORT"
    if port_busy; then printf 'listening health=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")"; else echo 'stopped'; fi ;;
  logs) tail -n "${2:-80}" "$LOG" ;;
  dry-run)
    case "$SPEC" in
      dflash2) mode="--draft-model $DRAFT_DIR" ;;
      mtp2) mode="--draft ${MTP_DRAFT:-2}" ;;
      ar) mode="--draft 0" ;;
      *) echo "unknown SPEC '$SPEC'" >&2; exit 4 ;;
    esac
    router=0
    ( . "$ENV_SCRIPT" >/dev/null 2>&1
      printf 'PY=%s spec=%s port=%s router_det=%s ndt=%s chunk=%s threads=%s state=%s mode_args=%s\n' \
        "$(command -v python)" "$SPEC" "$PORT" "$router" "$NDT" "$PREFILL_CHUNK" "$OMP_NUM_THREADS" "$RECURRENT_STATE_DTYPE" "$mode"
    ) ;;
  *) echo "usage: $0 [start|stop|kill|status|logs|dry-run]" >&2; exit 1 ;;
esac
