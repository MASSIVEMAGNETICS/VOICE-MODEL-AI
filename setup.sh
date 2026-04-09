#!/usr/bin/env bash
# ================================================================
#  Voice Model Studio — One-Click Linux / macOS Installer
# ================================================================
set -euo pipefail

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[INFO]  ${RESET}$*"; }
ok()    { echo -e "${GREEN}[ OK ]  ${RESET}$*"; }
warn()  { echo -e "${YELLOW}[WARN]  ${RESET}$*"; }
err()   { echo -e "${RED}[ERR ]  ${RESET}$*" >&2; }

echo -e "\n${BOLD}================================================================"
echo -e "  VOICE MODEL STUDIO — One-Click Installer"
echo -e "================================================================${RESET}\n"

# ── Python ──────────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON=$(command -v "$cmd")
        break
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.10+ not found. Please install it first."
    echo "  Ubuntu/Debian: sudo apt install python3.11"
    echo "  macOS:         brew install python@3.11"
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    err "Python 3.10+ required. Found $PY_VER."
    exit 1
fi
ok "Python $PY_VER found at $PYTHON"

# ── ffmpeg ──────────────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found."
else
    warn "ffmpeg not found. Install it for full audio format support."
    if [[ "$(uname)" == "Darwin" ]]; then
        warn "  brew install ffmpeg"
    else
        warn "  sudo apt install ffmpeg   (or equivalent for your distro)"
    fi
fi

# ── Run Python installer ─────────────────────────────────────────────────────
info "Running Python installer …"
"$PYTHON" install.py "$@"

echo -e "\n${BOLD}================================================================"
echo -e "  Installation complete!  Run ./start.sh to launch the studio."
echo -e "================================================================${RESET}\n"
