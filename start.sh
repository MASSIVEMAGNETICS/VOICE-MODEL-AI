#!/usr/bin/env bash
set -euo pipefail

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Starting Voice Model Studio …"
exec "$PYTHON" app.py "$@"
