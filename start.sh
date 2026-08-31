#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec python -m uvicorn lightworkbench.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8331}"
