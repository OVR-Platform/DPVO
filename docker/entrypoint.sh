#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "No command provided. Try: python run.py --help" >&2
    exit 1
fi

exec "$@"
