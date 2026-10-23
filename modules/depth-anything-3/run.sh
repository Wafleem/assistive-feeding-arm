#!/usr/bin/env bash
# Module entrypoint for viam-server.
set -euo pipefail
cd "$(dirname "$0")"
[ -f viam-env/.installed ] || ./setup.sh
export DA3_LOG_LEVEL="${DA3_LOG_LEVEL:-WARN}"
exec viam-env/bin/python -m da3_depth "$@"
