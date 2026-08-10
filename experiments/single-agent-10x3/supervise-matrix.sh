#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

mkdir -p "$SCRIPT_DIR/logs"
exec 9>"$SCRIPT_DIR/supervisor.lock"
if ! flock -n 9; then
  printf 'The experiment supervisor is already running.\n' >&2
  exit 3
fi

python3 - "$SCRIPT_DIR/supervisor.pid" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    stream.write(f"{os.getppid()}\n")
os.replace(temporary, path)
PY

cleanup() {
  python3 - "$SCRIPT_DIR/supervisor.pid" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).unlink(missing_ok=True)
PY
}
trap cleanup EXIT

"$SCRIPT_DIR/run-matrix.sh" all
