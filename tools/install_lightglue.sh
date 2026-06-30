#!/usr/bin/env bash
set -euo pipefail

python -m pip install git+https://github.com/cvg/LightGlue.git
python - <<'PY'
from lightglue import ALIKED, SIFT, LightGlue, SuperPoint

print("lightglue ok")
PY
