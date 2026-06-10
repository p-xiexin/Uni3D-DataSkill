from __future__ import annotations

import atexit
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_PI3_ROOT = Path(__file__).resolve().parent / "fake_pi3"
THIRDPARTY_PI3_ROOT = REPO_ROOT / "thirdparty" / "Pi3"
_cleanup_registered = False


def install_fake_pi3() -> None:
    global _cleanup_registered
    if THIRDPARTY_PI3_ROOT.exists():
        shutil.rmtree(THIRDPARTY_PI3_ROOT)
    THIRDPARTY_PI3_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FAKE_PI3_ROOT, THIRDPARTY_PI3_ROOT)
    if not _cleanup_registered:
        atexit.register(remove_fake_pi3)
        _cleanup_registered = True


def remove_fake_pi3() -> None:
    if THIRDPARTY_PI3_ROOT.exists():
        shutil.rmtree(THIRDPARTY_PI3_ROOT)
