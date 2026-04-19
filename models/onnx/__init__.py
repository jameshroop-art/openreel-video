"""
models/onnx/__init__.py
-----------------------
Thin shim that re-exports the ONNX package from the **existing local
installation** at the path stored in the environment variable
``ONNX_MODEL_DIR`` (defaults to the canonical Windows development path
``C:/UI/Experimental-UI_Reit/models/onnx``).

Compliance rules (always enforced, regardless of caller):
  • local_files_only = True   – no model assets are fetched from the internet
  • localhost_only   = True   – the ONNX runtime is never exposed over a
                                network interface by this shim
"""

from __future__ import annotations

import os
import sys

# ── 1. Resolve the existing local ONNX installation path ──────────────────
_DEFAULT_LOCAL_PATH = os.path.normpath(
    os.environ.get(
        "ONNX_MODEL_DIR",
        r"C:/UI/Experimental-UI_Reit/models/onnx",
    )
)

# Expose the resolved path so callers can inspect it without importing onnx.
LOCAL_ONNX_PATH: str = _DEFAULT_LOCAL_PATH

# ── 2. Inject the local path at the *front* of sys.path so that
#       ``import onnx`` resolves to the local installation first. ──────────
if _DEFAULT_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _DEFAULT_LOCAL_PATH)

# ── 3. Compliance guards ──────────────────────────────────────────────────
#   These module-level constants are read by hub.py and serialization.py.
LOCAL_FILES_ONLY: bool = True
LOCALHOST_ONLY: bool = True

# ── 4. Forward all public ONNX symbols from the local installation ─────────
try:
    import onnx as _onnx  # noqa: F401  (the local copy injected above)
    from onnx import *  # noqa: F401, F403
    from onnx import __version__, __version__ as version  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        f"Could not import 'onnx' from the local path '{_DEFAULT_LOCAL_PATH}'. "
        "Ensure the ONNX package is installed at that location, or override "
        "the path via the ONNX_MODEL_DIR environment variable."
    ) from _exc
