"""
models/onnx/hub.py
------------------
Local-only ONNX Hub loader.

All model loading is restricted to the existing local installation at
``ONNX_MODEL_DIR`` (default ``C:/UI/Experimental-UI_Reit/models/onnx``).
Remote URLs are unconditionally blocked.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

from models.onnx import LOCAL_FILES_ONLY, LOCAL_ONNX_PATH


def _resolve(path: str | os.PathLike[str]) -> pathlib.Path:
    """Return an absolute Path, anchored to LOCAL_ONNX_PATH when relative."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path(LOCAL_ONNX_PATH) / p
    _reject_remote(str(p))
    return p


def _reject_remote(path: str) -> None:
    """Raise RuntimeError for any URL-like path when local_files_only is set."""
    if LOCAL_FILES_ONLY:
        lowered = path.lower()
        if lowered.startswith(("http://", "https://", "ftp://", "s3://", "gs://")):
            raise RuntimeError(
                f"Remote paths are blocked (local_files_only=True): {path!r}\n"
                f"Use a path relative to or under: {LOCAL_ONNX_PATH!r}"
            )


def load(
    path: str | os.PathLike[str],
    *,
    load_external_data: bool = True,
) -> Any:
    """Load an ONNX model from the local installation path.

    Parameters
    ----------
    path:
        Path to the ``.onnx`` file.  Relative paths are resolved under
        ``LOCAL_ONNX_PATH``.
    load_external_data:
        Whether to load external tensor data referenced by the model.
    """
    resolved = _resolve(path)
    try:
        import onnx as _onnx  # local copy
    except ImportError as exc:
        raise ImportError("The 'onnx' package is required.") from exc
    return _onnx.load(str(resolved), load_external_data=load_external_data)


def list_models(subdir: str = "") -> list[str]:
    """List ``.onnx`` files available under the local installation.

    Parameters
    ----------
    subdir:
        Optional subdirectory relative to ``LOCAL_ONNX_PATH`` to scan.
    """
    root = pathlib.Path(LOCAL_ONNX_PATH)
    if subdir:
        root = root / subdir
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.onnx"))
