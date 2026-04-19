"""
models/hf_offline/__init__.py
-----------------------------
Fully-offline HuggingFace configuration shim.

Importing this module (or its sub-modules) enforces that **no network
request is ever made** by the ``transformers``, ``huggingface_hub``, or
``datasets`` libraries.  Model keys and API tokens are not required; only
the local model files on disk are used.

Offline behaviour (auto_approve=NO_DENIAL)
------------------------------------------
``HF_HUB_OFFLINE=1``        — huggingface_hub never contacts hf.co
``TRANSFORMERS_OFFLINE=1``  — transformers respects HF_HUB_OFFLINE and adds
                               its own guard
``HF_DATASETS_OFFLINE=1``   — datasets library also stays offline

These variables are written to ``os.environ`` before any HuggingFace
package is imported so the guards are in place from the very first call.

Local model directory
---------------------
All local model files are expected to live under ``HF_OFFLINE_MODEL_DIR``
(default ``C:/UI/Experimental-UI_Reit/models/hf_offline``).  Override
with the environment variable of the same name.

``from_pretrained`` helper
--------------------------
Use ``from_pretrained(cls, name_or_path, **kwargs)`` exported from this
module instead of ``SomeClass.from_pretrained(...)`` to guarantee that
``local_files_only=True`` is always forwarded and that relative paths are
resolved under ``HF_OFFLINE_MODEL_DIR``.
"""

from __future__ import annotations

import os
import pathlib

# ---------------------------------------------------------------------------
# 1. Resolve local model directory
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_DIR = os.path.normpath(
    os.environ.get(
        "HF_OFFLINE_MODEL_DIR",
        r"C:/UI/Experimental-UI_Reit/models/hf_offline",
    )
)

#: Absolute path to the local HuggingFace model cache / mirror.
#: Set the ``HF_OFFLINE_MODEL_DIR`` environment variable to override.
HF_OFFLINE_MODEL_DIR: str = _DEFAULT_MODEL_DIR

# ---------------------------------------------------------------------------
# 2. Set offline env vars BEFORE any HuggingFace import
# ---------------------------------------------------------------------------
# Setting these here ensures they are active even if individual modules
# import transformers or huggingface_hub before the server starts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Also tell huggingface_hub where to find local cached files.
os.environ.setdefault("HF_HOME", _DEFAULT_MODEL_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _DEFAULT_MODEL_DIR)

# Compliance constants exposed to callers
LOCAL_FILES_ONLY: bool = True
OFFLINE: bool = True


# ---------------------------------------------------------------------------
# 3. Path resolver
# ---------------------------------------------------------------------------

def resolve(name_or_path: str | os.PathLike[str]) -> str:
    """Return an absolute path for a model name or relative path.

    * If *name_or_path* is already absolute, it is returned as-is.
    * If it is a relative path or a short model identifier (no OS separator),
      it is resolved under :data:`HF_OFFLINE_MODEL_DIR`.

    Parameters
    ----------
    name_or_path:
        A HuggingFace model id (e.g. ``"openai/clip-vit-base-patch32"``),
        a relative path, or an absolute path.

    Returns
    -------
    str
        The resolved absolute path as a string.
    """
    p = pathlib.Path(name_or_path)
    if p.is_absolute():
        return str(p)
    # Replace HuggingFace "/" separators with OS path separators for
    # model IDs like "org/model-name".
    relative = os.path.join(*str(name_or_path).split("/"))
    return str(pathlib.Path(HF_OFFLINE_MODEL_DIR) / relative)


# ---------------------------------------------------------------------------
# 4. Offline from_pretrained wrapper
# ---------------------------------------------------------------------------

def from_pretrained(cls: type, name_or_path: str | os.PathLike[str], **kwargs):
    """Offline-safe wrapper around ``cls.from_pretrained``.

    Always passes ``local_files_only=True`` and resolves the path under
    :data:`HF_OFFLINE_MODEL_DIR` when a relative path or model id is given.
    No API key or model key is required.

    Parameters
    ----------
    cls:
        A HuggingFace class that exposes a ``from_pretrained`` class method,
        e.g. ``CLIPTokenizer``, ``T5TokenizerFast``, ``FluxTransformer2DModel``.
    name_or_path:
        Model identifier or path (see :func:`resolve`).
    **kwargs:
        Forwarded to ``cls.from_pretrained``.  ``local_files_only`` is
        always forced to ``True`` and cannot be overridden by the caller.

    Returns
    -------
    object
        Whatever ``cls.from_pretrained(...)`` returns.
    """
    kwargs["local_files_only"] = True
    resolved = resolve(name_or_path)
    return cls.from_pretrained(resolved, **kwargs)
