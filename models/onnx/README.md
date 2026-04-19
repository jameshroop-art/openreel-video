# models/onnx

This directory is a **thin shim** that loads the ONNX package from the
**existing local installation** — no files are duplicated here.

## Existing installation path

| Platform | Default path |
|----------|--------------|
| Windows  | `C:\UI\Experimental-UI_Reit\models\onnx` |
| Other    | Set `ONNX_MODEL_DIR` environment variable |

## Compliance: `localhost_only` + `local_files_only`

| Rule | Enforcement |
|------|-------------|
| `local_files_only = True` | `hub.py` and `__init__.py` block all `http://`, `https://`, `ftp://`, `s3://`, and `gs://` URIs at import time. |
| `localhost_only = True` | No network socket or server is opened by this shim. ONNX Runtime itself must be configured separately to bind to `127.0.0.1` only. |

## Usage

```python
# Always resolves to the local installation
from models.onnx import hub

model = hub.load("backend/test/data/node/test_relu/model.onnx")

# Or override path via env var (e.g. Linux CI)
# export ONNX_MODEL_DIR=/opt/models/onnx
```

## Environment variable

```
ONNX_MODEL_DIR=C:\UI\Experimental-UI_Reit\models\onnx   # default (Windows)
```

Override this variable to point to any local mirror of the ONNX package.
Remote paths will always be rejected when `local_files_only=True`.

## What lives here

| File | Purpose |
|------|---------|
| `__init__.py` | Injects `ONNX_MODEL_DIR` into `sys.path`; re-exports the `onnx` namespace |
| `hub.py` | `load()` / `list_models()` restricted to local filesystem |
| `.gitkeep` | Keeps this directory tracked in git |
| `README.md` | This file |

Binary model assets (`.onnx`, `.pb`, `.onnx_data`, `.pyc`, `.so`) are **not**
committed to git — they live exclusively at `ONNX_MODEL_DIR`.
