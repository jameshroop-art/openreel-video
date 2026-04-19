# models/hf_offline

Fully-offline HuggingFace configuration shim — **no API keys or model keys required**.

## What this library does

Importing `models.hf_offline` configures the entire Python process to run
HuggingFace libraries in **offline mode** (`auto_approve=NO_DENIAL`):

| Environment variable | Value | Effect |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | `huggingface_hub` never contacts hf.co |
| `TRANSFORMERS_OFFLINE` | `1` | `transformers` adds its own offline guard |
| `HF_DATASETS_OFFLINE` | `1` | `datasets` stays offline |
| `HF_HOME` | `HF_OFFLINE_MODEL_DIR` | local model cache root |
| `HUGGINGFACE_HUB_CACHE` | `HF_OFFLINE_MODEL_DIR` | local blob cache root |

All variables are set via `os.environ.setdefault()` — **before** any
HuggingFace package is imported — so they apply to every subsequent
`from_pretrained` call in the process.

## Local model directory

| Platform | Default path |
|---|---|
| Windows | `C:\UI\Experimental-UI_Reit\models\hf_offline` |
| Other | Set `HF_OFFLINE_MODEL_DIR` environment variable |

Place your offline model snapshots here.  The expected layout mirrors the
HuggingFace Hub cache format:

```
hf_offline/
  FLUX.1-dev/
    tokenizer/            ← CLIP tokenizer files
    tokenizer_2/          ← T5 tokenizer files
  openai--clip-vit-large-patch14/
    ...
```

## Usage

### Import early in your entry point

```python
# server/flux_server.py  (top of file, before any HuggingFace import)
import models.hf_offline  # activates offline mode
```

### Offline-safe `from_pretrained` wrapper

```python
from models import hf_offline
from transformers import CLIPTokenizer

tokenizer = hf_offline.from_pretrained(CLIPTokenizer, "FLUX.1-dev/tokenizer")
# resolves to:  C:\UI\Experimental-UI_Reit\models\hf_offline\FLUX.1-dev\tokenizer
# always passes local_files_only=True — no token needed
```

Absolute paths bypass the model directory:

```python
tokenizer = hf_offline.from_pretrained(
    CLIPTokenizer,
    r"C:\UI\Experimental-UI_Reit\models\Flux\FLUX.1-dev\tokenizer",
)
```

### Path resolver

```python
path = hf_offline.resolve("FLUX.1-dev/tokenizer")
# -> "C:\UI\Experimental-UI_Reit\models\hf_offline\FLUX.1-dev\tokenizer"
```

## Environment variable

```
HF_OFFLINE_MODEL_DIR=C:\UI\Experimental-UI_Reit\models\hf_offline   # default (Windows)
```

Override to point to any local directory containing model snapshots.

## What lives here

| File | Purpose |
|---|---|
| `__init__.py` | Sets offline env vars; exposes `resolve()` and `from_pretrained()` |
| `README.md` | This file |
| `.gitkeep` | Keeps directory tracked in git |

Binary model assets are **not** committed — they live exclusively at
`HF_OFFLINE_MODEL_DIR`.
