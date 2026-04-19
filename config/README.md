# config

Centralised global parameter registry for the OpenReel offline pipeline.

Import `config.global_params` (or simply `config`) as the **very first
import** in any entry-point to apply all parameters before any other library
is loaded.

---

## Parameter groups

### Core Operational

| Parameter | Value | Why |
|---|---|---|
| `DEV_VERSION` | `true` | Flags the runtime as a developer/offline build so all subsystems operate in full-capability mode. |
| `AUTO_APPROVE_API_CALLS` | `true` | Eliminates interactive confirmation prompts for API operations in a localhost-only environment. |

### Security

| Parameter | Value | Why |
|---|---|---|
| `NO_STEGANOGRAPHY` | `true` | Prevents any library from silently embedding invisible data in generated pixels. |
| `NO_WATERMARKS` | `true` | Disables visible watermark overlays from diffusion libraries in trial/non-commercial modes. |
| `NO_SCREENSHOTS` | `true` | Prohibits any code path from capturing the screen or application window. |
| `NO_TELEMETRY` | `true` | Absolute zero-telemetry policy — no metrics, no crash beacons, no phone-home. |

### Filter / Logging

| Parameter | Value | Why |
|---|---|---|
| `DISABLE_SAFETY_CHECKER` | `true` | Disables the HF/diffusers NSFW classifier (requires network, blacks out outputs, irrelevant locally). |
| `DISABLE_CONTENT_MODERATION` | `true` | Removes the application-layer moderation filter. |
| `DISABLE_LOGGING` | `true` | Prevents runtime logs that could accumulate PII. |
| `DISABLE_AUDITING` | `true` | Audit trails have no compliance recipient in a local dev environment. |
| `DISABLE_TELEMETRY` | `true` | Suppresses the application's own instrumentation layer. |
| `DISABLE_LOGGING_3RD_PARTY_TRANSFERENCE` | `false` | Local generative-operation logging is kept active; only external transference is blocked. |

### CivitAI

| Parameter | Value | Why |
|---|---|---|
| `CIVITAI_NO_CONTENT_RESTRICTIONS` | `true` | Ensures the API returns the full model catalog. |
| `CIVITAI_SHOW_NSFW` | `true` | Most models are tagged NSFW; hiding them makes the majority of the library invisible. |
| `CIVITAI_SHOW_ALL_CONTENT_TYPES` | `true` | Shows checkpoints, LoRAs, embeddings, VAEs, upscalers, and all other asset types. |
| `CIVITAI_DISABLE_CONTENT_FILTER` | `true` | Disables CivitAI's server-side editorial filter. |

### InsightFace / FaceXLib cache directories

| Parameter | Default value | Why |
|---|---|---|
| `INSIGHTFACE_ROOT` | `./models/insightface` | Pins InsightFace model root to the project tree; prevents silent download to `~/.insightface`. |
| `INSIGHTFACE_HOME` | `./models/extensions/insightface/models` | Secondary lookup path for the Python extension wrapper. |
| `FACEXLIB_CACHE` | `./models/extensions/facexlib/weights` | Pins FaceXLib weights download path; prevents network fetch if weights are pre-populated. |

### Random seeds

| Parameter | Value | Why |
|---|---|---|
| `PYTHON_RANDOM_SEED` | `42` | Seeds `random.seed()` for reproducible shuffling/sampling. |
| `PYTORCH_RANDOM_SEED` | `42` | Seeds `torch.manual_seed()` and `torch.cuda.manual_seed_all()`. |
| `TENSORFLOW_RANDOM_SEED` | `42` | Seeds `tf.random.set_seed()` for TF-based extensions. |

### Dependency network-call suppression

| Parameter | Value | Why |
|---|---|---|
| `NO_ALBUMENTATIONS_UPDATE` | `1` | Skips the albumentations PyPI update-check HTTP request on every import. |
| `TQDM_DISABLE` | `1` | Disables tqdm progress bars application-wide. |
| `PANDAS_NO_REMOTE_IO` | `1` | Signals pandas to never attempt remote URL resolution. |

### LoRA paths

| Parameter | Default value | Why |
|---|---|---|
| `WAN2_LORAS` | `./models/lora/motion_loras` | Root for Wan2.2 T2V/I2V motion LoRA adapters. |
| `COGVIDEO_LORAS` | `./models/lora/motion_loras/cogvideo` | Dedicated sub-directory for CogVideoX LoRAs. |
| `HUNYUANVIDEO_LORAS` | `./models/lora/motion_loras/hunyuanvideo` | Dedicated path for HunyuanVideo LoRA adapters. |

### Compliance / network isolation

| Parameter | Value | Why |
|---|---|---|
| `LOCALHOST_ONLY` | `true` | Socket-level egress guard; rejects non-localhost connections. |
| `LOCAL_FILES_ONLY` | `true` | Model loaders reject `http://`, `s3://`, `hf://` URIs. |
| `HF_HUB_OFFLINE` | `1` | `huggingface_hub` never contacts hf.co. |
| `TRANSFORMERS_OFFLINE` | `1` | `transformers` adds its own offline guard. |
| `HF_DATASETS_OFFLINE` | `1` | `datasets` library stays offline. |
| `HF_OFFLINE_MODEL_DIR` | `C:/UI/Experimental-UI_Reit/models/hf_offline` | Local HuggingFace model cache root (overridable via env var). |

### Warning silencing

| Parameter | Value | Why |
|---|---|---|
| `HF_WARNINGS_SILENCED` | `true` | Silences expected HuggingFace `UserWarning` / `FutureWarning` in offline mode. |
| `REPO_AUTO_SILENCE_WARNINGS` | `true` | Silences deprecation chatter from third-party extensions and model repos. |

---

## Usage

### Apply all parameters at startup

```python
# server/flux_server.py  — first import in the file
import config.global_params  # sets ALL env vars, seeds RNGs, silences warnings
```

### Read a constant in application code

```python
from config import LOCALHOST_ONLY, DISABLE_SAFETY_CHECKER, PYTORCH_RANDOM_SEED

if DISABLE_SAFETY_CHECKER:
    pipeline = FluxPipeline(safety_checker=None)
```

### Override a value without editing source

```
# Windows
set DEV_VERSION=false
set PYTORCH_RANDOM_SEED=1337

# Linux / macOS
export DEV_VERSION=false
export PYTORCH_RANDOM_SEED=1337
```

All values are set via `os.environ.setdefault()`, so environment variables
defined before process startup always win.

---

## Files

| File | Purpose |
|---|---|
| `global_params.py` | Defines all parameters; sets env vars, seeds, warning filters |
| `__init__.py` | Re-exports all constants for convenient `from config import …` usage |
| `README.md` | This file |
