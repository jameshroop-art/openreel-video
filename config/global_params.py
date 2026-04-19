"""
config/global_params.py
-----------------------
Centralised global parameter registry for the OpenReel offline pipeline.

Importing this module (ideally as the very first import in any entry-point)
applies every parameter in the table below:

  • Sets ``os.environ`` via ``setdefault()`` — explicit shell overrides win.
  • Seeds Python / PyTorch / TensorFlow RNGs.
  • Silences known-harmless upstream warnings.

All boolean parameters are exposed as Python constants so that application
code can branch on them without string-parsing env vars.

Parameter groups
----------------
  Core Operational        — DEV_VERSION, AUTO_APPROVE_API_CALLS
  Security                — NO_STEGANOGRAPHY, NO_WATERMARKS,
                            NO_SCREENSHOTS, NO_TELEMETRY
  Filter / Logging        — DISABLE_SAFETY_CHECKER,
                            DISABLE_CONTENT_MODERATION, DISABLE_LOGGING,
                            DISABLE_AUDITING, DISABLE_TELEMETRY,
                            DISABLE_LOGGING_3RD_PARTY_TRANSFERENCE
  CivitAI                 — CIVITAI_NO_CONTENT_RESTRICTIONS,
                            CIVITAI_SHOW_NSFW,
                            CIVITAI_SHOW_ALL_CONTENT_TYPES,
                            CIVITAI_DISABLE_CONTENT_FILTER
  InsightFace/FaceXLib     — INSIGHTFACE_ROOT, INSIGHTFACE_HOME,
                            FACEXLIB_CACHE
  Random seeds            — PYTHON_RANDOM_SEED, PYTORCH_RANDOM_SEED,
                            TENSORFLOW_RANDOM_SEED
  Dep. network suppression— NO_ALBUMENTATIONS_UPDATE, TQDM_DISABLE,
                            PANDAS_NO_REMOTE_IO
  LoRA paths              — WAN2_LORAS, COGVIDEO_LORAS,
                            HUNYUANVIDEO_LORAS
  Compliance              — LOCALHOST_ONLY, LOCAL_FILES_ONLY,
                            HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE,
                            HF_DATASETS_OFFLINE, HF_HOME,
                            HUGGINGFACE_HUB_CACHE, HF_OFFLINE_MODEL_DIR
  Warning silencing       — HF_WARNINGS_SILENCED,
                            REPO_AUTO_SILENCE_WARNINGS
"""

from __future__ import annotations

import os
import warnings

# ===========================================================================
# § 1  Core Operational
# ===========================================================================

# DEV_VERSION — flags runtime as a developer/offline build; enables full
# capability mode across pipeline selectors, UI unlocks, and install checks.
os.environ.setdefault("DEV_VERSION", "true")
DEV_VERSION: bool = os.environ["DEV_VERSION"].lower() == "true"

# AUTO_APPROVE_API_CALLS — eliminates interactive confirmation prompts for
# API operations.  Safe in a localhost-only environment where every call
# targets 127.0.0.1 or CivitAI and is already guarded at the socket level.
os.environ.setdefault("AUTO_APPROVE_API_CALLS", "true")
AUTO_APPROVE_API_CALLS: bool = os.environ["AUTO_APPROVE_API_CALLS"].lower() == "true"

# ===========================================================================
# § 2  Security
# ===========================================================================

# NO_STEGANOGRAPHY — prevents any library or pipeline from embedding
# invisible data (watermarks, provenance bits, model fingerprints) in outputs.
os.environ.setdefault("NO_STEGANOGRAPHY", "true")
NO_STEGANOGRAPHY: bool = os.environ["NO_STEGANOGRAPHY"].lower() == "true"

# NO_WATERMARKS — disables visible watermark overlays from diffusion
# libraries operating in non-commercial or trial modes.
os.environ.setdefault("NO_WATERMARKS", "true")
NO_WATERMARKS: bool = os.environ["NO_WATERMARKS"].lower() == "true"

# NO_SCREENSHOTS — categorically prohibits any code path from capturing the
# screen or application window (telemetry, crash reporters, UI harvest).
os.environ.setdefault("NO_SCREENSHOTS", "true")
NO_SCREENSHOTS: bool = os.environ["NO_SCREENSHOTS"].lower() == "true"

# NO_TELEMETRY — absolute zero-telemetry policy; no usage metrics, no crash
# beacons, no phone-home of any kind.
os.environ.setdefault("NO_TELEMETRY", "true")
NO_TELEMETRY: bool = os.environ["NO_TELEMETRY"].lower() == "true"

# ===========================================================================
# § 3  Filter / Logging
# ===========================================================================

# DISABLE_SAFETY_CHECKER — disables the HuggingFace/diffusers NSFW safety
# classifier (requires network access, blacks out outputs, irrelevant locally).
os.environ.setdefault("DISABLE_SAFETY_CHECKER", "true")
DISABLE_SAFETY_CHECKER: bool = os.environ["DISABLE_SAFETY_CHECKER"].lower() == "true"

# DISABLE_CONTENT_MODERATION — removes the application-layer moderation
# filter that would otherwise intercept prompts or outputs.
os.environ.setdefault("DISABLE_CONTENT_MODERATION", "true")
DISABLE_CONTENT_MODERATION: bool = os.environ["DISABLE_CONTENT_MODERATION"].lower() == "true"

# DISABLE_LOGGING — prevents the application from writing runtime logs that
# could accumulate PII (prompts, filenames, inference parameters) to disk.
os.environ.setdefault("DISABLE_LOGGING", "true")
DISABLE_LOGGING: bool = os.environ["DISABLE_LOGGING"].lower() == "true"

# DISABLE_AUDITING — audit trails have no legal/compliance recipient in a
# local dev environment; disabling saves disk space and avoids recording prompts.
os.environ.setdefault("DISABLE_AUDITING", "true")
DISABLE_AUDITING: bool = os.environ["DISABLE_AUDITING"].lower() == "true"

# DISABLE_TELEMETRY — suppresses the application's own instrumentation layer.
# Combined with NO_TELEMETRY, ensures both app and dependencies transmit nothing.
os.environ.setdefault("DISABLE_TELEMETRY", "true")
DISABLE_TELEMETRY: bool = os.environ["DISABLE_TELEMETRY"].lower() == "true"

# DISABLE_LOGGING_3RD_PARTY_TRANSFERENCE — false: local generative-operation
# logging is intentionally kept active for reproducibility and debugging.
# Only external log transference is blocked.
os.environ.setdefault("DISABLE_LOGGING_3RD_PARTY_TRANSFERENCE", "false")
DISABLE_LOGGING_3RD_PARTY_TRANSFERENCE: bool = (
    os.environ["DISABLE_LOGGING_3RD_PARTY_TRANSFERENCE"].lower() == "true"
)

# ===========================================================================
# § 4  CivitAI
# ===========================================================================

# CIVITAI_NO_CONTENT_RESTRICTIONS — ensures the API returns the full model
# catalog without silently omitting results when content restrictions are active.
os.environ.setdefault("CIVITAI_NO_CONTENT_RESTRICTIONS", "true")
CIVITAI_NO_CONTENT_RESTRICTIONS: bool = (
    os.environ["CIVITAI_NO_CONTENT_RESTRICTIONS"].lower() == "true"
)

# CIVITAI_SHOW_NSFW — most artist and character models are tagged NSFW;
# hiding them would make the majority of the model library invisible.
os.environ.setdefault("CIVITAI_SHOW_NSFW", "true")
CIVITAI_SHOW_NSFW: bool = os.environ["CIVITAI_SHOW_NSFW"].lower() == "true"

# CIVITAI_SHOW_ALL_CONTENT_TYPES — shows checkpoints, LoRAs, embeddings,
# VAEs, upscalers, and all other asset types without category filtering.
os.environ.setdefault("CIVITAI_SHOW_ALL_CONTENT_TYPES", "true")
CIVITAI_SHOW_ALL_CONTENT_TYPES: bool = (
    os.environ["CIVITAI_SHOW_ALL_CONTENT_TYPES"].lower() == "true"
)

# CIVITAI_DISABLE_CONTENT_FILTER — disables CivitAI's server-side content
# filter so user-requested models are never suppressed by platform policy.
os.environ.setdefault("CIVITAI_DISABLE_CONTENT_FILTER", "true")
CIVITAI_DISABLE_CONTENT_FILTER: bool = (
    os.environ["CIVITAI_DISABLE_CONTENT_FILTER"].lower() == "true"
)

# ===========================================================================
# § 5  InsightFace / FaceXLib cache directories
# ===========================================================================

# INSIGHTFACE_ROOT — InsightFace C++/Python backends resolve model files
# (buffalo_l, antelopev2, etc.) from this root; pinned to the project tree.
os.environ.setdefault("INSIGHTFACE_ROOT", "./models/insightface")
INSIGHTFACE_ROOT: str = os.environ["INSIGHTFACE_ROOT"]

# INSIGHTFACE_HOME — secondary lookup path used by the Python extension
# wrapper; prevents fallback to ~/.insightface which would trigger downloads.
os.environ.setdefault(
    "INSIGHTFACE_HOME", "./models/extensions/insightface/models"
)
INSIGHTFACE_HOME: str = os.environ["INSIGHTFACE_HOME"]

# FACEXLIB_CACHE — FaceXLib downloads face-restoration weights (GFPGAN,
# CodeFormer deps) on first use unless this path is pre-populated.
os.environ.setdefault(
    "FACEXLIB_CACHE", "./models/extensions/facexlib/weights"
)
FACEXLIB_CACHE: str = os.environ["FACEXLIB_CACHE"]

# ===========================================================================
# § 6  Random seeds
# ===========================================================================

_PYTHON_SEED = int(os.environ.get("PYTHON_RANDOM_SEED", "42"))
_PYTORCH_SEED = int(os.environ.get("PYTORCH_RANDOM_SEED", "42"))
_TF_SEED = int(os.environ.get("TENSORFLOW_RANDOM_SEED", "42"))

os.environ.setdefault("PYTHON_RANDOM_SEED", str(_PYTHON_SEED))
os.environ.setdefault("PYTORCH_RANDOM_SEED", str(_PYTORCH_SEED))
os.environ.setdefault("TENSORFLOW_RANDOM_SEED", str(_TF_SEED))

PYTHON_RANDOM_SEED: int = _PYTHON_SEED
PYTORCH_RANDOM_SEED: int = _PYTORCH_SEED
TENSORFLOW_RANDOM_SEED: int = _TF_SEED

# Apply Python stdlib seed immediately (always available).
import random as _random  # noqa: E402
_random.seed(_PYTHON_SEED)

# Apply PyTorch seed if the library is present.
try:
    import torch as _torch  # type: ignore
    _torch.manual_seed(_PYTORCH_SEED)
    _torch.cuda.manual_seed_all(_PYTORCH_SEED)
except ImportError:
    pass

# Apply TensorFlow seed if the library is present.
try:
    import tensorflow as _tf  # type: ignore
    _tf.random.set_seed(_TF_SEED)
except ImportError:
    pass

# ===========================================================================
# § 7  Dependency network-call suppression
# ===========================================================================

# NO_ALBUMENTATIONS_UPDATE — albumentations >= 1.0.0 fires an HTTP request
# to PyPI on every import; setting 1 skips the check entirely.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
NO_ALBUMENTATIONS_UPDATE: bool = os.environ["NO_ALBUMENTATIONS_UPDATE"] != "0"

# TQDM_DISABLE — disables tqdm progress bars application-wide; they write to
# stderr and can probe notebook environments in ways that break offline mode.
os.environ.setdefault("TQDM_DISABLE", "1")
TQDM_DISABLE: bool = os.environ["TQDM_DISABLE"] != "0"

# PANDAS_NO_REMOTE_IO — signals that pandas must never open http://, ftp://,
# or s3:// URLs as DataFrames; avoids confusing timeout errors.
os.environ.setdefault("PANDAS_NO_REMOTE_IO", "1")
PANDAS_NO_REMOTE_IO: bool = os.environ["PANDAS_NO_REMOTE_IO"] != "0"

# ===========================================================================
# § 8  LoRA paths
# ===========================================================================

# WAN2_LORAS — root directory for Wan2.2 T2V / I2V motion LoRA adapters;
# shared by the pipeline runner, GUI browser, and installer.
os.environ.setdefault("WAN2_LORAS", "./models/lora/motion_loras")
WAN2_LORAS: str = os.environ["WAN2_LORAS"]

# COGVIDEO_LORAS — dedicated sub-directory for CogVideoX LoRAs; prevents
# namespace collisions with Wan2 motion LoRAs that share similar filenames.
os.environ.setdefault("COGVIDEO_LORAS", "./models/lora/motion_loras/cogvideo")
COGVIDEO_LORAS: str = os.environ["COGVIDEO_LORAS"]

# HUNYUANVIDEO_LORAS — dedicated path for HunyuanVideo LoRA adapters;
# architecturally distinct from Wan2/CogVideo and must not cross-load.
os.environ.setdefault(
    "HUNYUANVIDEO_LORAS", "./models/lora/motion_loras/hunyuanvideo"
)
HUNYUANVIDEO_LORAS: str = os.environ["HUNYUANVIDEO_LORAS"]

# ===========================================================================
# § 9  Compliance / network isolation
# ===========================================================================

# LOCALHOST_ONLY — primary socket-level egress guard; rejects every outbound
# connection that is not 127.0.0.1 / ::1 (plus the CivitAI whitelist).
os.environ.setdefault("LOCALHOST_ONLY", "true")
LOCALHOST_ONLY: bool = os.environ["LOCALHOST_ONLY"].lower() == "true"

# LOCAL_FILES_ONLY — model loaders and asset resolvers accept only filesystem
# paths; http://, ftp://, s3://, and hf:// URIs are always rejected.
os.environ.setdefault("LOCAL_FILES_ONLY", "true")
LOCAL_FILES_ONLY: bool = os.environ["LOCAL_FILES_ONLY"].lower() == "true"

# HuggingFace offline mode — no network requests from transformers,
# huggingface_hub, or datasets; model keys / API tokens not required.
# These complement models/hf_offline/__init__.py (both use setdefault).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_HF_OFFLINE_MODEL_DIR = os.path.normpath(
    os.environ.get(
        "HF_OFFLINE_MODEL_DIR",
        r"C:/UI/Experimental-UI_Reit/models/hf_offline",
    )
)
os.environ.setdefault("HF_OFFLINE_MODEL_DIR", _HF_OFFLINE_MODEL_DIR)
os.environ.setdefault("HF_HOME", _HF_OFFLINE_MODEL_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_OFFLINE_MODEL_DIR)

HF_HUB_OFFLINE: bool = True
TRANSFORMERS_OFFLINE: bool = True
HF_DATASETS_OFFLINE: bool = True
HF_OFFLINE_MODEL_DIR: str = _HF_OFFLINE_MODEL_DIR

# ===========================================================================
# § 10  Warning silencing
# ===========================================================================

# HF_WARNINGS_SILENCED — HuggingFace libraries emit UserWarning messages
# about model card metadata, deprecated arguments, and missing files in
# offline mode; harmless here, silenced to keep console readable.
os.environ.setdefault("HF_WARNINGS_SILENCED", "true")
HF_WARNINGS_SILENCED: bool = os.environ["HF_WARNINGS_SILENCED"].lower() == "true"

# REPO_AUTO_SILENCE_WARNINGS — suppresses deprecation and startup-chatter
# warnings from third-party extensions and model repos at import time.
os.environ.setdefault("REPO_AUTO_SILENCE_WARNINGS", "true")
REPO_AUTO_SILENCE_WARNINGS: bool = (
    os.environ["REPO_AUTO_SILENCE_WARNINGS"].lower() == "true"
)

if HF_WARNINGS_SILENCED:
    # Suppress the common HuggingFace warning categories.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"huggingface_hub")

if REPO_AUTO_SILENCE_WARNINGS:
    # Suppress deprecation noise from third-party repos (extensions, model libs).
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
