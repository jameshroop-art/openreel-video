"""
FLUX ONNX Local Inference Server
=================================
Runs at http://localhost:8080 and exposes three endpoints:

  POST /generate      — text-to-image (schnell / dev ONNX / dev-uncensored Q4)
  POST /edit          — image editing via FLUX.1-Kontext-dev-onnx
  POST /detect-faces  — face bounding-box detection (used by auto-reframe)

Model inventory
---------------
All paths below are relative to C:\\UI\\Experimental-UI_Reit\\models\\Flux\\

  FLUX.1-schnell-onnx/
    clip.opt/model.onnx
    t5.opt/backbone.onnx_data + model.onnx
    t5-fp8.opt/backbone.onnx_data + model.onnx
    transformer.opt/bf16|fp4|fp8/backbone.onnx_data + model.onnx
    vae.opt/model.onnx

  FLUX.1-dev-onnx/
    clip.opt/model.onnx
    t5.opt/backbone.onnx_data + model.onnx
    t5-fp8.opt/backbone.onnx_data + model.onnx
    transformer.opt/bf16|fp4|fp8/backbone.onnx_data + model.onnx
    vae.opt/model.onnx

  FLUX.1-Kontext-dev-onnx/
    clip.opt/model.onnx
    t5.opt/backbone.onnx_data + model.onnx
    t5-fp8.opt/backbone.onnx_data + model.onnx
    transformer.opt/bf16|fp4_svd32|fp8/backbone.onnx_data + model.onnx   <- fp4_svd32
    vae.opt/model.onnx
    vae_encoder.opt/model.onnx          <- required for /edit; currently empty

  FLUX.1-dev/
    tokenizer/                          <- CLIP tokenizer files
    tokenizer_2/                        <- T5 tokenizer files

  flux.1-dev-uncensored-q4/             <- Q4 safetensors transformer (diffusers format)
    config.json
    diffusion_pytorch_model.safetensors

  flux2-klein-4B-uncensored-text-encoder/
    qwen3-4b-abl-q4_0.gguf             <- optional GGUF replacement for T5

  Flux_Lustly.ai_Uncensored_nsfw_v1/
    flux_lustly-ai_v1.safetensors       <- LoRA; applies to all transformer variants

onnxruntime resolves backbone.onnx_data sidecars automatically.

Setup (Windows PowerShell)
--------------------------
  pip install fastapi uvicorn[standard] onnxruntime transformers safetensors pillow numpy sentencepiece

  # CUDA GPU (strongly recommended):
  pip install onnxruntime-gpu

  # For model="dev-uncensored" (Q4 torch transformer):
  pip install torch diffusers accelerate

  # For GGUF Qwen3-4B text encoder (FLUX_USE_GGUF_T5=1):
  pip install llama-cpp-python

  # For in-memory LoRA application to ONNX transformers:
  pip install onnx

Run
---
  python server/flux_server.py

Environment variables
---------------------
  FLUX_SCHNELL_DIR          default: ...\\FLUX.1-schnell-onnx
  FLUX_DEV_ONNX_DIR         default: ...\\FLUX.1-dev-onnx
  FLUX_KONTEXT_DIR          default: ...\\FLUX.1-Kontext-dev-onnx
  FLUX_DEV_DIR              default: ...\\FLUX.1-dev          (tokenizer source)
  FLUX_DEV_UNCENSORED_DIR   default: ...\\flux.1-dev-uncensored-q4

  FLUX_SCHNELL_PRECISION    bf16 | fp4 | fp8          default: fp8
  FLUX_DEV_ONNX_PRECISION   bf16 | fp4 | fp8          default: fp8
  FLUX_KONTEXT_PRECISION    bf16 | fp4_svd32 | fp8    default: fp8

  FLUX_USE_FP8_T5           1 -> use t5-fp8.opt        default: 0
  FLUX_GGUF_T5_PATH         path to .gguf file
  FLUX_USE_GGUF_T5          1 -> use GGUF encoder      default: 0

  FLUX_LORA_PATH            path to active LoRA .safetensors (used during generation)
  FLUX_LORA_SCALE           LoRA strength 0.0-2.0      default: 1.0
  FLUX_LORA_DIR             directory scanned by GET /loras
                            default: C:\\UI\\Experimental-UI_Reit\\models\\lora

  FLUX_REACTOR_DIR          directory scanned by GET /reactor-models
                            default: C:\\UI\\Experimental-UI_Reit\\models\\Reactorplus

  FLUX_REALESRGAN_DIR       directory scanned by GET /upscalers
                            default: C:\\UI\\Experimental-UI_Reit\\models\\realesrgan

  FLUX_SD_DIR               directory scanned by GET /sd-models
                            default: C:\\UI\\Experimental-UI_Reit\\models\\Stable-diffusion

  FLUX_WAN2_DIR             directory scanned by GET /wan2-models
                            default: C:\\UI\\Experimental-UI_Reit\\models\\WAN2-x

  FLUX_ZIMAGE_DIR           directory scanned by GET /zimage-models
                            default: C:\\UI\\Experimental-UI_Reit\\models\\ZImage

  FLUX_VAE_DIR              directory scanned by GET /vae-models
                            default: C:\\UI\\Experimental-UI_Reit\\models\\VAE

  FLUX_TEXT_ENC_DIR         directory scanned by GET /text-encoders
                            default: C:\\UI\\Experimental-UI_Reit\\models\\text_encoder
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_BASE = r"C:\UI\Experimental-UI_Reit\models\Flux"
_LORA_BASE = r"C:\UI\Experimental-UI_Reit\models\lora"
_REACTOR_BASE = r"C:\UI\Experimental-UI_Reit\models\Reactorplus"
_REALESRGAN_BASE = r"C:\UI\Experimental-UI_Reit\models\realesrgan"
_SD_BASE = r"C:\UI\Experimental-UI_Reit\models\Stable-diffusion"
_WAN2_BASE = r"C:\UI\Experimental-UI_Reit\models\WAN2-x"
_ZIMAGE_BASE = r"C:\UI\Experimental-UI_Reit\models\ZImage"
_VAE_BASE = r"C:\UI\Experimental-UI_Reit\models\VAE"
_TEXT_ENC_BASE = r"C:\UI\Experimental-UI_Reit\models\text_encoder"

# ---------------------------------------------------------------------------
# Configuration -- model directories
# ---------------------------------------------------------------------------
SCHNELL_DIR = Path(os.getenv("FLUX_SCHNELL_DIR", rf"{_BASE}\FLUX.1-schnell-onnx"))
DEV_ONNX_DIR = Path(os.getenv("FLUX_DEV_ONNX_DIR", rf"{_BASE}\FLUX.1-dev-onnx"))
KONTEXT_DIR = Path(os.getenv("FLUX_KONTEXT_DIR", rf"{_BASE}\FLUX.1-Kontext-dev-onnx"))
FLUX_DEV_DIR = Path(os.getenv("FLUX_DEV_DIR", rf"{_BASE}\FLUX.1-dev"))
DEV_UNCENSORED_DIR = Path(
    os.getenv("FLUX_DEV_UNCENSORED_DIR", rf"{_BASE}\flux.1-dev-uncensored-q4")
)

# ---------------------------------------------------------------------------
# Configuration -- transformer precision
# ---------------------------------------------------------------------------
_SCHNELL_PRECISIONS = {"bf16", "fp4", "fp8"}
_DEV_ONNX_PRECISIONS = {"bf16", "fp4", "fp8"}
_KONTEXT_PRECISIONS = {"bf16", "fp4_svd32", "fp8"}

SCHNELL_PRECISION = os.getenv("FLUX_SCHNELL_PRECISION", "fp8").lower()
DEV_ONNX_PRECISION = os.getenv("FLUX_DEV_ONNX_PRECISION", "fp8").lower()
KONTEXT_PRECISION = os.getenv("FLUX_KONTEXT_PRECISION", "fp8").lower()

for _prec, _valid, _var in (
    (SCHNELL_PRECISION, _SCHNELL_PRECISIONS, "FLUX_SCHNELL_PRECISION"),
    (DEV_ONNX_PRECISION, _DEV_ONNX_PRECISIONS, "FLUX_DEV_ONNX_PRECISION"),
    (KONTEXT_PRECISION, _KONTEXT_PRECISIONS, "FLUX_KONTEXT_PRECISION"),
):
    if _prec not in _valid:
        raise ValueError(f"{_var} must be one of {_valid}, got {_prec!r}")

# ---------------------------------------------------------------------------
# Configuration -- text encoder
# ---------------------------------------------------------------------------
USE_FP8_T5: bool = os.getenv("FLUX_USE_FP8_T5", "0").strip() == "1"
GGUF_T5_PATH = Path(
    os.getenv(
        "FLUX_GGUF_T5_PATH",
        rf"{_BASE}\flux2-klein-4B-uncensored-text-encoder\qwen3-4b-abl-q4_0.gguf",
    )
)
USE_GGUF_T5: bool = os.getenv("FLUX_USE_GGUF_T5", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Configuration -- LoRA
# ---------------------------------------------------------------------------
LORA_PATH = Path(
    os.getenv(
        "FLUX_LORA_PATH",
        rf"{_BASE}\Flux_Lustly.ai_Uncensored_nsfw_v1\flux_lustly-ai_v1.safetensors",
    )
)
LORA_SCALE: float = float(os.getenv("FLUX_LORA_SCALE", "1.0"))
LORA_DIR = Path(os.getenv("FLUX_LORA_DIR", _LORA_BASE))
REACTOR_DIR = Path(os.getenv("FLUX_REACTOR_DIR", _REACTOR_BASE))
REALESRGAN_DIR = Path(os.getenv("FLUX_REALESRGAN_DIR", _REALESRGAN_BASE))
SD_DIR = Path(os.getenv("FLUX_SD_DIR", _SD_BASE))
WAN2_DIR = Path(os.getenv("FLUX_WAN2_DIR", _WAN2_BASE))
ZIMAGE_DIR = Path(os.getenv("FLUX_ZIMAGE_DIR", _ZIMAGE_BASE))
VAE_DIR = Path(os.getenv("FLUX_VAE_DIR", _VAE_BASE))
TEXT_ENC_DIR = Path(os.getenv("FLUX_TEXT_ENC_DIR", _TEXT_ENC_BASE))

# ---------------------------------------------------------------------------
# FLUX VAE / latent constants
# ---------------------------------------------------------------------------
FLUX_VAE_SCALE = 0.3611   # vae.config.scaling_factor
FLUX_VAE_SHIFT = 0.1159   # vae.config.shift_factor
CLIP_MAX_LENGTH = 77
T5_MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _component(root: Path, name: str) -> Path:
    """<root>/<name>.opt/model.onnx"""
    return root / f"{name}.opt" / "model.onnx"


def _transformer(root: Path, precision: str) -> Path:
    return root / "transformer.opt" / precision / "model.onnx"


def _t5(root: Path) -> Path:
    folder = "t5-fp8.opt" if USE_FP8_T5 else "t5.opt"
    return root / folder / "model.onnx"


def _validate_model_paths(
    root: Path, label: str, precision: str, has_vae_enc: bool = False
) -> None:
    checks: list[Path] = [
        _component(root, "clip"),
        _t5(root),
        _transformer(root, precision),
        _component(root, "vae"),
    ]
    if has_vae_enc:
        checks.append(_component(root, "vae_encoder"))
    for p in checks:
        log.info("[%-12s] %s  %s", label, "OK     " if p.exists() else "MISSING", p)


# ---------------------------------------------------------------------------
# ONNX session cache
# ---------------------------------------------------------------------------
_sessions: dict[str, object] = {}


def _session(model_path: Path) -> object:
    """Return a cached OnnxRuntime InferenceSession."""
    key = str(model_path)
    if key not in _sessions:
        import onnxruntime as ort  # type: ignore

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        log.info("Loading ONNX: %s  providers=%s", model_path.name, providers)
        _sessions[key] = ort.InferenceSession(str(model_path), providers=providers)
    return _sessions[key]


# ---------------------------------------------------------------------------
# In-memory LoRA merging for ONNX transformer sessions
# ---------------------------------------------------------------------------
_lora_sessions: dict[str, object] = {}


def _normalize_weight_name(name: str) -> str:
    return name.lower().replace(".", "/").strip("/")


def _session_with_lora(model_path: Path) -> object:
    """
    Like _session() but merges matching LoRA deltas from LORA_PATH in-memory before
    creating the ORT session.  Name matching is best-effort (normalised path comparison).
    Falls back to a plain session if safetensors or onnx packages are absent.

    Note: The ORT-optimised (.opt) transformer may have renamed or fused weight nodes,
    resulting in a low match rate.  Use model="dev-uncensored" for reliable LoRA application
    via the native diffusers loader.
    """
    key = str(model_path)
    if not LORA_PATH.exists():
        return _session(model_path)

    lora_key = f"{key}::lora"
    if lora_key in _lora_sessions:
        return _lora_sessions[lora_key]

    try:
        from safetensors.numpy import load_file as st_load
    except ImportError:
        log.warning("safetensors not installed -- LoRA skipped. pip install safetensors")
        return _session(model_path)

    try:
        import onnx
        import onnx.numpy_helper as nph
    except ImportError:
        log.warning("onnx not installed -- LoRA skipped. pip install onnx")
        return _session(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    log.info("Merging LoRA %s into %s ...", LORA_PATH.name, model_path.name)
    lora_weights = st_load(str(LORA_PATH))

    lora_a: dict[str, np.ndarray] = {}
    lora_b: dict[str, np.ndarray] = {}
    lora_alpha: dict[str, float] = {}
    for k, v in lora_weights.items():
        kl = k.lower()
        if "lora_a" in kl or "lora_down" in kl:
            base = re.sub(r"\.(lora_[aA]|lora_down)\.weight$", "", k)
            lora_a[base] = v
        elif "lora_b" in kl or "lora_up" in kl:
            base = re.sub(r"\.(lora_[bB]|lora_up)\.weight$", "", k)
            lora_b[base] = v
        elif "alpha" in kl:
            base = re.sub(r"\.alpha$", "", k)
            lora_alpha[base] = float(v)

    # Load ONNX model structure only (external data stays on disk)
    model = onnx.load(str(model_path), load_external_data=False)
    init_index = {init.name: i for i, init in enumerate(model.graph.initializer)}
    norm_to_name = {_normalize_weight_name(n): n for n in init_index}

    applied = 0
    for base, wa in lora_a.items():
        if base not in lora_b:
            continue
        wb = lora_b[base]
        rank = wa.shape[0]
        alpha = lora_alpha.get(base, float(rank))
        scale = LORA_SCALE * alpha / rank
        delta = (wb.astype(np.float32) @ wa.astype(np.float32)) * scale

        norm_base = _normalize_weight_name(
            re.sub(r"^(transformer|unet|model)\.", "", base)
        )
        onnx_name = norm_to_name.get(norm_base)
        if onnx_name is None:
            parts = norm_base.split("/")
            suffix = "/".join(parts[-3:]) if len(parts) >= 3 else norm_base
            for nn, on in norm_to_name.items():
                if nn.endswith(suffix):
                    onnx_name = on
                    break

        if onnx_name is None:
            log.debug("LoRA: no match for %s", base)
            continue

        idx = init_index[onnx_name]
        current = nph.to_array(model.graph.initializer[idx])
        if current.shape == delta.shape:
            merged = current + delta.astype(current.dtype)
            model.graph.initializer[idx].CopyFrom(
                nph.from_array(merged, name=onnx_name)
            )
            applied += 1
        else:
            log.debug("LoRA: shape mismatch for %s", base)

    pct = applied / max(len(lora_a), 1) * 100
    log.info(
        "LoRA: applied %d/%d deltas (%.0f%% match) scale=%.2f",
        applied, len(lora_a), pct, LORA_SCALE,
    )
    if pct < 10 and len(lora_a) > 0:
        log.warning(
            "LoRA match rate <10%% -- ORT-optimised (.opt) transformer may have renamed nodes. "
            "Consider model=\'dev-uncensored\' for reliable LoRA via native diffusers loader."
        )

    import onnxruntime as ort  # type: ignore
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    _lora_sessions[lora_key] = ort.InferenceSession(
        model.SerializeToString(), providers=providers
    )
    return _lora_sessions[lora_key]


# ---------------------------------------------------------------------------
# Diffusers transformer cache  (for dev-uncensored Q4)
# ---------------------------------------------------------------------------
_diffusers_transformer_cache: dict[str, object] = {}


def _get_diffusers_transformer(with_lora: bool = True) -> object:
    """
    Load and cache the Q4 uncensored transformer.
    Fuses the LoRA natively via diffusers load_attn_procs() when with_lora=True.
    Requires: pip install torch diffusers accelerate
    """
    cache_key = f"uncensored::lora={with_lora and LORA_PATH.exists()}"
    if cache_key in _diffusers_transformer_cache:
        return _diffusers_transformer_cache[cache_key]

    try:
        import torch
        from diffusers import FluxTransformer2DModel
    except ImportError as exc:
        raise RuntimeError(
            "model=\'dev-uncensored\' requires torch and diffusers:\n"
            "  pip install torch diffusers accelerate\n"
            f"  ({exc})"
        ) from exc

    if not DEV_UNCENSORED_DIR.exists():
        raise FileNotFoundError(
            f"flux.1-dev-uncensored-q4 directory not found: {DEV_UNCENSORED_DIR}"
        )

    log.info("Loading Q4 uncensored transformer from %s ...", DEV_UNCENSORED_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    transformer = FluxTransformer2DModel.from_pretrained(
        str(DEV_UNCENSORED_DIR),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    if with_lora and LORA_PATH.exists():
        log.info("Fusing LoRA %s into diffusers transformer ...", LORA_PATH.name)
        try:
            transformer.load_attn_procs(str(LORA_PATH.parent))
            transformer.fuse_lora(lora_scale=LORA_SCALE)
            log.info("LoRA fused successfully.")
        except Exception as exc:
            log.warning("LoRA fusion failed (%s) -- continuing without LoRA.", exc)

    transformer.eval()
    _diffusers_transformer_cache[cache_key] = transformer
    return transformer


# ---------------------------------------------------------------------------
# Tokenizer loading  (prefer local FLUX.1-dev; fall back to HuggingFace)
# ---------------------------------------------------------------------------
_tokenizer_cache: dict[str, object] = {}


def _clip_tokenizer() -> object:
    if "clip" not in _tokenizer_cache:
        from transformers import CLIPTokenizer  # type: ignore
        local = FLUX_DEV_DIR / "tokenizer"
        src = str(local) if local.is_dir() else "openai/clip-vit-large-patch14"
        log.info("CLIP tokenizer <- %s", src)
        _tokenizer_cache["clip"] = CLIPTokenizer.from_pretrained(src)
    return _tokenizer_cache["clip"]


def _t5_tokenizer() -> object:
    if "t5" not in _tokenizer_cache:
        from transformers import T5TokenizerFast  # type: ignore
        local = FLUX_DEV_DIR / "tokenizer_2"
        src = str(local) if local.is_dir() else "google/t5-v1_1-xxl"
        log.info("T5 tokenizer <- %s", src)
        _tokenizer_cache["t5"] = T5TokenizerFast.from_pretrained(src)
    return _tokenizer_cache["t5"]


# ---------------------------------------------------------------------------
# GGUF text encoder  (optional alternative to ONNX T5)
# ---------------------------------------------------------------------------
_gguf_model_cache: dict[str, object] = {}


def _get_gguf_model() -> object:
    if "gguf" not in _gguf_model_cache:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FLUX_USE_GGUF_T5=1 requires llama-cpp-python:\n"
                "  pip install llama-cpp-python\n"
                f"  ({exc})"
            ) from exc
        if not GGUF_T5_PATH.exists():
            raise FileNotFoundError(f"GGUF encoder not found: {GGUF_T5_PATH}")
        log.info("Loading GGUF encoder: %s", GGUF_T5_PATH.name)
        _gguf_model_cache["gguf"] = Llama(
            model_path=str(GGUF_T5_PATH),
            embedding=True,
            n_ctx=T5_MAX_LENGTH,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
    return _gguf_model_cache["gguf"]


def _encode_gguf_t5(prompt: str) -> np.ndarray:
    """
    Encode prompt with the GGUF Qwen3-4B encoder.
    Output shape: (1, T5_MAX_LENGTH, 4096) -- zero-padded if model dim < 4096.
    """
    T5_DIM = 4096
    result = _get_gguf_model().create_embedding(prompt)
    raw = np.array(result["data"][0]["embedding"], dtype=np.float32)
    if raw.ndim == 1:
        padded = np.zeros(T5_DIM, dtype=np.float32)
        n = min(raw.shape[0], T5_DIM)
        padded[:n] = raw[:n]
        return np.tile(padded[np.newaxis, np.newaxis, :], (1, T5_MAX_LENGTH, 1))
    out = np.zeros((1, T5_MAX_LENGTH, T5_DIM), dtype=np.float32)
    s, d = min(raw.shape[0], T5_MAX_LENGTH), min(raw.shape[1], T5_DIM)
    out[0, :s, :d] = raw[:s, :d]
    return out


# ---------------------------------------------------------------------------
# Text encoding helpers
# ---------------------------------------------------------------------------

def _encode_clip(sess: object, prompt: str) -> np.ndarray:
    """Run CLIP encoder -> pooled_projections (1, 768)."""
    tok = _clip_tokenizer()
    ids = tok(
        prompt, return_tensors="np", padding="max_length",
        max_length=CLIP_MAX_LENGTH, truncation=True,
    )
    input_ids = ids["input_ids"].astype(np.int64)
    outputs = sess.run(None, {sess.get_inputs()[0].name: input_ids})
    for i, out_info in enumerate(sess.get_outputs()):
        if "pooler" in out_info.name.lower():
            return outputs[i].astype(np.float32)
    return outputs[-1].astype(np.float32)


def _encode_t5_onnx(sess: object, prompt: str) -> np.ndarray:
    """Run ONNX T5 encoder -> last_hidden_state (1, T5_MAX_LENGTH, 4096)."""
    tok = _t5_tokenizer()
    enc = tok(
        prompt, return_tensors="np", padding="max_length",
        max_length=T5_MAX_LENGTH, truncation=True, return_attention_mask=True,
    )
    run_inputs = {sess.get_inputs()[0].name: enc["input_ids"].astype(np.int64)}
    for inp in sess.get_inputs():
        if "attention_mask" in inp.name.lower():
            run_inputs[inp.name] = enc["attention_mask"].astype(np.int64)
    (hidden,) = sess.run(None, run_inputs)
    return hidden.astype(np.float32)


def _get_t5_embedding(t5_sess: Optional[object], prompt: str) -> np.ndarray:
    if USE_GGUF_T5:
        return _encode_gguf_t5(prompt)
    return _encode_t5_onnx(t5_sess, prompt)


# ---------------------------------------------------------------------------
# FLUX latent helpers
# ---------------------------------------------------------------------------

def _patchify(latent: np.ndarray) -> np.ndarray:
    """(1, 16, H, W) -> (1, H/2 * W/2, 64)"""
    b, c, h, w = latent.shape
    x = latent.reshape(b, c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def _unpatchify(packed: np.ndarray, lh: int, lw: int) -> np.ndarray:
    """(1, H/2 * W/2, 64) -> (1, 16, H, W)"""
    b, _, d = packed.shape
    c = d // 4
    x = packed.reshape(b, lh // 2, lw // 2, c, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, lh, lw)


def _make_img_ids(lh: int, lw: int) -> np.ndarray:
    ph, pw = lh // 2, lw // 2
    ids = np.zeros((ph, pw, 3), dtype=np.float32)
    ids[:, :, 1] = np.arange(ph)[:, np.newaxis]
    ids[:, :, 2] = np.arange(pw)[np.newaxis, :]
    return ids.reshape(1, ph * pw, 3)


def _make_txt_ids(txt_len: int) -> np.ndarray:
    return np.zeros((1, txt_len, 3), dtype=np.float32)


def _build_transformer_inputs(
    sess: object,
    packed: np.ndarray,
    enc_hidden: np.ndarray,
    pooled: np.ndarray,
    timestep: np.ndarray,
    img_ids: np.ndarray,
    txt_ids: np.ndarray,
    guidance: Optional[np.ndarray] = None,
) -> dict:
    """Match arrays to ONNX inputs by standard FLUX naming conventions."""
    result: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        n = inp.name.lower()
        if "hidden_states" in n and "encoder" not in n:
            result[inp.name] = packed
        elif "encoder_hidden_states" in n:
            result[inp.name] = enc_hidden
        elif "pooled_projections" in n:
            result[inp.name] = pooled
        elif "timestep" in n or n == "t":
            result[inp.name] = timestep
        elif "img_ids" in n:
            result[inp.name] = img_ids
        elif "txt_ids" in n:
            result[inp.name] = txt_ids
        elif "guidance" in n and guidance is not None:
            result[inp.name] = guidance
    return result


# ---------------------------------------------------------------------------
# Shared denoising loop (numpy / ONNX)
# ---------------------------------------------------------------------------

def _denoise_onnx(
    transformer_sess: object,
    packed: np.ndarray,
    enc_hidden: np.ndarray,
    pooled: np.ndarray,
    img_ids: np.ndarray,
    txt_ids: np.ndarray,
    steps: int,
    guidance: Optional[np.ndarray] = None,
) -> np.ndarray:
    for t_val in np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]:
        t_tensor = np.full((1,), t_val, dtype=np.float32)
        run_inputs = _build_transformer_inputs(
            transformer_sess, packed, enc_hidden, pooled,
            t_tensor, img_ids, txt_ids, guidance=guidance,
        )
        (noise_pred,) = transformer_sess.run(None, run_inputs)
        packed = packed + noise_pred * (-1.0 / steps)
    return packed


def _decode_vae(vae_sess: object, latents: np.ndarray) -> Image.Image:
    vae_in = latents / FLUX_VAE_SCALE + FLUX_VAE_SHIFT
    (decoded,) = vae_sess.run(None, {vae_sess.get_inputs()[0].name: vae_in})
    arr = (np.clip(decoded[0].transpose(1, 2, 0), -1.0, 1.0) * 0.5 + 0.5) * 255
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Pipeline: FLUX.1-schnell  (4-step, no CFG)
# ---------------------------------------------------------------------------

def _run_schnell(
    prompt: str, width: int, height: int, steps: int,
    guidance_scale: float, seed: int, use_lora: bool,
) -> Image.Image:
    clip_sess = _session(_component(SCHNELL_DIR, "clip"))
    t5_sess = None if USE_GGUF_T5 else _session(_t5(SCHNELL_DIR))
    trans_path = _transformer(SCHNELL_DIR, SCHNELL_PRECISION)
    transformer_sess = _session_with_lora(trans_path) if use_lora else _session(trans_path)
    vae_sess = _session(_component(SCHNELL_DIR, "vae"))

    rng = np.random.default_rng(seed)
    pooled = _encode_clip(clip_sess, prompt)
    enc_hidden = _get_t5_embedding(t5_sess, prompt)
    lh, lw = height // 8, width // 8
    latents = rng.standard_normal((1, 16, lh, lw)).astype(np.float32)
    packed = _patchify(latents)
    img_ids, txt_ids = _make_img_ids(lh, lw), _make_txt_ids(enc_hidden.shape[1])
    packed = _denoise_onnx(transformer_sess, packed, enc_hidden, pooled, img_ids, txt_ids, steps)
    return _decode_vae(vae_sess, _unpatchify(packed, lh, lw))


# ---------------------------------------------------------------------------
# Pipeline: FLUX.1-dev ONNX  (guided, higher quality)
# ---------------------------------------------------------------------------

def _run_dev_onnx(
    prompt: str, width: int, height: int, steps: int,
    guidance_scale: float, seed: int, use_lora: bool,
) -> Image.Image:
    clip_sess = _session(_component(DEV_ONNX_DIR, "clip"))
    t5_sess = None if USE_GGUF_T5 else _session(_t5(DEV_ONNX_DIR))
    trans_path = _transformer(DEV_ONNX_DIR, DEV_ONNX_PRECISION)
    transformer_sess = _session_with_lora(trans_path) if use_lora else _session(trans_path)
    vae_sess = _session(_component(DEV_ONNX_DIR, "vae"))

    rng = np.random.default_rng(seed)
    pooled = _encode_clip(clip_sess, prompt)
    enc_hidden = _get_t5_embedding(t5_sess, prompt)
    lh, lw = height // 8, width // 8
    latents = rng.standard_normal((1, 16, lh, lw)).astype(np.float32)
    packed = _patchify(latents)
    img_ids, txt_ids = _make_img_ids(lh, lw), _make_txt_ids(enc_hidden.shape[1])
    g = np.full((1,), guidance_scale, dtype=np.float32)
    packed = _denoise_onnx(transformer_sess, packed, enc_hidden, pooled, img_ids, txt_ids, steps, guidance=g)
    return _decode_vae(vae_sess, _unpatchify(packed, lh, lw))


# ---------------------------------------------------------------------------
# Pipeline: flux.1-dev-uncensored-q4  (torch transformer + ONNX clip/t5/vae)
# ---------------------------------------------------------------------------

def _run_dev_uncensored(
    prompt: str, width: int, height: int, steps: int,
    guidance_scale: float, seed: int, use_lora: bool,
) -> Image.Image:
    """
    Mixed pipeline: CLIP / T5 / VAE from FLUX.1-dev-onnx (ONNX),
    transformer from flux.1-dev-uncensored-q4 (torch diffusers).
    LoRA is fused natively via diffusers.load_attn_procs().
    Requires: pip install torch diffusers accelerate
    """
    import torch

    transformer = _get_diffusers_transformer(with_lora=use_lora)
    device: str = next(transformer.parameters()).device.type
    dtype = next(transformer.parameters()).dtype

    clip_sess = _session(_component(DEV_ONNX_DIR, "clip"))
    t5_sess = None if USE_GGUF_T5 else _session(_t5(DEV_ONNX_DIR))
    vae_sess = _session(_component(DEV_ONNX_DIR, "vae"))

    rng = np.random.default_rng(seed)
    pooled_np = _encode_clip(clip_sess, prompt)
    enc_np = _get_t5_embedding(t5_sess, prompt)
    lh, lw = height // 8, width // 8
    latents_np = rng.standard_normal((1, 16, lh, lw)).astype(np.float32)

    packed = torch.from_numpy(_patchify(latents_np)).to(device, dtype=dtype)
    pooled = torch.from_numpy(pooled_np).to(device, dtype=dtype)
    enc_hidden = torch.from_numpy(enc_np).to(device, dtype=dtype)
    img_ids_t = torch.from_numpy(_make_img_ids(lh, lw)).to(device)
    txt_ids_t = torch.from_numpy(_make_txt_ids(enc_np.shape[1])).to(device)
    guidance_t = torch.full((1,), guidance_scale, device=device, dtype=dtype)

    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]
    with torch.no_grad():
        for t in timesteps:
            noise_pred = transformer(
                hidden_states=packed,
                encoder_hidden_states=enc_hidden,
                pooled_projections=pooled,
                timestep=t.unsqueeze(0),
                img_ids=img_ids_t,
                txt_ids=txt_ids_t,
                guidance=guidance_t,
                return_dict=False,
            )[0]
            packed = packed + noise_pred * (-1.0 / steps)

    latents_out = _unpatchify(packed.float().cpu().numpy(), lh, lw)
    return _decode_vae(vae_sess, latents_out)


# ---------------------------------------------------------------------------
# Pipeline: FLUX.1-Kontext-dev  (image editing, requires vae_encoder)
# ---------------------------------------------------------------------------

def _run_kontext(
    prompt: str, source_image: Image.Image,
    width: int, height: int, steps: int,
    guidance_scale: float, seed: int, use_lora: bool,
) -> Image.Image:
    vae_enc_path = _component(KONTEXT_DIR, "vae_encoder")
    if not vae_enc_path.exists():
        raise FileNotFoundError(
            f"FLUX.1-Kontext-dev VAE encoder not yet available:\n  {vae_enc_path}\n"
            "Download vae_encoder.opt/model.onnx and place it in the Kontext directory."
        )

    clip_sess = _session(_component(KONTEXT_DIR, "clip"))
    t5_sess = None if USE_GGUF_T5 else _session(_t5(KONTEXT_DIR))
    vae_enc_sess = _session(vae_enc_path)
    trans_path = _transformer(KONTEXT_DIR, KONTEXT_PRECISION)
    transformer_sess = _session_with_lora(trans_path) if use_lora else _session(trans_path)
    vae_dec_sess = _session(_component(KONTEXT_DIR, "vae"))

    rng = np.random.default_rng(seed)
    pooled = _encode_clip(clip_sess, prompt)
    enc_hidden = _get_t5_embedding(t5_sess, prompt)
    lh, lw = height // 8, width // 8

    src = source_image.resize((width, height), Image.LANCZOS)
    src_arr = (np.array(src, dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]
    (raw_latent,) = vae_enc_sess.run(None, {vae_enc_sess.get_inputs()[0].name: src_arr})
    img_z = (raw_latent - FLUX_VAE_SHIFT) * FLUX_VAE_SCALE

    noise = rng.standard_normal((1, 16, lh, lw)).astype(np.float32)
    packed = _patchify(img_z + noise)
    img_ids, txt_ids = _make_img_ids(lh, lw), _make_txt_ids(enc_hidden.shape[1])
    g = np.full((1,), guidance_scale, dtype=np.float32)
    packed = _denoise_onnx(transformer_sess, packed, enc_hidden, pooled, img_ids, txt_ids, steps, guidance=g)
    return _decode_vae(vae_dec_sess, _unpatchify(packed, lh, lw))


# ---------------------------------------------------------------------------
# Face detection heuristic
# ---------------------------------------------------------------------------

def _detect_faces_heuristic(img: Image.Image) -> list:
    arr = np.array(img.resize((320, 240), Image.BILINEAR), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    skin = (
        (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b)
        & (np.abs(r - g) > 15)
        & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15)
    )
    block, h, w = 8, *skin.shape
    boxes = []
    for by in range(0, h, block):
        for bx in range(0, w, block):
            patch = skin[by: by + block, bx: bx + block]
            if patch.mean() < 0.4:
                continue
            y0, y1 = max(0, by - block * 2), min(h, by + block * 4)
            x0, x1 = max(0, bx - block * 2), min(w, bx + block * 4)
            boxes.append(FaceBox(
                x=x0 / w * img.width, y=y0 / h * img.height,
                width=(x1 - x0) / w * img.width, height=(y1 - y0) / h * img.height,
                confidence=float(patch.mean()),
            ))
    boxes.sort(key=lambda b: b.width * b.height, reverse=True)
    return boxes[:3]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="FLUX ONNX Inference Server", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:55173", "http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    model: Literal["schnell", "dev", "dev-uncensored"] = "schnell"
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    # None -> 4 for schnell, 20 for dev variants
    steps: Optional[int] = None
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)
    seed: Optional[int] = None
    use_lora: bool = True


class EditRequest(BaseModel):
    prompt: str
    image_base64: str
    width: Optional[int] = None
    height: Optional[int] = None
    steps: int = Field(default=20, ge=1, le=50)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)
    seed: Optional[int] = None
    use_lora: bool = True


class DetectFacesRequest(BaseModel):
    image_base64: str


class ImageResponse(BaseModel):
    image_base64: str


class FaceBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float


class DetectFacesResponse(BaseModel):
    faces: list[FaceBox]


# ---------------------------------------------------------------------------
# Audio / .pth model catalog
# ---------------------------------------------------------------------------

def _discover_pth_models() -> list[dict]:
    """
    Recursively scan _BASE for PyTorch .pth files and classify each one.

    Classification rules (in order):
      rvc        — paired with a FAISS .index file in the same directory, or the
                   directory name contains common RVC cues
      extension  — lives under an extensions/ subdirectory
      other      — everything else (unknown standalone .pth)

    Returns a list of dicts sorted by kind then name, each containing:
      name      — stem of the filename
      filename  — full filename
      path      — absolute path string
      kind      — "rvc" | "extension" | "other"
      has_index — True when a companion .index file exists (RVC characteristic)
      size_mb   — file size in MB
    """
    base = Path(_BASE)
    if not base.exists():
        return []

    results = []
    for pth in sorted(base.rglob("*.pth")):
        try:
            size_mb = round(pth.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None

        has_index = bool(list(pth.parent.glob("*.index")))
        rel_parts = {p.lower() for p in pth.relative_to(base).parts}

        if has_index or rel_parts & {"rvc", "voice", "singing"}:
            kind = "rvc"
        elif "extensions" in rel_parts:
            kind = "extension"
        else:
            kind = "other"

        results.append(
            {
                "name": pth.stem,
                "filename": pth.name,
                "path": str(pth),
                "kind": kind,
                "has_index": has_index,
                "size_mb": size_mb,
            }
        )

    results.sort(key=lambda m: (m["kind"], m["name"].lower()))
    return results


# ---------------------------------------------------------------------------
# LoRA catalog
# ---------------------------------------------------------------------------

def _discover_loras() -> list[dict]:
    """
    Recursively scan LORA_DIR for LoRA weight files (.safetensors).

    Returns a list of dicts sorted by name, each containing:
      name     — stem of the filename
      filename — full filename
      path     — absolute path string
      size_mb  — file size in MB (None on error)
    """
    if not LORA_DIR.exists():
        return []

    results = []
    for sf in sorted(LORA_DIR.rglob("*.safetensors")):
        try:
            size_mb = round(sf.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None

        results.append(
            {
                "name": sf.stem,
                "filename": sf.name,
                "path": str(sf),
                "size_mb": size_mb,
            }
        )

    results.sort(key=lambda m: m["name"].lower())
    return results


# ---------------------------------------------------------------------------
# Reactorplus asset catalog
# ---------------------------------------------------------------------------

# Model weight extensions to catalog
_MODEL_SUFFIXES = {".safetensors", ".bin", ".pth", ".pt"}


def _discover_reactor_assets() -> list[dict]:
    """
    Recursively scan REACTOR_DIR for model weight files and classify each one.

    Classification rules (in order):
      classifier  — inside attractive_faces_celebs_detection/ (face-scoring ViT)
      embedding   — filename starts with "learned_embeds" (textual inversion)
      checkpoint  — inside a checkpoint-*/ subdirectory (training snapshot)
      diffusers   — inside a diffusers component subdirectory
                    (unet/, vae/, text_encoder/, feature_extractor/, scheduler/)
      other       — everything else

    Returns a list of dicts sorted by kind then name, each containing:
      name     — stem of the filename
      filename — full filename
      path     — absolute path string
      kind     — "classifier" | "embedding" | "checkpoint" | "diffusers" | "other"
      size_mb  — file size in MB (None on error)
    """
    if not REACTOR_DIR.exists():
        return []

    _DIFFUSERS_COMPONENTS = {"unet", "vae", "text_encoder", "feature_extractor", "scheduler"}

    results = []
    for f in sorted(REACTOR_DIR.rglob("*")):
        if f.suffix.lower() not in _MODEL_SUFFIXES:
            continue
        if not f.is_file():
            continue

        try:
            size_mb = round(f.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None

        rel_parts_lower = [p.lower() for p in f.relative_to(REACTOR_DIR).parts]

        if "attractive_faces_celebs_detection" in rel_parts_lower:
            kind = "classifier"
        elif f.stem.lower().startswith("learned_embeds"):
            kind = "embedding"
        elif any(p.startswith("checkpoint-") for p in rel_parts_lower):
            kind = "checkpoint"
        elif _DIFFUSERS_COMPONENTS & set(rel_parts_lower):
            kind = "diffusers"
        else:
            kind = "other"

        results.append(
            {
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "kind": kind,
                "size_mb": size_mb,
            }
        )

    results.sort(key=lambda m: (m["kind"], m["name"].lower()))
    return results


# ---------------------------------------------------------------------------
# Real-ESRGAN upscaler catalog
# ---------------------------------------------------------------------------

# Scale-factor keywords → kind label (checked in order against lowercase filename stem)
_ESRGAN_SCALE_RULES: list[tuple[str, str]] = [
    ("x4",           "x4plus"),
    ("x2",           "x2plus"),
    ("discriminator", "discriminator"),
]
_ESRGAN_ANIME_KEYWORDS = {"anime", "animation"}
_ESRGAN_SUFFIXES = {".pth", ".onnx"}


def _discover_realesrgan_models() -> list[dict]:
    """
    Recursively scan REALESRGAN_DIR for Real-ESRGAN weight files (.pth, .onnx)
    and classify each one.

    Classification rules (in order, applied to the lowercase filename stem):
      x4plus_anime  — scale x4 AND filename contains an anime keyword
      x4plus        — filename contains "x4"
      x2plus        — filename contains "x2"
      discriminator — filename contains "discriminator"
      other         — everything else

    Returns a list of dicts sorted by kind then name, each containing:
      name     — stem of the filename
      filename — full filename
      path     — absolute path string
      kind     — "x4plus_anime" | "x4plus" | "x2plus" | "discriminator" | "other"
      format   — "onnx" | "pth"
      size_mb  — file size in MB (None on error)
    """
    if not REALESRGAN_DIR.exists():
        return []

    results = []
    for f in sorted(REALESRGAN_DIR.rglob("*")):
        if f.suffix.lower() not in _ESRGAN_SUFFIXES:
            continue
        if not f.is_file():
            continue

        try:
            size_mb = round(f.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None

        stem_lower = f.stem.lower()
        has_x4 = "x4" in stem_lower
        has_anime = bool(_ESRGAN_ANIME_KEYWORDS & set(stem_lower.split("_")))

        if has_x4 and has_anime:
            kind = "x4plus_anime"
        elif has_x4:
            kind = "x4plus"
        elif "x2" in stem_lower:
            kind = "x2plus"
        elif "discriminator" in stem_lower:
            kind = "discriminator"
        else:
            kind = "other"

        results.append(
            {
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "kind": kind,
                "format": "onnx" if f.suffix.lower() == ".onnx" else "pth",
                "size_mb": size_mb,
            }
        )

    results.sort(key=lambda m: (m["kind"], m["name"].lower()))
    return results


# ---------------------------------------------------------------------------
# Stable Diffusion pipeline catalog
# ---------------------------------------------------------------------------

_SD_PIPELINE_COMPONENTS = {
    "unet", "vae", "vae_1_0", "vae_encoder", "vae_decoder",
    "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoders",
    "tokenizer", "tokenizer_2", "tokenizer_3",
    "transformer", "image_encoder", "safety_checker",
    "feature_extractor", "scheduler",
}
_SD_WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".bin", ".pth"}


def _sd_variant(path_parts: list[str]) -> str:
    """Identify the SD architecture variant from directory path parts."""
    joined = " ".join(p.lower() for p in path_parts)
    if any(k in joined for k in ("sv3d", "sv4d")):
        return "sv3d_sv4d"
    if any(k in joined for k in ("svd", "stable-video", "img2vid", "stable_video")):
        return "svd"
    if any(k in joined for k in ("3.5", "sd3", "sd35", "stable-diffusion-3")):
        return "sd3"
    if any(k in joined for k in ("xl", "sdxl")):
        return "sdxl"
    if any(k in joined for k in ("v1-5", "sd1", "sd15", "stable-diffusion-v1")):
        return "sd1.5"
    if any(k in joined for k in ("v2", "sd2", "sd20", "sd21")):
        return "sd2.x"
    return "other"


def _discover_sd_pipelines() -> list[dict]:
    """
    Scan SD_DIR for diffusers pipelines (detected by model_index.json) and return
    a summary of each including variant, present components, and any root-level
    standalone checkpoint files (.ckpt / .safetensors).

    Returns a list sorted by variant then pipeline name.
    """
    if not SD_DIR.exists():
        return []

    pipelines: list[dict] = []
    seen: set[Path] = set()

    for idx_file in sorted(SD_DIR.rglob("model_index.json")):
        pipeline_dir = idx_file.parent
        if pipeline_dir in seen:
            continue
        seen.add(pipeline_dir)

        rel_parts = list(pipeline_dir.relative_to(SD_DIR).parts)
        variant = _sd_variant(rel_parts)

        components = sorted(
            p.name for p in pipeline_dir.iterdir()
            if p.is_dir() and p.name.lower() in _SD_PIPELINE_COMPONENTS
        )

        checkpoints: list[dict] = []
        for f in sorted(pipeline_dir.glob("*")):
            if f.suffix.lower() in {".safetensors", ".ckpt"} and f.is_file():
                try:
                    size_mb = round(f.stat().st_size / 1_048_576, 1)
                except OSError:
                    size_mb = None
                checkpoints.append({"filename": f.name, "size_mb": size_mb, "format": f.suffix.lstrip(".")})

        pipelines.append({
            "name": pipeline_dir.name,
            "path": str(pipeline_dir),
            "variant": variant,
            "components": components,
            "checkpoints": checkpoints,
        })

    pipelines.sort(key=lambda p: (p["variant"], p["name"].lower()))
    return pipelines


def _discover_sd_extensions() -> list[dict]:
    """
    Scan SD_DIR/extensions/ for .pth model files used by ControlNet, DreamBooth,
    Inpainting, and similar SD extensions.
    """
    ext_dir = SD_DIR / "extensions"
    if not ext_dir.exists():
        return []

    results: list[dict] = []
    for pth in sorted(ext_dir.rglob("*.pth")):
        try:
            size_mb = round(pth.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None
        results.append({
            "name": pth.stem,
            "filename": pth.name,
            "path": str(pth),
            "extension": pth.parent.name,
            "size_mb": size_mb,
        })
    return results


# ---------------------------------------------------------------------------
# WAN2 video generation model catalog
# ---------------------------------------------------------------------------

_WAN2_COMPONENTS = {
    "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2",
    "transformer", "vae", "scheduler", "image_encoder",
}
_WAN2_WEIGHT_SUFFIXES = {".safetensors", ".pth"}


def _wan2_meta(name: str) -> dict:
    """Classify a WAN2 model directory name into task / version / size."""
    nl = name.lower()

    if "ti2v" in nl:
        task = "text_image_to_video"
    elif "i2v" in nl:
        task = "image_to_video"
    elif "t2v" in nl:
        task = "text_to_video"
    elif "lora" in nl:
        task = "lora"
    elif nl in ("high_noise_model", "low_noise_model"):
        task = "noise_model"
    else:
        task = "other"

    if "14b" in nl or "a14b" in nl:
        size = "14b"
    elif "a7b" in nl or "7b" in nl:
        size = "7b"
    elif "5b" in nl:
        size = "5b"
    elif "1.3b" in nl:
        size = "1.3b"
    else:
        size = None

    if "2.2" in nl:
        version = "2.2"
    elif "2.1" in nl:
        version = "2.1"
    else:
        version = None

    return {"task": task, "size": size, "version": version}


def _discover_wan2_models() -> list[dict]:
    """
    Scan WAN2_DIR (one level deep) for Wan2.x video generation model directories.

    Each entry contains:
      name              — directory name
      path              — absolute path string
      format            — "diffusers" (has model_index.json) | "native" (has config.json
                          or configuration.json) | "unknown"
      task              — "text_to_video" | "image_to_video" | "text_image_to_video"
                          | "noise_model" | "lora" | "other"
      version           — "2.1" | "2.2" | null
      size              — "1.3b" | "5b" | "7b" | "14b" | null
      components        — list of standard diffusers component subdirectory names present
      root_weights      — standalone weight files (.pth / .safetensors) at the directory root
      transformer_shards— number of sharded safetensors in the transformer/ subdir
    """
    if not WAN2_DIR.exists():
        return []

    results: list[dict] = []
    for d in sorted(WAN2_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue

        meta = _wan2_meta(d.name)

        if (d / "model_index.json").exists():
            fmt = "diffusers"
        elif (d / "config.json").exists() or (d / "configuration.json").exists():
            fmt = "native"
        else:
            fmt = "unknown"

        root_weights: list[dict] = []
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in _WAN2_WEIGHT_SUFFIXES and f.is_file():
                try:
                    size_mb = round(f.stat().st_size / 1_048_576, 1)
                except OSError:
                    size_mb = None
                root_weights.append({"filename": f.name, "size_mb": size_mb})

        components = sorted(
            p.name for p in d.iterdir()
            if p.is_dir() and p.name in _WAN2_COMPONENTS
        )

        transformer_dir = d / "transformer"
        transformer_shards = (
            len(list(transformer_dir.glob("*.safetensors")))
            if transformer_dir.exists() else 0
        )

        results.append({
            "name": d.name,
            "path": str(d),
            "format": fmt,
            "task": meta["task"],
            "version": meta["version"],
            "size": meta["size"],
            "components": components,
            "root_weights": root_weights,
            "transformer_shards": transformer_shards,
        })

    results.sort(key=lambda m: (m["task"], m["name"].lower()))
    return results


# ---------------------------------------------------------------------------
# Z-Image model catalog
# ---------------------------------------------------------------------------

_ZIMAGE_COMPONENTS = {
    "text_encoder", "tokenizer", "transformer", "vae", "scheduler",
    "image_encoder", "text_encoder_2",
}


def _zimage_variant(name: str) -> str:
    nl = name.lower()
    if "turbo" in nl:
        return "turbo"
    if "edit" in nl:
        return "edit"
    if "omni" in nl:
        return "omni"
    return "base"


def _discover_zimage_models() -> list[dict]:
    """
    Scan ZIMAGE_DIR (one level deep) for Z-Image model directories.

    Format detection:
      diffusers  — directory contains model_index.json
      partial    — component subdirectories exist but no model_index.json
      unknown    — no recognisable pipeline structure

    Variants: base | edit | turbo | omni

    Returns a list sorted by variant then name, each entry containing:
      name, path, variant, format, components, transformer_shards
    """
    if not ZIMAGE_DIR.exists():
        return []

    results: list[dict] = []
    for d in sorted(ZIMAGE_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue

        variant = _zimage_variant(d.name)

        if (d / "model_index.json").exists():
            fmt = "diffusers"
        elif any((d / c).is_dir() for c in _ZIMAGE_COMPONENTS):
            fmt = "partial"
        else:
            fmt = "unknown"

        components = sorted(
            p.name for p in d.iterdir()
            if p.is_dir() and p.name in _ZIMAGE_COMPONENTS
        )

        transformer_dir = d / "transformer"
        transformer_shards = (
            len(list(transformer_dir.glob("*.safetensors")))
            if transformer_dir.exists() else 0
        )

        results.append({
            "name": d.name,
            "path": str(d),
            "variant": variant,
            "format": fmt,
            "components": components,
            "transformer_shards": transformer_shards,
        })

    results.sort(key=lambda m: (m["variant"], m["name"].lower()))
    return results


# ---------------------------------------------------------------------------
# Standalone VAE catalog
# ---------------------------------------------------------------------------

_VAE_WEIGHT_SUFFIXES = {".safetensors", ".pth", ".onnx"}
_VAE_SHARD_RE = re.compile(r"-\d+-of-\d+", re.IGNORECASE)


def _vae_kind(stem: str, rel_parts: list[str]) -> str:
    sl = stem.lower()
    path_lower = " ".join(p.lower() for p in rel_parts)
    if "flux" in path_lower or sl == "ae":
        return "flux"
    if sl.startswith("wan") or "wan2" in sl:
        return "wan2"
    if "xl" in sl or "sdxl" in sl or "xl" in path_lower:
        return "sdxl"
    if "onnx" in sl or "onnx" in path_lower:
        return "onnx"
    # Named community VAEs (e.g. VAE_ftasticVAE_v10)
    if stem.startswith("VAE_"):
        return "community"
    return "sd"


def _vae_format(filename: str) -> str:
    fl = filename.lower()
    if fl.endswith(".onnx"):
        return "onnx"
    if fl.endswith(".pth"):
        return "pth"
    if ".fp16." in fl:
        return "safetensors_fp16"
    return "safetensors"


def _discover_vae_models() -> list[dict]:
    """
    Recursively scan VAE_DIR for VAE weight files and classify each one.

    Kinds:
      flux       — FLUX.1 autoencoder (ae.safetensors, FLUX.1-dev* subdirs)
      wan2       — Wan2.x VAE (.pth files named Wan*.pth)
      sdxl       — SDXL VAE (sd_xl_*, stable-diffusion-xl-* subdirs)
      onnx       — ONNX format (.onnx files)
      community  — named community VAEs (filenames starting with VAE_)
      sd         — generic Stable Diffusion VAE

    Returns a list sorted by kind then name.
    """
    if not VAE_DIR.exists():
        return []

    results: list[dict] = []
    for f in sorted(VAE_DIR.rglob("*")):
        if f.suffix.lower() not in _VAE_WEIGHT_SUFFIXES or not f.is_file():
            continue
        try:
            size_mb = round(f.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None

        rel_parts = list(f.relative_to(VAE_DIR).parts[:-1])
        kind = _vae_kind(f.stem, rel_parts)
        fmt = _vae_format(f.name)
        is_shard = bool(_VAE_SHARD_RE.search(f.stem))

        results.append({
            "name": f.stem,
            "filename": f.name,
            "path": str(f),
            "kind": kind,
            "format": fmt,
            "is_shard": is_shard,
            "size_mb": size_mb,
        })

    results.sort(key=lambda m: (m["kind"], m["name"].lower()))
    return results


# ---------------------------------------------------------------------------
# Text encoder catalog
# ---------------------------------------------------------------------------

_TE_WEIGHT_SUFFIXES = {".safetensors", ".bin"}
_TE_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)(?:_(\d+))?$", re.IGNORECASE)


def _te_kind(stem: str) -> str:
    sl = stem.lower()
    if "clip_l" in sl:
        return "clip_l"
    if "clip_g" in sl:
        return "clip_g"
    if "t5xxl" in sl or "t5-xxl" in sl or "t5_xxl" in sl:
        return "t5"
    if "openvino" in sl:
        return "openvino"
    if "text_encoder_3" in sl:
        return "te3"
    if "text_encoder_2" in sl:
        return "te2"
    if "text_encoder" in sl:
        return "te1"
    return "other"


def _discover_text_encoders() -> dict:
    """
    Scan TEXT_ENC_DIR (flat, no recursion) for text encoder weight files and
    FLUX model_index JSON configs.

    Weight files are classified by encoder type:
      clip_l    — CLIP-L (used by FLUX, SDXL, SD3)
      clip_g    — CLIP-G (used by SDXL, SD3)
      t5        — T5-XXL encoder (fp8 / fp16 variants)
      te1       — text_encoder primary (sharded or single)
      te2       — text_encoder_2 (sharded or single)
      te3       — text_encoder_3 (sharded or single)
      openvino  — OpenVINO binary weights (.bin)
      other     — unclassified

    Returns a dict with:
      model_indexes  — FLUX model_index JSON files in the directory
      encoders       — list of weight file entries sorted by kind then name
      by_kind        — count summary per kind
    """
    if not TEXT_ENC_DIR.exists():
        return {"model_indexes": [], "encoders": [], "by_kind": {}}

    model_indexes: list[dict] = []
    encoders: list[dict] = []

    for f in sorted(TEXT_ENC_DIR.iterdir()):
        if not f.is_file():
            continue

        # FLUX model_index JSON files
        if f.suffix.lower() == ".json" and "model_index" in f.name.lower():
            model_indexes.append({
                "filename": f.name,
                "path": str(f),
                "model": f.name.split("_model_index")[0],  # e.g. "FLUX.1-dev"
            })
            continue

        if f.suffix.lower() not in _TE_WEIGHT_SUFFIXES:
            continue

        try:
            size_mb = round(f.stat().st_size / 1_048_576, 1)
        except OSError:
            size_mb = None

        kind = _te_kind(f.stem)

        # Detect sharding: stem ends with -NNN-of-MMM or -NNN-of-MMM_V
        shard_match = _TE_SHARD_RE.search(f.stem)
        if shard_match:
            shard_num = int(shard_match.group(1))
            shard_total = int(shard_match.group(2))
            variant = shard_match.group(3)  # suffix like "_1" → "1"
        else:
            shard_num = shard_total = None
            variant = None

        encoders.append({
            "name": f.stem,
            "filename": f.name,
            "path": str(f),
            "kind": kind,
            "format": "openvino" if f.suffix.lower() == ".bin" else "safetensors",
            "precision": "fp16" if ".fp16." in f.name.lower() or "fp16" in f.stem.lower()
                         else ("fp8" if "fp8" in f.stem.lower() else "full"),
            "shard_num": shard_num,
            "shard_total": shard_total,
            "variant": variant,
            "size_mb": size_mb,
        })

    encoders.sort(key=lambda e: (e["kind"], e["name"].lower()))
    by_kind = {
        k: sum(1 for e in encoders if e["kind"] == k)
        for k in ("clip_l", "clip_g", "t5", "te1", "te2", "te3", "openvino", "other")
        if any(e["kind"] == k for e in encoders)
    }
    return {"model_indexes": model_indexes, "encoders": encoders, "by_kind": by_kind}

def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _random_seed(seed: Optional[int]) -> int:
    return seed if seed is not None else int.from_bytes(os.urandom(4), "little")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models": {
            "schnell": {
                "dir": str(SCHNELL_DIR),
                "precision": SCHNELL_PRECISION,
                "available": SCHNELL_DIR.exists(),
            },
            "dev": {
                "dir": str(DEV_ONNX_DIR),
                "precision": DEV_ONNX_PRECISION,
                "available": DEV_ONNX_DIR.exists(),
            },
            "dev-uncensored": {
                "dir": str(DEV_UNCENSORED_DIR),
                "available": DEV_UNCENSORED_DIR.exists(),
                "note": "requires torch + diffusers",
            },
            "kontext": {
                "dir": str(KONTEXT_DIR),
                "precision": KONTEXT_PRECISION,
                "available": KONTEXT_DIR.exists(),
                "vae_encoder_ready": _component(KONTEXT_DIR, "vae_encoder").exists(),
            },
        },
        "text_encoder": "gguf" if USE_GGUF_T5 else ("t5-fp8" if USE_FP8_T5 else "t5-full"),
        "gguf_encoder": {
            "path": str(GGUF_T5_PATH),
            "available": GGUF_T5_PATH.exists(),
        },
        "lora": {
            "path": str(LORA_PATH),
            "scale": LORA_SCALE,
            "available": LORA_PATH.exists(),
        },
        "loras": {
            "dir": str(LORA_DIR),
            "count": len(_discover_loras()),
            "note": "GET /loras for full list",
        },
        "tokenizer_source": (
            str(FLUX_DEV_DIR / "tokenizer")
            if (FLUX_DEV_DIR / "tokenizer").is_dir()
            else "huggingface"
        ),
        "audio_models": {
            "note": "GET /audio-models for full list",
            **{
                k: sum(1 for m in _discover_pth_models() if m["kind"] == k)
                for k in ("rvc", "extension", "other")
            },
        },
        "reactor_models": {
            "dir": str(REACTOR_DIR),
            "note": "GET /reactor-models for full list",
            **{
                k: sum(1 for m in _discover_reactor_assets() if m["kind"] == k)
                for k in ("classifier", "embedding", "checkpoint", "diffusers", "other")
            },
        },
        "upscalers": {
            "dir": str(REALESRGAN_DIR),
            "note": "GET /upscalers for full list",
            **{
                k: sum(1 for m in _discover_realesrgan_models() if m["kind"] == k)
                for k in ("x4plus_anime", "x4plus", "x2plus", "discriminator", "other")
            },
        },
        "sd_models": {
            "dir": str(SD_DIR),
            "pipelines": len(_discover_sd_pipelines()),
            "note": "GET /sd-models for full list",
        },
        "wan2_models": {
            "dir": str(WAN2_DIR),
            "count": len(_discover_wan2_models()),
            "note": "GET /wan2-models for full list",
        },
        "zimage_models": {
            "dir": str(ZIMAGE_DIR),
            "count": len(_discover_zimage_models()),
            "note": "GET /zimage-models for full list",
        },
        "vae_models": {
            "dir": str(VAE_DIR),
            "note": "GET /vae-models for full list",
            **{
                k: sum(1 for m in _discover_vae_models() if m["kind"] == k)
                for k in ("flux", "wan2", "sdxl", "onnx", "community", "sd")
            },
        },
        "text_encoders": {
            "dir": str(TEXT_ENC_DIR),
            "note": "GET /text-encoders for full list",
            **_discover_text_encoders()["by_kind"],
        },
    }


@app.get("/audio-models")
async def audio_models() -> dict:
    """
    Return every .pth file found under the Flux model root, classified by type.

    Kinds:
      rvc        — RVC (Retrieval-based Voice Conversion) model; has a companion .index file
      extension  — model stored under an extensions/ subdirectory
      other      — unclassified standalone .pth
    """
    found = _discover_pth_models()
    return {
        "total": len(found),
        "by_kind": {
            k: [m for m in found if m["kind"] == k]
            for k in ("rvc", "extension", "other")
        },
    }


@app.get("/loras")
async def loras() -> dict:
    """
    Return every LoRA .safetensors file found under LORA_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\lora).

    Override the scan root with the FLUX_LORA_DIR environment variable.
    """
    found = _discover_loras()
    return {
        "dir": str(LORA_DIR),
        "total": len(found),
        "loras": found,
    }


@app.get("/reactor-models")
async def reactor_models() -> dict:
    """
    Return every model weight file found under REACTOR_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\Reactorplus), classified by type.

    Kinds:
      classifier  — face attractiveness / detection classifier (attractive_faces_celebs_detection/)
      embedding   — textual inversion learned embeddings (learned_embeds*.bin)
      checkpoint  — training snapshots inside checkpoint-*/ subdirectories
      diffusers   — diffusers pipeline component (unet/, vae/, text_encoder/, ...)
      other       — unclassified weight files

    Override the scan root with the FLUX_REACTOR_DIR environment variable.
    """
    found = _discover_reactor_assets()
    return {
        "dir": str(REACTOR_DIR),
        "total": len(found),
        "by_kind": {
            k: [m for m in found if m["kind"] == k]
            for k in ("classifier", "embedding", "checkpoint", "diffusers", "other")
        },
    }


@app.get("/upscalers")
async def upscalers() -> dict:
    """
    Return every Real-ESRGAN weight file found under REALESRGAN_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\realesrgan), classified by variant.

    Kinds:
      x4plus_anime  — 4× upscaler tuned for anime/animation content
      x4plus        — 4× upscaler for general photographic content
      x2plus        — 2× upscaler
      discriminator — GAN discriminator (training use only)
      other         — unclassified weight files

    Override the scan root with the FLUX_REALESRGAN_DIR environment variable.
    """
    found = _discover_realesrgan_models()
    return {
        "dir": str(REALESRGAN_DIR),
        "total": len(found),
        "by_kind": {
            k: [m for m in found if m["kind"] == k]
            for k in ("x4plus_anime", "x4plus", "x2plus", "discriminator", "other")
        },
    }


@app.get("/sd-models")
async def sd_models() -> dict:
    """
    Return Stable Diffusion pipelines and extension models found under SD_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\Stable-diffusion).

    Pipelines are grouped by architecture variant:
      sd1.5       — Stable Diffusion v1.5 family
      sd2.x       — Stable Diffusion v2.x family
      sd3         — Stable Diffusion 3.x family
      sdxl        — Stable Diffusion XL
      svd         — Stable Video Diffusion (img2vid)
      sv3d_sv4d   — SV3D / SV4D multi-view video
      other       — unclassified pipelines

    Extensions lists .pth weight files found under extensions/.

    Override the scan root with the FLUX_SD_DIR environment variable.
    """
    pipelines = _discover_sd_pipelines()
    extensions = _discover_sd_extensions()
    _variants = ("sd1.5", "sd2.x", "sd3", "sdxl", "svd", "sv3d_sv4d", "other")
    return {
        "dir": str(SD_DIR),
        "pipelines": {
            "total": len(pipelines),
            "by_variant": {
                v: [p for p in pipelines if p["variant"] == v]
                for v in _variants
            },
        },
        "extensions": {
            "total": len(extensions),
            "models": extensions,
        },
    }


@app.get("/wan2-models")
async def wan2_models() -> dict:
    """
    Return Wan2.x video generation model directories found under WAN2_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\WAN2-x).

    Tasks:
      text_to_video        — T2V models (Wan2.1-T2V-*)
      image_to_video       — I2V models (Wan2.2-I2V-*)
      text_image_to_video  — TI2V models (Wan2.2-TI2V-*)
      noise_model          — high_noise_model / low_noise_model directories
      lora                 — LoRA collections (Wan2.2-LoRAs)
      other                — unclassified directories (MoE variants, etc.)

    Each entry includes format (diffusers | native | unknown), detected version,
    size, present pipeline components, root weight files, and transformer shard count.

    Override the scan root with the FLUX_WAN2_DIR environment variable.
    """
    found = _discover_wan2_models()
    _tasks = ("text_to_video", "image_to_video", "text_image_to_video", "noise_model", "lora", "other")
    return {
        "dir": str(WAN2_DIR),
        "total": len(found),
        "by_task": {
            t: [m for m in found if m["task"] == t]
            for t in _tasks
        },
    }


@app.get("/zimage-models")
async def zimage_models() -> dict:
    """
    Return Z-Image model directories found under ZIMAGE_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\ZImage).

    Variants:
      base   — standard text-to-image pipeline (Z-Image)
      edit   — image editing pipeline (z-image-edit)
      turbo  — distilled fast pipeline (Z-Image-Turbo)
      omni   — multimodal base pipeline (Z-Image-Omni-*)

    Override the scan root with the FLUX_ZIMAGE_DIR environment variable.
    """
    found = _discover_zimage_models()
    return {
        "dir": str(ZIMAGE_DIR),
        "total": len(found),
        "by_variant": {
            v: [m for m in found if m["variant"] == v]
            for v in ("base", "edit", "turbo", "omni", "other")
        },
    }


@app.get("/vae-models")
async def vae_models() -> dict:
    """
    Return VAE model files found under VAE_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\VAE), classified by kind.

    Kinds:
      flux       — FLUX.1 autoencoder (ae.safetensors, FLUX.1-* subdirs)
      wan2       — Wan2.x VAE (.pth files: Wan2.1_VAE.pth, Wan2.2_VAE.pth)
      sdxl       — Stable Diffusion XL VAE (sd_xl_*, stable-diffusion-xl-* subdirs)
      onnx       — ONNX format VAE models (.onnx)
      community  — named community VAEs (filenames starting with VAE_)
      sd         — generic Stable Diffusion VAE weights

    Override the scan root with the FLUX_VAE_DIR environment variable.
    """
    found = _discover_vae_models()
    return {
        "dir": str(VAE_DIR),
        "total": len(found),
        "by_kind": {
            k: [m for m in found if m["kind"] == k]
            for k in ("flux", "wan2", "sdxl", "onnx", "community", "sd")
        },
    }


@app.get("/text-encoders")
async def text_encoders() -> dict:
    """
    Return text encoder weight files found in TEXT_ENC_DIR
    (default: C:\\UI\\Experimental-UI_Reit\\models\\text_encoder), classified by type.

    Kinds:
      clip_l    — CLIP-L encoder (used by FLUX, SDXL, SD3)
      clip_g    — CLIP-G encoder (used by SDXL, SD3)
      t5        — T5-XXL encoder; fp8 and fp16 variants
      te1       — text_encoder primary (single-file or sharded sets)
      te2       — text_encoder_2 (single-file or sharded sets)
      te3       — text_encoder_3 (config + sharded safetensors)
      openvino  — OpenVINO binary weights (.bin)
      other     — unclassified

    Also returns model_indexes — the FLUX model_index JSON files stored here
    that declare which encoder combination a given FLUX pipeline uses.

    Override the scan root with the FLUX_TEXT_ENC_DIR environment variable.
    """
    result = _discover_text_encoders()
    return {
        "dir": str(TEXT_ENC_DIR),
        **result,
    }


@app.post("/generate", response_model=ImageResponse)
async def generate(req: GenerateRequest) -> ImageResponse:
    """Text-to-image.  model = schnell | dev | dev-uncensored"""
    seed = _random_seed(req.seed)
    steps = req.steps if req.steps is not None else (4 if req.model == "schnell" else 20)
    log.info(
        "generate model=%s  %dx%d  steps=%d  seed=%d  guidance=%.1f  lora=%s",
        req.model, req.width, req.height, steps, seed, req.guidance_scale, req.use_lora,
    )
    try:
        if req.model == "schnell":
            if not SCHNELL_DIR.exists():
                raise HTTPException(503, detail=f"FLUX.1-schnell-onnx not found: {SCHNELL_DIR}")
            img = _run_schnell(req.prompt, req.width, req.height, steps, req.guidance_scale, seed, req.use_lora)
        elif req.model == "dev":
            if not DEV_ONNX_DIR.exists():
                raise HTTPException(503, detail=f"FLUX.1-dev-onnx not found: {DEV_ONNX_DIR}")
            img = _run_dev_onnx(req.prompt, req.width, req.height, steps, req.guidance_scale, seed, req.use_lora)
        else:  # dev-uncensored
            if not DEV_UNCENSORED_DIR.exists():
                raise HTTPException(503, detail=f"flux.1-dev-uncensored-q4 not found: {DEV_UNCENSORED_DIR}")
            img = _run_dev_uncensored(req.prompt, req.width, req.height, steps, req.guidance_scale, seed, req.use_lora)
        return ImageResponse(image_base64=_pil_to_b64(img))
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/edit", response_model=ImageResponse)
async def edit(req: EditRequest) -> ImageResponse:
    """Image editing with FLUX.1-Kontext-dev."""
    if not KONTEXT_DIR.exists():
        raise HTTPException(503, detail=f"FLUX.1-Kontext-dev-onnx not found: {KONTEXT_DIR}")
    if not _component(KONTEXT_DIR, "vae_encoder").exists():
        raise HTTPException(
            503,
            detail=(
                "FLUX.1-Kontext-dev VAE encoder is not yet available.\n"
                f"Expected: {_component(KONTEXT_DIR, 'vae_encoder')}\n"
                "Download vae_encoder.opt/model.onnx and place it in the Kontext directory."
            ),
        )
    try:
        source = _b64_to_pil(req.image_base64)
        width = req.width or source.width
        height = req.height or source.height
        seed = _random_seed(req.seed)
        log.info("edit  %dx%d  steps=%d  seed=%d  lora=%s", width, height, req.steps, seed, req.use_lora)
        img = _run_kontext(req.prompt, source, width, height, req.steps, req.guidance_scale, seed, req.use_lora)
        return ImageResponse(image_base64=_pil_to_b64(img))
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("edit failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detect-faces", response_model=DetectFacesResponse)
async def detect_faces(req: DetectFacesRequest) -> DetectFacesResponse:
    """Face bounding-box detection used by auto-reframe."""
    try:
        return DetectFacesResponse(faces=_detect_faces_heuristic(_b64_to_pil(req.image_base64)))
    except Exception as exc:
        log.exception("detect-faces failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== FLUX Inference Server v2.0 -- startup model check ===")
    _validate_model_paths(SCHNELL_DIR, "schnell", SCHNELL_PRECISION)
    _validate_model_paths(DEV_ONNX_DIR, "dev-onnx", DEV_ONNX_PRECISION)
    _validate_model_paths(KONTEXT_DIR, "kontext", KONTEXT_PRECISION, has_vae_enc=True)
    log.info(
        "[uncensored  ] %s  %s",
        "OK     " if DEV_UNCENSORED_DIR.exists() else "MISSING",
        DEV_UNCENSORED_DIR,
    )
    log.info(
        "[lora        ] %s  %s  scale=%.2f",
        "OK     " if LORA_PATH.exists() else "MISSING",
        LORA_PATH.name,
        LORA_SCALE,
    )
    log.info(
        "[gguf-t5     ] %s  %s",
        "OK     " if GGUF_T5_PATH.exists() else "MISSING",
        GGUF_T5_PATH.name,
    )
    log.info("[tok-clip    ] %s", FLUX_DEV_DIR / "tokenizer")
    log.info("[tok-t5      ] %s", FLUX_DEV_DIR / "tokenizer_2")
    log.info(
        "Active text encoder: %s",
        "GGUF Qwen3-4B" if USE_GGUF_T5 else ("T5-fp8 ONNX" if USE_FP8_T5 else "T5-full ONNX"),
    )
    _pth = _discover_pth_models()
    _rvc = [m for m in _pth if m["kind"] == "rvc"]
    _ext = [m for m in _pth if m["kind"] == "extension"]
    _oth = [m for m in _pth if m["kind"] == "other"]
    log.info(
        "Audio .pth models: %d total  (%d rvc, %d extension, %d other)",
        len(_pth), len(_rvc), len(_ext), len(_oth),
    )
    for m in _rvc:
        log.info("  [rvc      ] %-45s  %.1f MB  index=%s", m["name"], m["size_mb"] or 0, m["has_index"])
    for m in _ext:
        log.info("  [extension] %-45s  %.1f MB", m["name"], m["size_mb"] or 0)
    _sd_pipelines = _discover_sd_pipelines()
    log.info("SD pipelines: %d total", len(_sd_pipelines))
    for p in _sd_pipelines:
        log.info("  [sd %-10s] %-40s  %d components  %d checkpoints",
                 p["variant"], p["name"], len(p["components"]), len(p["checkpoints"]))
    _wan2 = _discover_wan2_models()
    log.info("WAN2 models: %d total", len(_wan2))
    for m in _wan2:
        log.info("  [wan2 %-8s] %-40s  fmt=%-10s  ver=%s  size=%s",
                 m["task"][:8], m["name"], m["format"],
                 m["version"] or "?", m["size"] or "?")
    _zi = _discover_zimage_models()
    log.info("Z-Image models: %d total", len(_zi))
    for m in _zi:
        log.info("  [zimage %-6s] %-40s  fmt=%s  %d components",
                 m["variant"], m["name"], m["format"], len(m["components"]))
    _vae = _discover_vae_models()
    log.info("VAE models: %d total  (flux=%d wan2=%d sdxl=%d onnx=%d community=%d sd=%d)",
             len(_vae),
             sum(1 for m in _vae if m["kind"] == "flux"),
             sum(1 for m in _vae if m["kind"] == "wan2"),
             sum(1 for m in _vae if m["kind"] == "sdxl"),
             sum(1 for m in _vae if m["kind"] == "onnx"),
             sum(1 for m in _vae if m["kind"] == "community"),
             sum(1 for m in _vae if m["kind"] == "sd"))
    _te = _discover_text_encoders()
    log.info("Text encoders: %d files  model_indexes=%d  kinds=%s",
             len(_te["encoders"]), len(_te["model_indexes"]),
             ", ".join(f"{k}={v}" for k, v in _te["by_kind"].items()))
    log.info("=========================================================")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
