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
    log.info("=========================================================")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
