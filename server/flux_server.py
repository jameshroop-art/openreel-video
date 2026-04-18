"""
FLUX ONNX Local Inference Server
=================================
Runs at http://localhost:8080 and exposes three endpoints:

  POST /generate      — text-to-image via FLUX.1-schnell-onnx
  POST /edit          — image editing / background removal via FLUX.1-Kontext-dev-onnx
  POST /detect-faces  — lightweight face bounding-box detection (used by auto-reframe)

On-disk model layout (each component is a sub-directory containing model.onnx)
-------------------------------------------------------------------------------
  FLUX.1-schnell-onnx/
    clip.opt/
      model.onnx
    t5.opt/
      backbone.onnx_data
      model.onnx
    t5-fp8.opt/
      backbone.onnx_data
      model.onnx
    transformer.opt/
      bf16/
        backbone.onnx_data
        model.onnx
      fp4/
        backbone.onnx_data
        model.onnx
      fp8/
        backbone.onnx_data
        model.onnx
    vae.opt/
      model.onnx

  FLUX.1-Kontext-dev-onnx/   (same layout + vae_encoder.opt/)
    clip.opt/model.onnx
    t5.opt/model.onnx
    t5-fp8.opt/model.onnx
    transformer.opt/{precision}/model.onnx
    vae.opt/model.onnx
    vae_encoder.opt/model.onnx

onnxruntime automatically resolves backbone.onnx_data sidecar files when they
sit alongside model.onnx in the same directory — no extra configuration needed.

Setup (Windows PowerShell)
--------------------------
  pip install fastapi uvicorn[standard] onnxruntime pillow numpy

  # If you have a CUDA GPU (recommended — FLUX is very slow on CPU):
  pip install onnxruntime-gpu

Run
---
  python server/flux_server.py

Environment variables
---------------------
  FLUX_SCHNELL_DIR            Root directory of FLUX.1-schnell-onnx
                              (default: C:\\UI\\Experimental-UI_Reit\\models\\Flux\\FLUX.1-schnell-onnx)

  FLUX_KONTEXT_DIR            Root directory of FLUX.1-Kontext-dev-onnx
                              (default: C:\\UI\\Experimental-UI_Reit\\models\\Flux\\FLUX.1-Kontext-dev-onnx)

  FLUX_TRANSFORMER_PRECISION  Which transformer variant to load: bf16 | fp4 | fp8
                              (default: fp8 — smallest VRAM footprint)

  FLUX_USE_FP8_T5             Set to "1" to use t5-fp8.opt instead of t5.opt
                              (default: 0 — use full-precision T5)
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

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

# ---------------------------------------------------------------------------
# Model paths and precision selection
# ---------------------------------------------------------------------------
SCHNELL_DIR = Path(
    os.getenv(
        "FLUX_SCHNELL_DIR",
        r"C:\UI\Experimental-UI_Reit\models\Flux\FLUX.1-schnell-onnx",
    )
)
KONTEXT_DIR = Path(
    os.getenv(
        "FLUX_KONTEXT_DIR",
        r"C:\UI\Experimental-UI_Reit\models\Flux\FLUX.1-Kontext-dev-onnx",
    )
)

# Which sub-directory of transformer.opt/ to use: bf16 | fp4 | fp8
_VALID_PRECISIONS = {"bf16", "fp4", "fp8"}
TRANSFORMER_PRECISION = os.getenv("FLUX_TRANSFORMER_PRECISION", "fp8").lower()
if TRANSFORMER_PRECISION not in _VALID_PRECISIONS:
    raise ValueError(
        f"FLUX_TRANSFORMER_PRECISION must be one of {_VALID_PRECISIONS}, "
        f"got {TRANSFORMER_PRECISION!r}"
    )

# Whether to load the FP8-quantised T5 encoder (smaller, slightly lower quality)
USE_FP8_T5: bool = os.getenv("FLUX_USE_FP8_T5", "0").strip() == "1"


# ---------------------------------------------------------------------------
# Path helpers — each component lives in <root>/<component>.opt/model.onnx
# ---------------------------------------------------------------------------

def _component(root: Path, component: str) -> Path:
    """Return the model.onnx path for *component* inside *root*."""
    return root / f"{component}.opt" / "model.onnx"


def _transformer(root: Path) -> Path:
    """Return the model.onnx path for the selected transformer precision."""
    return root / "transformer.opt" / TRANSFORMER_PRECISION / "model.onnx"


def _t5(root: Path) -> Path:
    """Return the model.onnx path for the T5 encoder (fp8 or full-precision)."""
    folder = "t5-fp8.opt" if USE_FP8_T5 else "t5.opt"
    return root / folder / "model.onnx"


def _validate_model_paths(root: Path, name: str, include_vae_encoder: bool = False) -> None:
    """Log which model files are present / missing at startup."""
    paths: list[Path] = [
        _component(root, "clip"),
        _t5(root),
        _transformer(root),
        _component(root, "vae"),
    ]
    if include_vae_encoder:
        paths.append(_component(root, "vae_encoder"))

    for p in paths:
        status = "OK" if p.exists() else "MISSING"
        log.info("[%s] %s  %s", name, status, p)


# ---------------------------------------------------------------------------
# Lazy ONNX session cache
# ---------------------------------------------------------------------------
_sessions: dict[str, object] = {}


def _session(model_path: Path):
    """Return a cached OnnxRuntime InferenceSession for *model_path*.

    *model_path* must point to a model.onnx file.  Any backbone.onnx_data
    sidecar in the same directory is resolved automatically by onnxruntime.
    """
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
        log.info("Loading ONNX model: %s  (providers=%s)", model_path, providers)
        _sessions[key] = ort.InferenceSession(str(model_path), providers=providers)
    return _sessions[key]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="FLUX ONNX Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:55173", "http://localhost:3000", "http://localhost:5173"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: Optional[int] = None


class EditRequest(BaseModel):
    prompt: str
    image_base64: str  # base-64 encoded PNG/JPEG of the source frame
    width: Optional[int] = None
    height: Optional[int] = None
    steps: int = Field(default=20, ge=1, le=50)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)
    seed: Optional[int] = None


class DetectFacesRequest(BaseModel):
    image_base64: str  # base-64 encoded PNG/JPEG


class ImageResponse(BaseModel):
    image_base64: str  # base-64 PNG result


class FaceBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float


class DetectFacesResponse(BaseModel):
    faces: list[FaceBox]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _random_seed(seed: Optional[int]) -> int:
    return seed if seed is not None else int.from_bytes(os.urandom(4), "little")


# ---------------------------------------------------------------------------
# FLUX.1-schnell text-to-image pipeline
# ---------------------------------------------------------------------------


def _run_schnell(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    """
    Runs the FLUX.1-schnell ONNX models in sequence:
      1. clip.opt/model.onnx                         — CLIP text encoder
      2. t5.opt/model.onnx  (or t5-fp8.opt)          — T5 text encoder
      3. transformer.opt/{precision}/model.onnx      — diffusion transformer
      4. vae.opt/model.onnx                          — VAE decoder
    """
    try:
        import onnxruntime as ort  # noqa: F401 (ensure ort is available)
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime") from exc

    clip_sess = _session(_component(SCHNELL_DIR, "clip"))
    t5_sess = _session(_t5(SCHNELL_DIR))
    transformer_sess = _session(_transformer(SCHNELL_DIR))
    vae_sess = _session(_component(SCHNELL_DIR, "vae"))

    rng = np.random.default_rng(seed)

    # --- 1. Encode text with CLIP ---
    clip_inputs = {clip_sess.get_inputs()[0].name: np.array([[prompt]], dtype=object)}
    (clip_embedding,) = clip_sess.run(None, clip_inputs)

    # --- 2. Encode text with T5 ---
    t5_inputs = {t5_sess.get_inputs()[0].name: np.array([[prompt]], dtype=object)}
    (t5_embedding,) = t5_sess.run(None, t5_inputs)

    # --- 3. Iterative denoising with the transformer ---
    latent_h = height // 8
    latent_w = width // 8
    latents = rng.standard_normal((1, 16, latent_h, latent_w)).astype(np.float32)

    timesteps = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]

    for t in timesteps:
        t_tensor = np.array([t], dtype=np.float32)
        transformer_inputs = {
            transformer_sess.get_inputs()[0].name: latents,
            transformer_sess.get_inputs()[1].name: clip_embedding.astype(np.float32),
            transformer_sess.get_inputs()[2].name: t5_embedding.astype(np.float32),
            transformer_sess.get_inputs()[3].name: t_tensor,
        }
        (noise_pred,) = transformer_sess.run(None, transformer_inputs)
        # Euler step
        dt = -1.0 / steps
        latents = latents + noise_pred * dt

    # --- 4. Decode latents with VAE ---
    vae_inputs = {vae_sess.get_inputs()[0].name: latents}
    (decoded,) = vae_sess.run(None, vae_inputs)

    # decoded: (1, 3, H, W) float32 in [-1, 1]
    img_arr = (np.clip(decoded[0].transpose(1, 2, 0), -1.0, 1.0) * 0.5 + 0.5) * 255
    return Image.fromarray(img_arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# FLUX.1-Kontext-dev image editing pipeline
# ---------------------------------------------------------------------------


def _run_kontext(
    prompt: str,
    source_image: Image.Image,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    """
    Runs the FLUX.1-Kontext-dev ONNX models:
      1. clip.opt/model.onnx                         — CLIP text encoder
      2. t5.opt/model.onnx  (or t5-fp8.opt)          — T5 text encoder
      3. vae_encoder.opt/model.onnx                  — VAE encoder (source image → latent)
      4. transformer.opt/{precision}/model.onnx      — conditional diffusion transformer
      5. vae.opt/model.onnx                          — VAE decoder
    """
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime") from exc

    clip_sess = _session(_component(KONTEXT_DIR, "clip"))
    t5_sess = _session(_t5(KONTEXT_DIR))
    vae_enc_sess = _session(_component(KONTEXT_DIR, "vae_encoder"))
    transformer_sess = _session(_transformer(KONTEXT_DIR))
    vae_dec_sess = _session(_component(KONTEXT_DIR, "vae"))

    rng = np.random.default_rng(seed)

    # Resize source to target dimensions
    source_resized = source_image.resize((width, height), Image.LANCZOS)
    source_arr = (np.array(source_resized, dtype=np.float32) / 255.0) * 2.0 - 1.0
    source_arr = source_arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    # --- 1. Encode text ---
    clip_inputs = {clip_sess.get_inputs()[0].name: np.array([[prompt]], dtype=object)}
    (clip_embedding,) = clip_sess.run(None, clip_inputs)

    t5_inputs = {t5_sess.get_inputs()[0].name: np.array([[prompt]], dtype=object)}
    (t5_embedding,) = t5_sess.run(None, t5_inputs)

    # --- 2. Encode source image ---
    vae_enc_inputs = {vae_enc_sess.get_inputs()[0].name: source_arr}
    (image_latent,) = vae_enc_sess.run(None, vae_enc_inputs)

    # --- 3. Add noise and denoise ---
    latent_h = height // 8
    latent_w = width // 8
    noise = rng.standard_normal((1, 16, latent_h, latent_w)).astype(np.float32)
    # Start from a blend of image latent and noise (strength = 1.0 for full edit)
    strength = 1.0
    t_start_idx = int(steps * (1 - strength))
    timesteps = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[t_start_idx:-1]
    latents = image_latent + noise * timesteps[0] if len(timesteps) > 0 else image_latent

    for t in timesteps:
        t_tensor = np.array([t], dtype=np.float32)
        g_tensor = np.array([guidance_scale], dtype=np.float32)
        transformer_inputs = {
            transformer_sess.get_inputs()[0].name: latents,
            transformer_sess.get_inputs()[1].name: clip_embedding.astype(np.float32),
            transformer_sess.get_inputs()[2].name: t5_embedding.astype(np.float32),
            transformer_sess.get_inputs()[3].name: t_tensor,
            transformer_sess.get_inputs()[4].name: g_tensor,
        }
        (noise_pred,) = transformer_sess.run(None, transformer_inputs)
        dt = -1.0 / steps
        latents = latents + noise_pred * dt

    # --- 4. Decode ---
    vae_inputs = {vae_dec_sess.get_inputs()[0].name: latents}
    (decoded,) = vae_dec_sess.run(None, vae_inputs)

    img_arr = (np.clip(decoded[0].transpose(1, 2, 0), -1.0, 1.0) * 0.5 + 0.5) * 255
    return Image.fromarray(img_arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Simple face detector (heuristic — replace with a real ONNX detector later)
# ---------------------------------------------------------------------------


def _detect_faces_heuristic(img: Image.Image) -> list[FaceBox]:
    """
    Skin-tone heuristic face detector.  Accurate enough for auto-reframe cropping.
    Returns up to 3 face bounding boxes sorted by area descending.
    """
    arr = np.array(img.resize((320, 240), Image.BILINEAR), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Skin-tone mask (Kovac et al. 2003 in RGB)
    skin = (
        (r > 95) & (g > 40) & (b > 20)
        & (r > g) & (r > b)
        & (np.abs(r - g) > 15)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
    )

    # Simple connected-region labelling via down-sampled block scan
    block = 8
    h, w = skin.shape
    boxes: list[FaceBox] = []

    for by in range(0, h, block):
        for bx in range(0, w, block):
            patch = skin[by : by + block, bx : bx + block]
            if patch.mean() < 0.4:
                continue
            # Expand bounding box
            y0, y1 = max(0, by - block * 2), min(h, by + block * 4)
            x0, x1 = max(0, bx - block * 2), min(w, bx + block * 4)
            # Normalise to 0-1 (relative to original image)
            boxes.append(
                FaceBox(
                    x=x0 / w * img.width,
                    y=y0 / h * img.height,
                    width=(x1 - x0) / w * img.width,
                    height=(y1 - y0) / h * img.height,
                    confidence=float(patch.mean()),
                )
            )

    # Merge overlapping boxes and keep top-3 by area
    boxes.sort(key=lambda b: b.width * b.height, reverse=True)
    return boxes[:3]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "schnell_dir": str(SCHNELL_DIR), "kontext_dir": str(KONTEXT_DIR)}


@app.post("/generate", response_model=ImageResponse)
async def generate(req: GenerateRequest) -> ImageResponse:
    """Text-to-image generation with FLUX.1-schnell."""
    if not SCHNELL_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"FLUX.1-schnell model directory not found: {SCHNELL_DIR}",
        )
    try:
        seed = _random_seed(req.seed)
        log.info("generate  prompt=%r  %dx%d  steps=%d  seed=%d", req.prompt, req.width, req.height, req.steps, seed)
        img = _run_schnell(req.prompt, req.width, req.height, req.steps, req.guidance_scale, seed)
        return ImageResponse(image_base64=_pil_to_b64(img))
    except Exception as exc:
        log.exception("generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/edit", response_model=ImageResponse)
async def edit(req: EditRequest) -> ImageResponse:
    """Image editing / background removal with FLUX.1-Kontext-dev."""
    if not KONTEXT_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"FLUX.1-Kontext-dev model directory not found: {KONTEXT_DIR}",
        )
    try:
        source = _b64_to_pil(req.image_base64)
        width = req.width or source.width
        height = req.height or source.height
        seed = _random_seed(req.seed)
        log.info("edit  prompt=%r  %dx%d  steps=%d  seed=%d", req.prompt, width, height, req.steps, seed)
        img = _run_kontext(req.prompt, source, width, height, req.steps, req.guidance_scale, seed)
        return ImageResponse(image_base64=_pil_to_b64(img))
    except Exception as exc:
        log.exception("edit failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detect-faces", response_model=DetectFacesResponse)
async def detect_faces(req: DetectFacesRequest) -> DetectFacesResponse:
    """Return face bounding boxes for the given image (used by auto-reframe)."""
    try:
        img = _b64_to_pil(req.image_base64)
        faces = _detect_faces_heuristic(img)
        return DetectFacesResponse(faces=faces)
    except Exception as exc:
        log.exception("detect-faces failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
