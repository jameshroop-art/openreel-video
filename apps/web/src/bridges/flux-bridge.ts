/**
 * FluxBridge
 *
 * Singleton bridge that calls the local FLUX ONNX inference server running at
 * http://localhost:8080 (see server/flux_server.py).
 *
 * Exposes:
 *   generate(prompt, width, height, steps) → File (PNG) suitable for MediaBridge.importFile()
 *   edit(prompt, imageBase64)              → File (PNG) suitable for MediaBridge.importFile()
 *
 * Both methods throw if the server is unreachable so callers should catch and
 * surface appropriate UI feedback via the flux-store.
 */

const FLUX_SERVER_BASE = "http://localhost:8080";

export interface FluxGenerateOptions {
  prompt: string;
  width?: number;
  height?: number;
  /** Number of denoising steps. FLUX.1-schnell is designed for 1–4. */
  steps?: number;
  /** Classifier-free guidance scale. 0 for schnell (distilled), 3.5 for kontext. */
  guidanceScale?: number;
  /** Optional fixed seed for reproducibility. */
  seed?: number;
}

export interface FluxEditOptions {
  prompt: string;
  /** Base-64 encoded source image (PNG or JPEG). */
  imageBase64: string;
  width?: number;
  height?: number;
  steps?: number;
  guidanceScale?: number;
  seed?: number;
}

export interface FluxFaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export class FluxBridge {
  private serverBase: string;

  constructor(serverBase = FLUX_SERVER_BASE) {
    this.serverBase = serverBase;
  }

  /**
   * Ping the inference server health endpoint.
   * Returns true if the server is reachable and healthy.
   */
  async isServerAvailable(): Promise<boolean> {
    try {
      const res = await fetch(`${this.serverBase}/health`, {
        signal: AbortSignal.timeout(2000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Text-to-image generation using FLUX.1-schnell.
   *
   * @returns A `File` object that can be passed directly to `MediaBridge.importFile()`.
   */
  async generate(options: FluxGenerateOptions): Promise<File> {
    const body = {
      prompt: options.prompt,
      width: options.width ?? 1024,
      height: options.height ?? 1024,
      steps: options.steps ?? 4,
      guidance_scale: options.guidanceScale ?? 0.0,
      seed: options.seed ?? null,
    };

    const res = await fetch(`${this.serverBase}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`FLUX generate failed (${res.status}): ${detail}`);
    }

    const { image_base64 } = (await res.json()) as { image_base64: string };
    return this.base64ToFile(image_base64, "flux_generated.png", "image/png");
  }

  /**
   * Image editing using FLUX.1-Kontext-dev.
   * Useful for background removal, object editing, and compositing.
   *
   * @returns A `File` object that can be passed directly to `MediaBridge.importFile()`.
   */
  async edit(options: FluxEditOptions): Promise<File> {
    const body = {
      prompt: options.prompt,
      image_base64: options.imageBase64,
      width: options.width ?? null,
      height: options.height ?? null,
      steps: options.steps ?? 20,
      guidance_scale: options.guidanceScale ?? 3.5,
      seed: options.seed ?? null,
    };

    const res = await fetch(`${this.serverBase}/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`FLUX edit failed (${res.status}): ${detail}`);
    }

    const { image_base64 } = (await res.json()) as { image_base64: string };
    return this.base64ToFile(image_base64, "flux_edited.png", "image/png");
  }

  /**
   * Request face bounding-box detection for an image from the local server.
   * Used by `AutoReframeEngine` as a more accurate alternative to its heuristic.
   *
   * @param imageBase64 - Base-64 encoded source image (PNG or JPEG).
   */
  async detectFaces(imageBase64: string): Promise<FluxFaceBox[]> {
    const res = await fetch(`${this.serverBase}/detect-faces`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64 }),
    });

    if (!res.ok) {
      throw new Error(`detectFaces failed (${res.status}): ${res.statusText}`);
    }

    const { faces } = (await res.json()) as { faces: FluxFaceBox[] };
    return faces;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private base64ToFile(b64: string, filename: string, mimeType: string): File {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new File([bytes], filename, { type: mimeType });
  }
}

// ---------------------------------------------------------------------------
// Singleton management (mirrors the pattern used by all other bridges)
// ---------------------------------------------------------------------------

let fluxBridgeInstance: FluxBridge | null = null;

export function getFluxBridge(): FluxBridge | null {
  return fluxBridgeInstance;
}

export function initializeFluxBridge(serverBase?: string): FluxBridge {
  if (!fluxBridgeInstance) {
    fluxBridgeInstance = new FluxBridge(serverBase);
  }
  return fluxBridgeInstance;
}

export function disposeFluxBridge(): void {
  fluxBridgeInstance = null;
}
