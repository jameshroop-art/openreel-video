/**
 * FLUX WASM / onnxruntime-web browser-runnable path
 *
 * This module is a forward-compatible stub for running a quantized FLUX
 * model entirely in the browser via onnxruntime-web (WebGPU or WASM backend).
 *
 * Current state
 * -------------
 * Because the full FLUX transformer is 17–23 GB, the in-browser path requires
 * a heavily quantized variant (INT4 / FP8).  Until such a model is bundled,
 * `isWasmFluxAvailable()` returns false and the bridge falls back to the local
 * inference server (see apps/web/src/bridges/flux-bridge.ts).
 *
 * How to enable the browser path
 * --------------------------------
 * 1. Install onnxruntime-web:
 *      pnpm add onnxruntime-web -w --filter @openreel/core
 * 2. Place your quantized model files under:
 *      packages/core/src/wasm/flux/models/
 *        clip.onnx
 *        t5_int8.onnx
 *        transformer_int4.onnx
 *        vae.onnx
 * 3. Update `loadWasmFlux()` below to load those files.
 * 4. The `WasmModuleStatus` in wasm/index.ts will then report "ready" for flux.
 */

export interface FluxWasmSession {
  /** Generate an image from a text prompt, returns raw PNG bytes. */
  generate(prompt: string, width: number, height: number, steps: number): Promise<Uint8Array>;
}

let wasmFluxSession: FluxWasmSession | null = null;
let loadPromise: Promise<FluxWasmSession | null> | null = null;

async function loadWasmFlux(): Promise<FluxWasmSession | null> {
  // Guard: onnxruntime-web is only loaded on demand to avoid bundle bloat.
  // When the quantized models are available, uncomment and adapt the block below.
  //
  // try {
  //   const ort = await import("onnxruntime-web");
  //   ort.env.wasm.wasmPaths = "/ort-wasm/";
  //
  //   const clipUrl    = new URL("./models/clip.onnx",                import.meta.url);
  //   const t5Url      = new URL("./models/t5_int8.onnx",             import.meta.url);
  //   const transUrl   = new URL("./models/transformer_int4.onnx",    import.meta.url);
  //   const vaeUrl     = new URL("./models/vae.onnx",                 import.meta.url);
  //
  //   const [clipSess, t5Sess, transSess, vaeSess] = await Promise.all([
  //     ort.InferenceSession.create(clipUrl.href,  { executionProviders: ["webgpu", "wasm"] }),
  //     ort.InferenceSession.create(t5Url.href,    { executionProviders: ["webgpu", "wasm"] }),
  //     ort.InferenceSession.create(transUrl.href, { executionProviders: ["webgpu", "wasm"] }),
  //     ort.InferenceSession.create(vaeUrl.href,   { executionProviders: ["webgpu", "wasm"] }),
  //   ]);
  //
  //   return {
  //     async generate(prompt, width, height, steps) {
  //       // TODO: implement full pipeline using clipSess, t5Sess, transSess, vaeSess
  //       void [prompt, width, height, steps, clipSess, t5Sess, transSess, vaeSess];
  //       throw new Error("Browser FLUX pipeline not yet implemented");
  //     },
  //   };
  // } catch {
  //   return null;
  // }

  return null; // Browser path not yet available — use local server instead.
}

/**
 * Attempt to initialise the in-browser FLUX ONNX session.
 * Returns true if the session is ready for use.
 */
export async function initWasmFlux(): Promise<boolean> {
  if (wasmFluxSession) return true;
  if (!loadPromise) {
    loadPromise = loadWasmFlux();
  }
  wasmFluxSession = await loadPromise;
  return wasmFluxSession !== null;
}

/** Returns true if the in-browser FLUX session has been successfully loaded. */
export function isWasmFluxAvailable(): boolean {
  return wasmFluxSession !== null;
}

/**
 * Returns the active in-browser FLUX session, or null if unavailable.
 * Callers should fall back to the local server when this returns null.
 */
export function getWasmFluxSession(): FluxWasmSession | null {
  return wasmFluxSession;
}
