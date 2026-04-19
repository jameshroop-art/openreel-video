/**
 * Flux generation state store
 *
 * Tracks the lifecycle of a FLUX image / edit generation request so that
 * progress and errors persist across dialog closes:
 *
 *   idle → generating → done      (happy path)
 *   idle → generating → error     (server error / network failure)
 *   error → (user retries) → idle
 *
 * The resulting media item ID is stored so the calling component can add
 * the generated image to the timeline via MediaBridge / ProjectStore.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type FluxGenerationStatus = "idle" | "generating" | "done" | "error";

export interface FluxState {
  /** Current lifecycle status. */
  status: FluxGenerationStatus;
  /** The text prompt used for the last generation. */
  prompt: string;
  /** 0–100 progress indication (best-effort; set to 50 while waiting for server). */
  progress: number;
  /** Human-readable error message when status === "error". */
  error: string | null;
  /** ID of the MediaItem that was imported after a successful generation. */
  resultMediaId: string | null;
  /** Project the result belongs to (used to scope access). */
  projectId: string | null;

  // Actions
  startGeneration: (prompt: string, projectId: string) => void;
  setProgress: (progress: number) => void;
  completeGeneration: (mediaId: string) => void;
  failGeneration: (error: string) => void;
  reset: () => void;
}

const INITIAL_STATE: Pick<
  FluxState,
  "status" | "prompt" | "progress" | "error" | "resultMediaId" | "projectId"
> = {
  status: "idle",
  prompt: "",
  progress: 0,
  error: null,
  resultMediaId: null,
  projectId: null,
};

export const useFluxStore = create<FluxState>()(
  persist(
    (set) => ({
      ...INITIAL_STATE,

      startGeneration: (prompt: string, projectId: string) =>
        set({
          status: "generating",
          prompt,
          projectId,
          progress: 0,
          error: null,
          resultMediaId: null,
        }),

      setProgress: (progress: number) => set({ progress }),

      completeGeneration: (mediaId: string) =>
        set({ status: "done", progress: 100, resultMediaId: mediaId }),

      failGeneration: (error: string) =>
        set({ status: "error", error, progress: 0 }),

      reset: () => set(INITIAL_STATE),
    }),
    { name: "flux-generation-state" },
  ),
);
