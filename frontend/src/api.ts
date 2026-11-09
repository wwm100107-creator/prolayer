export interface SessionInfo {
  id: string;
  width: number;
  height: number;
}

/** Empty when the page is served by the backend itself. */
const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? "").replace(/\/$/, "");
const KEY_STORAGE = "prolayer-access-key";

export const apiUrl = (path: string) => API_BASE + path;

export function getKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setKey(key: string): void {
  try {
    localStorage.setItem(KEY_STORAGE, key);
  } catch {
    // Private browsing: the key then has to be entered again next visit.
  }
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export interface TextInfo {
  content: string;
  fontFamily: string;
  postScriptName: string;
  fontWeight: number;
  fontSize: number;
  color: [number, number, number];
  angle: number;
  /** Start of the baseline, in image pixels. */
  x: number;
  y: number;
}

export interface LayerInfo {
  id: string;
  name: string;
  left: number;
  top: number;
  width: number;
  height: number;
  blendMode: string;
  url: string;
  kind: "raster" | "shadow" | "text";
  text: TextInfo | null;
}

export interface Prompt {
  points: { x: number; y: number; positive: boolean }[];
  box: [number, number, number, number] | null;
  level: 0 | 1 | 2 | null;
}

export interface SegmentResult {
  maskId: string;
  score: number;
  empty: boolean;
  preview: string;
}

export interface Health {
  ready: boolean;
  device: string;
  gpu: string | null;
  error: string | null;
  keyRequired: boolean;
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return response.json();
  let message = `Máy chủ báo lỗi ${response.status}.`;
  try {
    const body = await response.json();
    if (typeof body.detail === "string") message = body.detail;
  } catch {
    // Non-JSON error page: the status message above is the best available.
  }
  throw new ApiError(message, response.status);
}

function post<T>(url: string, body: unknown): Promise<T> {
  return fetch(apiUrl(url), {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-ProLayer-Key": getKey() },
    body: JSON.stringify(body),
  }).then(unwrap<T>);
}

export const api = {
  health: () => fetch(apiUrl("/api/health")).then(unwrap<Health>),
  upload(file: File) {
    const form = new FormData();
    form.append("file", file);
    return fetch(apiUrl("/api/sessions"), { method: "POST", body: form, headers: { "X-ProLayer-Key": getKey() } }).then(unwrap<SessionInfo>);
  },
  segment: (id: string, prompt: Prompt) => post<SegmentResult>(`/api/sessions/${id}/segment`, prompt),
  extract: (id: string, maskId: string, edge: number, transparent: boolean, shadow: boolean) =>
    post<{ layers: LayerInfo[]; backgroundVersion: number }>(`/api/sessions/${id}/extract`, {
      mask_id: maskId,
      edge,
      transparent,
      shadow,
    }),
  decompose: (id: string, options: { detail: string; edge: number; text: boolean; shadow: boolean }) =>
    post<{ layers: LayerInfo[]; backgroundVersion: number; fidelityPsnr: number }>(
      `/api/sessions/${id}/decompose`,
      options,
    ),
  progress: (id: string) => fetch(apiUrl(`/api/sessions/${id}/progress`)).then(unwrap<{ stage: string; value: number }>),
  layerSvg: (id: string, layerId: string) =>
    fetch(apiUrl(`/api/sessions/${id}/layers/${layerId}/svg`)).then(unwrap<{ svg: string | null }>),
};
