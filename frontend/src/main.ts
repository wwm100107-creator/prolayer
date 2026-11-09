import "@fontsource-variable/archivo/wdth.css";
import "./styles.css";
import { IText } from "fabric";
import { ApiError, api, getKey, setKey, type SessionInfo } from "./api";
import { Editor, type EditorLayer, type Tool } from "./editor";
import { buildPsd, exportLayersZip, exportPng, exportPsd, exportSvg } from "./exporter";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const pasteboard = $<HTMLElement>("pasteboard");
const editor = new Editor($<HTMLCanvasElement>("stage"), pasteboard);
const status = $<HTMLParagraphElement>("status");
const progress = $<HTMLDivElement>("progress");
const fileInput = $<HTMLInputElement>("file");
const selectionBar = $<HTMLFormElement>("selection");
const levelField = $<HTMLFieldSetElement>("level");
const extractButton = $<HTMLButtonElement>("extract");
const edgeInput = $<HTMLInputElement>("edge");
const transparentInput = $<HTMLInputElement>("transparent");
const list = $<HTMLOListElement>("layers");
const empty = $<HTMLParagraphElement>("layers-empty");
const props = $<HTMLDivElement>("layer-props");
const opacityInput = $<HTMLInputElement>("opacity");
const blendInput = $<HTMLSelectElement>("blend");
const stack = $<HTMLDivElement>("stack");

const docButtons = ["compare", "export-png", "export-zip", "export-svg", "export-psd", "auto"].map((id) => $<HTMLButtonElement>(id));

let session: SessionInfo | null = null;
let maskId: string | null = null;
let segmentSeq = 0;
let busy = false;

function say(message: string): void {
  status.textContent = message;
  status.classList.remove("error");
}

const keyDialog = $<HTMLDialogElement>("key-dialog");
const keyInput = $<HTMLInputElement>("key-input");

keyDialog.addEventListener("close", () => {
  if (keyDialog.returnValue === "save" && keyInput.value) {
    setKey(keyInput.value);
    keyInput.value = "";
    say("Đã lưu mã truy cập. Hãy thực hiện lại thao tác vừa rồi.");
  }
});

function askForKey(): void {
  if (!keyDialog.open) keyDialog.showModal();
}

function fail(error: unknown): void {
  if (error instanceof ApiError && error.status === 401) askForKey();
  status.textContent = error instanceof Error ? error.message : String(error);
  status.classList.add("error");
}

function setProgress(value: number | null): void {
  progress.hidden = value === null;
  if (value !== null) (progress.firstElementChild as HTMLElement).style.width = `${Math.round(value * 100)}%`;
}

async function withBusy(message: string, task: () => Promise<void>): Promise<void> {
  if (busy) return;
  busy = true;
  document.body.classList.add("busy");
  say(message);
  try {
    await task();
  } catch (error) {
    fail(error);
  } finally {
    busy = false;
    document.body.classList.remove("busy");
    setProgress(null);
    syncExtract();
  }
}

// ------------------------------------------------------------------ opening

async function openFile(file: File): Promise<void> {
  await withBusy("Đang tải ảnh lên", async () => {
    session = await api.upload(file);
    await editor.open(session);
    $<HTMLElement>("dropzone").hidden = true;
    docButtons.forEach((button) => (button.disabled = false));
    setTool("point");
    say(`${file.name}, ${session.width} × ${session.height} px`);
  });
}

$("pick").addEventListener("click", () => fileInput.click());
$("open").addEventListener("click", () => {
  if (confirmReplace("Mở ảnh khác sẽ bỏ các layer đang có. Tiếp tục?")) fileInput.click();
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) void openFile(file);
  fileInput.value = "";
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer?.files[0];
  if (file && file.type.startsWith("image/") && confirmReplace("Mở ảnh này sẽ bỏ các layer đang có. Tiếp tục?")) {
    void openFile(file);
  }
});

const textField = $<HTMLLabelElement>("text-field");
const textInput = $<HTMLTextAreaElement>("text-content");

function confirmReplace(question: string): boolean {
  return editor.layers.length === 0 || window.confirm(question);
}

// -------------------------------------------------------------------- tools

function setTool(tool: Tool): void {
  editor.setTool(tool);
  document.querySelectorAll<HTMLButtonElement>(".tool").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.tool === tool));
  });
  selectionBar.hidden = tool === "move" || !session;
  $("selection-hint").textContent =
    tool === "box" ? "Kéo một khung quanh phần cần tách." : "Bấm vào phần cần tách. Bấm chuột phải để loại bớt vùng.";
}

document.querySelectorAll<HTMLButtonElement>(".tool").forEach((button) => {
  button.addEventListener("click", () => {
    if (session) setTool(button.dataset.tool as Tool);
  });
});

window.addEventListener("keydown", (event) => {
  const target = event.target as HTMLElement;
  if (target.matches("input[type=text], textarea") || !session) return;
  const tool = ({ v: "move", c: "point", b: "box" } as Record<string, Tool>)[event.key.toLowerCase()];
  if (tool && !event.ctrlKey && !event.metaKey) setTool(tool);
  if (event.key === "Escape") editor.clearPrompt();
  if (event.key === "Enter" && maskId && !target.matches("button")) selectionBar.requestSubmit();
  if ((event.key === "Delete" || event.key === "Backspace") && editor.selected) editor.remove(editor.selected);
});

// ---------------------------------------------------------------- selection

function syncExtract(): void {
  extractButton.disabled = !maskId || busy;
}

async function runSegment(): Promise<void> {
  if (!session) return;
  const { points, box } = editor.prompt;
  if (!points.length && !box) {
    maskId = null;
    levelField.hidden = true;
    syncExtract();
    return;
  }
  // Part / whole only means something for a single unambiguous click.
  levelField.hidden = !(points.length === 1 && !box);
  const seq = ++segmentSeq;
  try {
    const result = await api.segment(session.id, editor.prompt);
    if (seq !== segmentSeq) return;
    maskId = result.empty ? null : result.maskId;
    await editor.setOverlay(result.empty ? null : result.preview);
    syncExtract();
    say(result.empty ? "Không tìm thấy phần tử ở đó, hãy thử chỗ khác." : "Đã chọn. Bấm thêm để tinh chỉnh, hoặc tách thành layer.");
  } catch (error) {
    if (seq === segmentSeq) fail(error);
  }
}

editor.onPromptChanged = () => void runSegment();

levelField.addEventListener("change", () => {
  const value = new FormData(selectionBar).get("level");
  editor.prompt.level = value === "auto" ? null : (Number(value) as 0 | 1 | 2);
  void runSegment();
});

$("clear-selection").addEventListener("click", () => editor.clearPrompt());

selectionBar.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!session || !maskId) return;
  const id = session.id;
  const mask = maskId;
  void withBusy("Đang tách layer", async () => {
    const withShadow = $<HTMLInputElement>("with-shadow").checked;
    const result = await api.extract(id, mask, Number(edgeInput.value), transparentInput.checked, withShadow);
    maskId = null;
    for (const layer of result.layers) await editor.addLayer(layer);
    await editor.refreshBackground(result.backgroundVersion);
    editor.clearPrompt();
    const shadows = result.layers.some((layer) => layer.kind === "shadow");
    const name = result.layers[result.layers.length - 1].name;
    say(withShadow && !shadows ? `Đã tách ${name}, không thấy bóng đổ rõ ràng.` : `Đã tách ${name}.`);
  });
});

// -------------------------------------------------------------- auto layers

$("auto").addEventListener("click", () => {
  if (!session) return;
  if (!confirmReplace("Tự tách sẽ thay toàn bộ layer đang có. Tiếp tục?")) return;
  const id = session.id;
  const choice = document.querySelector<HTMLInputElement>('input[name="detail"]:checked')!.value;
  void withBusy("Đang phân tích ảnh", async () => {
    editor.clearPrompt();
    setProgress(0);
    const timer = window.setInterval(async () => {
      try {
        const state = await api.progress(id);
        if (state.stage) {
          say(state.stage);
          setProgress(state.value);
        }
      } catch {
        // A missed progress tick is harmless; the main request reports errors.
      }
    }, 400);
    try {
      const result = await api.decompose(id, {
        detail: choice,
        edge: Number(edgeInput.value),
        text: $<HTMLInputElement>("auto-text").checked,
        shadow: $<HTMLInputElement>("auto-shadow").checked,
      });
      await editor.replaceLayers(result.layers);
      await editor.refreshBackground(result.backgroundVersion);
      setTool("move");
      say(
        result.layers.length
          ? describeResult(result.layers)
          : "Không tìm thấy phần tử rõ ràng. Hãy thử mức Nhiều lớp, hoặc chọn thủ công.",
      );
      if (result.layers.length) playStackReveal();
    } finally {
      window.clearInterval(timer);
    }
  });
});

function describeResult(layers: { kind: string }[]): string {
  const count = (kind: string) => layers.filter((layer) => layer.kind === kind).length;
  const extras = [
    count("text") ? `${count("text")} layer chữ` : "",
    count("shadow") ? `${count("shadow")} layer bóng` : "",
  ].filter(Boolean);
  return `Đã tách thành ${layers.length} layer và một lớp nền${extras.length ? ` (gồm ${extras.join(" và ")})` : ""}.`;
}

/** A layer's picture as something drawable, whatever kind of layer it is. */
function layerPicture(layer: EditorLayer): { source: CanvasImageSource; width: number; height: number } {
  const { object } = layer;
  if (object instanceof IText) {
    const canvas = object.toCanvasElement();
    return { source: canvas, width: canvas.width, height: canvas.height };
  }
  return { source: object.getElement() as CanvasImageSource, width: object.width, height: object.height };
}

// --------------------------------------------------------------- layer list

function thumbnail(source: CanvasImageSource, width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.className = "thumb";
  canvas.width = 72;
  canvas.height = 54;
  const scale = Math.min(72 / width, 54 / height);
  canvas
    .getContext("2d")!
    .drawImage(source, (72 - width * scale) / 2, (54 - height * scale) / 2, width * scale, height * scale);
  return canvas;
}

const EYE_ON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="line" d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="line" d="M4 4l16 16M9.5 5.4A10 10 0 0 1 12 5c6.4 0 10 7 10 7a18 18 0 0 1-3 3.8M6.2 7.2A17 17 0 0 0 2 12s3.6 7 10 7a10 10 0 0 0 4-.8"/></svg>';

function row(name: string, visible: boolean, source: CanvasImageSource, width: number, height: number, onToggle: () => void): HTMLLIElement {
  const item = document.createElement("li");
  item.className = "layer";
  item.append(thumbnail(source, width, height));
  const label = document.createElement("span");
  label.className = "layer-name";
  label.textContent = name;
  const eye = document.createElement("button");
  eye.className = "icon";
  eye.innerHTML = visible ? EYE_ON : EYE_OFF;
  eye.setAttribute("aria-label", visible ? `Ẩn ${name}` : `Hiện ${name}`);
  eye.addEventListener("click", (event) => {
    event.stopPropagation();
    onToggle();
  });
  item.classList.toggle("off", !visible);
  item.append(label, eye);
  return item;
}

function renderLayers(): void {
  list.replaceChildren();
  const layers = [...editor.layers].reverse(); // the list reads top layer first
  for (const layer of layers) {
    const picture = layerPicture(layer);
    const item = row(layer.name, layer.visible, picture.source, picture.width, picture.height, () =>
      editor.setVisible(layer, !layer.visible),
    );
    item.dataset.id = layer.id;
    item.draggable = true;
    item.tabIndex = 0;
    item.addEventListener("click", () => editor.select(layer));
    item.addEventListener("keydown", (event) => keyboardReorder(event, layer));
    item.addEventListener("dragstart", (event) => {
      event.dataTransfer!.setData("text/layer", layer.id);
      event.dataTransfer!.effectAllowed = "move";
      item.classList.add("dragging");
    });
    item.addEventListener("dragend", () => item.classList.remove("dragging"));
    item.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const after = event.offsetY > item.clientHeight / 2;
      item.classList.toggle("drop-after", after);
      item.classList.toggle("drop-before", !after);
    });
    item.addEventListener("dragleave", () => item.classList.remove("drop-before", "drop-after"));
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const after = item.classList.contains("drop-after");
      item.classList.remove("drop-before", "drop-after");
      const dragged = editor.layers.find((candidate) => candidate.id === event.dataTransfer!.getData("text/layer"));
      if (!dragged || dragged === layer) return;
      const remaining = editor.layers.filter((candidate) => candidate !== dragged);
      // The list is reversed: dropping below a row means under it in the stack.
      const anchor = remaining.indexOf(layer);
      editor.move(dragged, after ? anchor : anchor + 1);
    });
    list.append(item);
  }
  if (editor.background && session) {
    list.append(
      row("Nền", editor.backgroundVisible, editor.background.getElement() as CanvasImageSource, session.width, session.height, () =>
        editor.setBackgroundVisible(!editor.backgroundVisible),
      ),
    );
  }
  empty.hidden = editor.layers.length > 0;
  props.hidden = editor.layers.length === 0;
  renderSelection();
}

function keyboardReorder(event: KeyboardEvent, layer: EditorLayer): void {
  const index = editor.layers.indexOf(layer);
  if (event.key === "Enter") editor.select(layer);
  if (event.key === "ArrowUp" && event.altKey && index < editor.layers.length - 1) editor.move(layer, index + 1);
  if (event.key === "ArrowDown" && event.altKey && index > 0) editor.move(layer, index - 1);
}

function renderSelection(): void {
  const selected = editor.selected;
  list.querySelectorAll<HTMLElement>(".layer").forEach((item) => {
    item.setAttribute("aria-current", String(!!selected && item.dataset.id === selected.id));
  });
  opacityInput.disabled = blendInput.disabled = !selected;
  textField.hidden = !selected?.text;
  if (selected?.text && document.activeElement !== textInput) textInput.value = editor.textOf(selected);
  if (selected) {
    opacityInput.value = String(Math.round(selected.object.opacity * 100));
    blendInput.value = selected.blendMode;
  }
}

editor.onLayersChanged = renderLayers;
editor.onSelectionChanged = renderSelection;
editor.onEditText = () => {
  textInput.focus();
  textInput.select();
};

textInput.addEventListener("input", () => {
  if (editor.selected?.text) editor.setText(editor.selected, textInput.value);
});

opacityInput.addEventListener("input", () => {
  if (editor.selected) editor.setOpacity(editor.selected, Number(opacityInput.value) / 100);
});
blendInput.addEventListener("change", () => {
  if (editor.selected) editor.setBlendMode(editor.selected, blendInput.value as GlobalCompositeOperation);
});

const compare = $<HTMLButtonElement>("compare");
compare.addEventListener("pointerdown", () => editor.showOriginal(true));
for (const type of ["pointerup", "pointerleave"]) compare.addEventListener(type, () => editor.showOriginal(false));

// ----------------------------------------------------------------- exports

function exporting(label: string, task: () => void | Promise<void>): void {
  void withBusy(`Đang xuất ${label}`, async () => {
    // Let the status repaint before the synchronous encode blocks the thread.
    await new Promise((resolve) => setTimeout(resolve, 30));
    await task();
    say(`Đã xuất ${label}.`);
  });
}

$("export-psd").addEventListener("click", () => exporting("PSD", () => exportPsd(editor)));
$("export-svg").addEventListener("click", () => exporting("SVG", () => exportSvg(editor)));
$("export-png").addEventListener("click", () => exporting("PNG", () => exportPng(editor)));
$("export-zip").addEventListener("click", () => exporting("từng layer", () => exportLayersZip(editor)));

// -------------------------------------------------------------- stack view

/** Lays every layer out as a sheet of film, so the separation can be seen. */
function buildStack(): void {
  if (!session) return;
  const sheetWidth = Math.min(pasteboard.clientWidth * 0.5, 620);
  const scale = sheetWidth / session.width;
  const gap = Math.min(70, 460 / (editor.layers.length + 1));
  const sheets: { node: HTMLElement; transform: string }[] = [
    { node: (editor.background!.getElement() as HTMLElement).cloneNode() as HTMLElement, transform: "" },
    ...editor.layers.map((layer) => {
      const { object } = layer;
      if (object instanceof IText) {
        const box = object.getBoundingRect();
        return { node: object.toCanvasElement() as HTMLElement, transform: `translate(${box.left}px,${box.top}px)` };
      }
      const [a, b, c, d, e, f] = object.calcTransformMatrix();
      // CSS origin is the element's corner, Fabric's matrix is about its centre.
      const matrix = [a, b, c, d, e - (a * object.width + c * object.height) / 2, f - (b * object.width + d * object.height) / 2];
      return { node: (object.getElement() as HTMLElement).cloneNode() as HTMLElement, transform: `matrix(${matrix.join(",")})` };
    }),
  ];
  stack.replaceChildren();
  sheets.forEach(({ node, transform }, index) => {
    const sheet = document.createElement("div");
    sheet.className = "sheet";
    sheet.style.width = `${sheetWidth}px`;
    sheet.style.height = `${session!.height * scale}px`;
    sheet.style.setProperty("--z", `${(index - (sheets.length - 1) / 2) * gap}px`);
    node.style.transform = `scale(${scale}) ${transform}`;
    sheet.append(node);
    stack.append(sheet);
  });
}

function showStack(on: boolean): void {
  if (on) buildStack();
  stack.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => stack.classList.toggle("open", on)));
  if (!on) {
    window.setTimeout(() => {
      if (!stack.classList.contains("open")) stack.hidden = true;
    }, 800);
  }
}

const stackView = $<HTMLButtonElement>("stack-view");
stackView.addEventListener("click", () => {
  const opening = !stack.classList.contains("open");
  showStack(opening);
  stackView.textContent = opening ? "Quay lại chỉnh sửa" : "Xem các lớp xếp chồng";
});

/** The one unprompted moment: the sheets part once to show what just happened. */
function playStackReveal(): void {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  showStack(true);
  window.setTimeout(() => {
    if (stackView.textContent === "Xem các lớp xếp chồng") showStack(false);
  }, 2200);
}

// ------------------------------------------------------------------- health

async function watchHealth(): Promise<void> {
  try {
    const health = await api.health();
    if (health.keyRequired && !getKey()) askForKey();
    if (health.error) {
      fail(`Không nạp được model AI: ${health.error}`);
      return;
    }
    if (!health.ready) {
      if (!session && !busy) say("Đang nạp model AI, lần đầu có thể mất vài phút");
      window.setTimeout(watchHealth, 2000);
    } else if (!session && !busy) {
      say(health.gpu ? `Sẵn sàng trên ${health.gpu}` : "Sẵn sàng (chạy bằng CPU, sẽ chậm)");
    }
  } catch {
    fail("Không kết nối được máy chủ AI. Hãy khởi động backend rồi tải lại trang.");
    window.setTimeout(watchHealth, 3000);
  }
}

void watchHealth();

// Exposed for automated checks only.
declare global {
  interface Window {
    __prolayer?: { editor: Editor; buildPsd: typeof buildPsd };
  }
}
if (import.meta.env.DEV) window.__prolayer = { editor, buildPsd };
