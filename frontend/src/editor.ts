import { Canvas, Circle, FabricImage, IText, Pattern, Point, Rect, type FabricObject, type TPointerEventInfo } from "fabric";
import { apiUrl, type LayerInfo, type Prompt, type SessionInfo, type TextInfo } from "./api";

export type Tool = "move" | "point" | "box";

export interface EditorLayer {
  id: string;
  name: string;
  object: FabricImage | IText;
  kind: LayerInfo["kind"];
  /** The recognised words, font and position. */
  text: TextInfo | null;
  /** The original pixels of a text layer, shown for as long as its words are unchanged. */
  raster: FabricImage | null;
  /** Where the image sat when it was extracted, to carry moves over to live text. */
  home: { left: number; top: number };
  visible: boolean;
  blendMode: GlobalCompositeOperation;
}

/** Distance from the top of a Fabric text box to its baseline, in font sizes (at lineHeight 1). */
export const TEXT_BASELINE = 0.93;

const PLACED = { originX: "left", originY: "top" } as const;
const HANDLES = {
  cornerColor: "#c9126b",
  cornerStrokeColor: "#fff",
  borderColor: "#c9126b",
  transparentCorners: false,
  cornerSize: 11,
} as const;
const INERT = { selectable: false, evented: false, ...PLACED } as const;
const MAGENTA = "#c9126b";
const CYAN = "#0a86b8";

function checkerboard(): HTMLCanvasElement {
  const cell = 12;
  const tile = document.createElement("canvas");
  tile.width = tile.height = cell * 2;
  const ctx = tile.getContext("2d")!;
  ctx.fillStyle = "#eef0f2";
  ctx.fillRect(0, 0, cell * 2, cell * 2);
  ctx.fillStyle = "#d4d8dd";
  ctx.fillRect(0, 0, cell, cell);
  ctx.fillRect(cell, cell, cell, cell);
  return tile;
}

export class Editor {
  readonly canvas: Canvas;
  session: SessionInfo | null = null;
  background: FabricImage | null = null;
  backgroundVisible = true;
  /** Bottom to top. */
  layers: EditorLayer[] = [];
  tool: Tool = "move";
  prompt: Prompt = { points: [], box: null, level: null };

  onLayersChanged: () => void = () => {};
  onSelectionChanged: () => void = () => {};
  onPromptChanged: () => void = () => {};
  onEditText: () => void = () => {};

  private paper: Rect | null = null;
  private original: FabricImage | null = null;
  private overlay: FabricImage | null = null;
  private markers: FabricObject[] = [];
  private dragBox: { rect: Rect; x: number; y: number } | null = null;
  private panning: { x: number; y: number } | null = null;
  private spaceHeld = false;

  constructor(element: HTMLCanvasElement, private host: HTMLElement) {
    this.canvas = new Canvas(element, {
      selection: false,
      preserveObjectStacking: true,
      fireRightClick: true,
      stopContextMenu: true,
      backgroundColor: "transparent",
    });
    this.canvas.on("mouse:down", (event) => this.pointerDown(event));
    this.canvas.on("mouse:move", (event) => this.pointerMove(event));
    this.canvas.on("mouse:up", () => this.pointerUp());
    this.canvas.on("mouse:dblclick", () => {
      if (this.selected?.text) this.onEditText();
    });
    this.canvas.on("mouse:wheel", (event) => this.wheel(event));
    this.canvas.on("after:render", () => this.drawCropMarks());
    this.canvas.on("selection:created", () => this.onSelectionChanged());
    this.canvas.on("selection:updated", () => this.onSelectionChanged());
    this.canvas.on("selection:cleared", () => this.onSelectionChanged());
    new ResizeObserver(() => this.resize()).observe(host);
    window.addEventListener("keydown", (event) => {
      if (event.code === "Space" && !(event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement)) {
        this.spaceHeld = true;
        this.canvas.defaultCursor = "grab";
      }
    });
    window.addEventListener("keyup", (event) => {
      if (event.code === "Space") {
        this.spaceHeld = false;
        this.applyTool();
      }
    });
  }

  get selected(): EditorLayer | null {
    const active = this.canvas.getActiveObject();
    return this.layers.find((layer) => layer.object === active) ?? null;
  }

  // ---------------------------------------------------------------- document

  async open(session: SessionInfo): Promise<void> {
    this.canvas.clear();
    this.canvas.backgroundColor = "transparent";
    this.session = session;
    this.layers = [];
    this.backgroundVisible = true;
    this.overlay = null;
    this.markers = [];
    this.prompt = { points: [], box: null, level: null };

    const { width, height } = session;
    this.paper = new Rect({
      ...INERT,
      left: 0,
      top: 0,
      width,
      height,
      strokeWidth: 0,
      fill: new Pattern({ source: checkerboard(), repeat: "repeat" }),
    });
    this.canvas.add(this.paper);
    this.original = await this.loadImage(apiUrl(`/api/sessions/${session.id}/image.png`));
    this.background = await this.loadImage(apiUrl(`/api/sessions/${session.id}/background.png?v=0`));
    this.canvas.add(this.background);
    this.fit();
    this.applyTool();
    this.onLayersChanged();
  }

  private async loadImage(url: string): Promise<FabricImage> {
    return FabricImage.fromURL(url, { crossOrigin: "anonymous" }, { ...INERT, left: 0, top: 0 });
  }

  async refreshBackground(version: number): Promise<void> {
    if (!this.session || !this.background) return;
    const next = await this.loadImage(apiUrl(`/api/sessions/${this.session.id}/background.png?v=${version}`));
    next.visible = this.backgroundVisible;
    const index = this.canvas.getObjects().indexOf(this.background);
    this.canvas.remove(this.background);
    this.canvas.insertAt(index, next);
    this.background = next;
    this.canvas.requestRenderAll();
    this.onLayersChanged();
  }

  async addLayer(info: LayerInfo): Promise<EditorLayer> {
    const blendMode = info.blendMode as GlobalCompositeOperation;
    const object = await FabricImage.fromURL(apiUrl(info.url), { crossOrigin: "anonymous" }, {
      ...PLACED,
      ...HANDLES,
      left: info.left,
      top: info.top,
      globalCompositeOperation: blendMode,
    });
    object.on("modified", () => this.onLayersChanged());
    const layer: EditorLayer = {
      id: info.id,
      name: info.name,
      object,
      kind: info.kind,
      text: info.text,
      raster: info.text ? object : null,
      home: { left: info.left, top: info.top },
      visible: true,
      blendMode,
    };
    this.layers.push(layer);
    this.canvas.add(object);
    this.restack();
    this.applyTool();
    return layer;
  }

  private makeText(text: TextInfo, content: string): IText {
    const radians = (text.angle * Math.PI) / 180;
    const rise = text.fontSize * TEXT_BASELINE;
    return new IText(content, {
      ...PLACED,
      ...HANDLES,
      left: text.x + Math.sin(radians) * rise,
      top: text.y - Math.cos(radians) * rise,
      angle: text.angle,
      scaleX: text.scaleX,
      fontFamily: text.fontFamily,
      fontWeight: text.fontWeight,
      fontSize: text.fontSize,
      fill: `rgb(${text.color.join(",")})`,
      lineHeight: 1,
      // Words are typed in the side panel: a real text field handles Vietnamese
      // input methods reliably, Fabric's in-canvas editor does not.
      editable: false,
    });
  }

  /** The words a text layer currently says. */
  textOf(layer: EditorLayer): string {
    return layer.object instanceof IText ? layer.object.text : (layer.text?.content ?? "");
  }

  /**
   * Original pixels stay on screen until the words really change, because no
   * matched font can look exactly like them. Typing the original back restores them.
   */
  setText(layer: EditorLayer, value: string): void {
    if (!layer.text || !layer.raster) return;
    const unchanged = value === layer.text.content;
    layer.name = value.trim() || layer.text.content;
    const current = layer.object;
    if (current instanceof IText) {
      if (unchanged) {
        const base = this.makeText(layer.text, value);
        layer.raster.set({ left: layer.home.left + current.left - base.left, top: layer.home.top + current.top - base.top });
        this.swapObject(layer, layer.raster);
      } else {
        current.set("text", value);
        this.canvas.requestRenderAll();
        this.onLayersChanged();
      }
    } else if (!unchanged) {
      const live = this.makeText(layer.text, value);
      live.set({
        left: live.left + current.left - layer.home.left,
        top: live.top + current.top - layer.home.top,
        opacity: current.opacity,
        visible: current.visible,
        globalCompositeOperation: layer.blendMode,
      });
      live.on("modified", () => this.onLayersChanged());
      this.swapObject(layer, live);
    }
  }

  private swapObject(layer: EditorLayer, next: FabricImage | IText): void {
    const active = this.canvas.getActiveObject() === layer.object;
    next.set({ opacity: layer.object.opacity, visible: layer.visible });
    this.canvas.remove(layer.object);
    layer.object = next;
    this.canvas.add(next);
    next.setCoords();
    this.restack();
    if (active) this.canvas.setActiveObject(next);
    this.canvas.requestRenderAll();
    this.onLayersChanged();
  }

  /** Keeps Fabric's paint order equal to `layers`, overlay and markers on top. */
  private restack(): void {
    let index = 2; // paper, background
    for (const layer of this.layers) this.canvas.moveObjectTo(layer.object, index++);
    if (this.overlay) this.canvas.moveObjectTo(this.overlay, index++);
    for (const marker of this.markers) this.canvas.moveObjectTo(marker, index++);
    this.canvas.requestRenderAll();
  }

  async replaceLayers(infos: LayerInfo[]): Promise<void> {
    this.canvas.discardActiveObject();
    for (const layer of this.layers) this.canvas.remove(layer.object);
    this.layers = [];
    for (const info of infos) await this.addLayer(info);
    this.onLayersChanged();
  }

  // ------------------------------------------------------------ layer edits

  select(layer: EditorLayer): void {
    if (this.tool !== "move") this.setTool("move");
    this.canvas.setActiveObject(layer.object);
    this.canvas.requestRenderAll();
    this.onSelectionChanged();
  }

  setVisible(layer: EditorLayer, visible: boolean): void {
    layer.visible = visible;
    layer.object.visible = visible;
    if (!visible && this.canvas.getActiveObject() === layer.object) this.canvas.discardActiveObject();
    this.canvas.requestRenderAll();
    this.onLayersChanged();
  }

  setBackgroundVisible(visible: boolean): void {
    this.backgroundVisible = visible;
    if (this.background) this.background.visible = visible;
    this.canvas.requestRenderAll();
    this.onLayersChanged();
  }

  setOpacity(layer: EditorLayer, opacity: number): void {
    layer.object.set("opacity", opacity);
    this.canvas.requestRenderAll();
  }

  setBlendMode(layer: EditorLayer, mode: GlobalCompositeOperation): void {
    layer.blendMode = mode;
    layer.object.set("globalCompositeOperation", mode);
    this.canvas.requestRenderAll();
  }

  remove(layer: EditorLayer): void {
    this.layers = this.layers.filter((item) => item !== layer);
    this.canvas.remove(layer.object);
    this.canvas.discardActiveObject();
    this.canvas.requestRenderAll();
    this.onLayersChanged();
  }

  /** `to` is an index into `layers` after the moved layer has been taken out. */
  move(layer: EditorLayer, to: number): void {
    this.layers = this.layers.filter((item) => item !== layer);
    this.layers.splice(to, 0, layer);
    this.restack();
    this.onLayersChanged();
  }

  showOriginal(on: boolean): void {
    if (!this.original) return;
    if (on) {
      this.canvas.add(this.original);
    } else {
      this.canvas.remove(this.original);
    }
    this.canvas.requestRenderAll();
  }

  // ---------------------------------------------------------------- tools

  setTool(tool: Tool): void {
    this.tool = tool;
    if (tool !== "move") {
      this.canvas.discardActiveObject();
      this.onSelectionChanged();
    }
    this.applyTool();
  }

  private applyTool(): void {
    const moving = this.tool === "move";
    this.canvas.skipTargetFind = !moving;
    this.canvas.defaultCursor = moving ? "default" : "crosshair";
    this.canvas.hoverCursor = moving ? "move" : "crosshair";
    this.canvas.requestRenderAll();
  }

  clearPrompt(): void {
    this.prompt = { points: [], box: null, level: null };
    for (const marker of this.markers) this.canvas.remove(marker);
    this.markers = [];
    this.setOverlay(null);
    this.onPromptChanged();
  }

  async setOverlay(dataUrl: string | null): Promise<void> {
    const previous = this.overlay;
    this.overlay = null;
    if (dataUrl && this.session) {
      const next = await FabricImage.fromURL(dataUrl, {}, { ...INERT, left: 0, top: 0 });
      next.scaleX = this.session.width / next.width;
      next.scaleY = this.session.height / next.height;
      this.overlay = next;
      this.canvas.add(next);
      this.restack();
    }
    if (previous) this.canvas.remove(previous);
    this.canvas.requestRenderAll();
  }

  private inDocument(x: number, y: number): boolean {
    return !!this.session && x >= 0 && y >= 0 && x < this.session.width && y < this.session.height;
  }

  private pointerDown(event: TPointerEventInfo): void {
    const mouse = event.e as MouseEvent;
    if (!this.session) return;
    if (this.spaceHeld || mouse.button === 1) {
      this.panning = { x: mouse.clientX, y: mouse.clientY };
      return;
    }
    const { x, y } = this.canvas.getScenePoint(mouse);
    if (this.tool === "point" && this.inDocument(x, y)) {
      this.addPoint(x, y, mouse.button === 0 && !mouse.altKey);
    } else if (this.tool === "box" && mouse.button === 0) {
      const zoom = this.canvas.getZoom();
      const rect = new Rect({
        ...INERT,
        left: x,
        top: y,
        width: 0,
        height: 0,
        fill: "rgba(10,134,184,.12)",
        stroke: CYAN,
        strokeWidth: 1.5 / zoom,
      });
      this.dragBox = { rect, x, y };
      this.canvas.add(rect);
    }
  }

  private addPoint(x: number, y: number, positive: boolean): void {
    const zoom = this.canvas.getZoom();
    const marker = new Circle({
      ...INERT,
      originX: "center",
      originY: "center",
      left: x,
      top: y,
      radius: 5 / zoom,
      fill: positive ? MAGENTA : "#fff",
      stroke: positive ? "#fff" : MAGENTA,
      strokeWidth: 1.5 / zoom,
    });
    this.markers.push(marker);
    this.canvas.add(marker);
    this.prompt.points.push({ x, y, positive });
    this.onPromptChanged();
  }

  private pointerMove(event: TPointerEventInfo): void {
    const mouse = event.e as MouseEvent;
    if (this.panning) {
      this.canvas.relativePan(new Point(mouse.clientX - this.panning.x, mouse.clientY - this.panning.y));
      this.panning = { x: mouse.clientX, y: mouse.clientY };
      return;
    }
    if (!this.dragBox) return;
    const { x, y } = this.canvas.getScenePoint(mouse);
    const { rect, x: startX, y: startY } = this.dragBox;
    rect.set({ left: Math.min(x, startX), top: Math.min(y, startY), width: Math.abs(x - startX), height: Math.abs(y - startY) });
    this.canvas.requestRenderAll();
  }

  private pointerUp(): void {
    this.panning = null;
    if (!this.dragBox || !this.session) return;
    const { rect } = this.dragBox;
    this.dragBox = null;
    if (rect.width * this.canvas.getZoom() < 6 || rect.height * this.canvas.getZoom() < 6) {
      this.canvas.remove(rect);
      return;
    }
    for (const marker of this.markers) this.canvas.remove(marker);
    this.markers = [rect];
    const { width, height } = this.session;
    this.prompt = {
      points: [],
      level: null,
      box: [
        Math.max(0, rect.left),
        Math.max(0, rect.top),
        Math.min(width, rect.left + rect.width),
        Math.min(height, rect.top + rect.height),
      ],
    };
    this.onPromptChanged();
  }

  // --------------------------------------------------------------- viewport

  private wheel(event: TPointerEventInfo<WheelEvent>): void {
    event.e.preventDefault();
    const zoom = Math.min(16, Math.max(0.05, this.canvas.getZoom() * 0.999 ** event.e.deltaY));
    this.canvas.zoomToPoint(new Point(event.e.offsetX, event.e.offsetY), zoom);
  }

  private resize(): void {
    this.canvas.setDimensions({ width: this.host.clientWidth, height: this.host.clientHeight });
    if (this.session) this.fit();
  }

  fit(): void {
    if (!this.session) return;
    const { width, height } = this.session;
    const side = 48;
    const top = 36;
    const bottom = 124; // clears the selection bar, which may wrap to two rows
    const available = { w: this.host.clientWidth - side * 2, h: this.host.clientHeight - top - bottom };
    const zoom = Math.min(available.w / width, available.h / height, 4);
    this.canvas.setViewportTransform([
      zoom,
      0,
      0,
      zoom,
      (this.host.clientWidth - width * zoom) / 2,
      top + (available.h - height * zoom) / 2,
    ]);
  }

  /** Printer's crop marks: the document edge stays legible when nothing fills it. */
  private drawCropMarks(): void {
    if (!this.session) return;
    const ctx = this.canvas.getContext();
    const [zoom, , , , tx, ty] = this.canvas.viewportTransform;
    const x0 = tx;
    const y0 = ty;
    const x1 = tx + this.session.width * zoom;
    const y1 = ty + this.session.height * zoom;
    const gap = 6;
    const length = 16;
    ctx.save();
    ctx.setTransform(this.canvas.getRetinaScaling(), 0, 0, this.canvas.getRetinaScaling(), 0, 0);
    ctx.strokeStyle = "#16243a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const x of [x0, x1]) {
      for (const [from, sign] of [[y0, -1], [y1, 1]] as const) {
        ctx.moveTo(x + 0.5, from + sign * gap);
        ctx.lineTo(x + 0.5, from + sign * (gap + length));
      }
    }
    for (const y of [y0, y1]) {
      for (const [from, sign] of [[x0, -1], [x1, 1]] as const) {
        ctx.moveTo(from + sign * gap, y + 0.5);
        ctx.lineTo(from + sign * (gap + length), y + 0.5);
      }
    }
    ctx.stroke();
    ctx.restore();
  }
}
