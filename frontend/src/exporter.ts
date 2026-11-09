import { writePsd, type BlendMode, type Layer as PsdLayer } from "ag-psd";
import { FabricImage, IText } from "fabric";
import { zipSync } from "fflate";
import { api, type TextInfo } from "./api";
import { TEXT_BASELINE, type Editor, type EditorLayer } from "./editor";

interface Baked {
  canvas: HTMLCanvasElement;
  left: number;
  top: number;
}

function makeCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** Renders a layer with its move/scale/rotate applied, cropped to what shows. */
function bake(layer: EditorLayer, docWidth: number, docHeight: number): Baked | null {
  const { object } = layer;
  const corners = object.getCoords();
  // Glyphs can overhang their text box by a little.
  const overhang = object instanceof IText ? Math.ceil(object.fontSize * 0.25) : 0;
  const left = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.x))) - overhang);
  const top = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.y))) - overhang);
  const right = Math.min(docWidth, Math.ceil(Math.max(...corners.map((c) => c.x))) + overhang);
  const bottom = Math.min(docHeight, Math.ceil(Math.max(...corners.map((c) => c.y))) + overhang);
  if (right <= left || bottom <= top) return null;

  const canvas = makeCanvas(right - left, bottom - top);
  const ctx = canvas.getContext("2d")!;
  ctx.translate(-left, -top);
  if (object instanceof FabricImage) {
    const [a, b, c, d, e, f] = object.calcTransformMatrix();
    ctx.transform(a, b, c, d, e, f);
    ctx.drawImage(object.getElement() as CanvasImageSource, -object.width / 2, -object.height / 2);
  } else {
    object.render(ctx);
  }
  return { canvas, left, top };
}

function composite(editor: Editor): HTMLCanvasElement {
  const { width, height } = editor.session!;
  const out = makeCanvas(width, height);
  const ctx = out.getContext("2d")!;
  if (editor.backgroundVisible && editor.background) {
    ctx.drawImage(editor.background.getElement() as CanvasImageSource, 0, 0);
  }
  for (const layer of editor.layers) {
    if (!layer.visible) continue;
    const baked = bake(layer, width, height);
    if (!baked) continue;
    ctx.globalAlpha = layer.object.opacity;
    ctx.globalCompositeOperation = layer.blendMode;
    ctx.drawImage(baked.canvas, baked.left, baked.top);
  }
  return out;
}

function backgroundCanvas(editor: Editor): HTMLCanvasElement {
  const { width, height } = editor.session!;
  const canvas = makeCanvas(width, height);
  canvas.getContext("2d")!.drawImage(editor.background!.getElement() as CanvasImageSource, 0, 0);
  return canvas;
}

/** The layer's text with its transform, measured from the baseline start. */
function textPlacement(object: IText): { matrix: number[]; content: string } {
  const [a, b, c, d, e, f] = object.calcTransformMatrix();
  const x = -object.width / 2;
  const y = -object.height / 2 + object.fontSize * TEXT_BASELINE;
  return { matrix: [a, b, c, d, a * x + c * y + e, b * x + d * y + f], content: object.text };
}

/** The same for a text layer still showing its original pixels. */
function rasterTextPlacement(layer: EditorLayer, text: TextInfo): number[] {
  const radians = (text.angle * Math.PI) / 180;
  const [cos, sin] = [Math.cos(radians), Math.sin(radians)];
  const x = text.x + layer.object.left - layer.home.left;
  const y = text.y + layer.object.top - layer.home.top;
  return [cos * text.scaleX, sin * text.scaleX, -sin, cos, x, y];
}

function save(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

function pngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Không tạo được PNG."))), "image/png"),
  );
}

/** Builds the PSD. ag-psd lists children bottom-first and takes opacity in 0-1. */
export function buildPsd(editor: Editor): ArrayBuffer {
  const { width, height } = editor.session!;
  const children: PsdLayer[] = [
    { name: "Nền", canvas: backgroundCanvas(editor), left: 0, top: 0, hidden: !editor.backgroundVisible },
  ];
  for (const layer of editor.layers) {
    const baked = bake(layer, width, height);
    if (!baked) continue;
    const psdLayer: PsdLayer = {
      name: layer.name,
      canvas: baked.canvas,
      left: baked.left,
      top: baked.top,
      hidden: !layer.visible,
      opacity: layer.object.opacity,
      blendMode: layer.blendMode as BlendMode,
    };
    if (layer.text) {
      const { matrix, content } =
        layer.object instanceof IText
          ? textPlacement(layer.object)
          : { matrix: rasterTextPlacement(layer, layer.text), content: layer.text.content };
      const [r, g, b] = layer.text.color;
      psdLayer.text = {
        text: content,
        transform: matrix,
        style: { font: { name: layer.text.postScriptName }, fontSize: layer.text.fontSize, fillColor: { r, g, b } },
      };
    }
    children.push(psdLayer);
  }
  // invalidateTextLayers makes Photoshop redraw the text with the real font.
  return writePsd({ width, height, canvas: composite(editor), children }, { generateThumbnail: true, invalidateTextLayers: true });
}

export function exportPsd(editor: Editor): void {
  save(new Blob([buildPsd(editor)], { type: "image/vnd.adobe.photoshop" }), "prolayer.psd");
}

export async function exportPng(editor: Editor): Promise<void> {
  save(await pngBlob(composite(editor)), "prolayer.png");
}

export async function exportLayersZip(editor: Editor): Promise<void> {
  const { width, height } = editor.session!;
  const files: Record<string, Uint8Array> = {};
  const add = async (name: string, canvas: HTMLCanvasElement) => {
    files[name] = new Uint8Array(await (await pngBlob(canvas)).arrayBuffer());
  };
  await add("00-nen.png", backgroundCanvas(editor));
  let index = 1;
  for (const layer of editor.layers) {
    const baked = bake(layer, width, height);
    if (!baked) continue;
    const full = makeCanvas(width, height);
    full.getContext("2d")!.drawImage(baked.canvas, baked.left, baked.top);
    await add(`${String(index++).padStart(2, "0")}-layer.png`, full);
  }
  // PNGs are already compressed: level 0 just stores them.
  const zipped = zipSync(files, { level: 0 });
  save(new Blob([zipped.slice().buffer as ArrayBuffer], { type: "application/zip" }), "prolayer-layers.zip");
}

const escapeXml = (value: string) =>
  value.replace(/[<>&"']/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;" })[c]!);

function elementDataUrl(element: CanvasImageSource, width: number, height: number, type: string): string {
  const canvas = makeCanvas(width, height);
  canvas.getContext("2d")!.drawImage(element, 0, 0);
  return canvas.toDataURL(type, 0.92);
}

export async function exportSvg(editor: Editor): Promise<void> {
  const session = editor.session!;
  const { width, height } = session;
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`];
  if (editor.backgroundVisible && editor.background) {
    const href = elementDataUrl(editor.background.getElement() as CanvasImageSource, width, height, "image/jpeg");
    parts.push(`<image data-name="Nền" width="${width}" height="${height}" href="${href}"/>`);
  }
  for (const layer of editor.layers) {
    if (!layer.visible) continue;
    const { object } = layer;
    const style = layer.blendMode === "source-over" || (layer.blendMode as string) === "normal" ? "" : ` style="mix-blend-mode:${layer.blendMode}"`;
    const common = `data-name="${escapeXml(layer.name)}" opacity="${object.opacity}"${style}`;
    if (object instanceof IText && layer.text) {
      const { matrix, content } = textPlacement(object);
      parts.push(
        `<text ${common} transform="matrix(${matrix.join(" ")})" font-family="${escapeXml(layer.text.fontFamily)}" ` +
          `font-size="${object.fontSize}" font-weight="${layer.text.fontWeight}" fill="rgb(${layer.text.color.join(",")})" ` +
          `xml:space="preserve">${escapeXml(content)}</text>`,
      );
    } else if (object instanceof FabricImage) {
      const [a, b, c, d, e, f] = object.calcTransformMatrix();
      const dx = e - (a * object.width + c * object.height) / 2;
      const dy = f - (b * object.width + d * object.height) / 2;
      const transform = `transform="matrix(${[a, b, c, d, dx, dy].join(" ")})"`;
      const { svg } = await api.layerSvg(session.id, layer.id);
      if (svg) {
        parts.push(`<g ${common} ${transform}>${svg}</g>`);
      } else {
        const href = elementDataUrl(object.getElement() as CanvasImageSource, object.width, object.height, "image/png");
        parts.push(`<image ${common} ${transform} width="${object.width}" height="${object.height}" href="${href}"/>`);
      }
    }
  }
  parts.push("</svg>");
  save(new Blob([parts.join("\n")], { type: "image/svg+xml" }), "prolayer.svg");
}
