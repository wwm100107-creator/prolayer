"""End-to-end check on real weights: python -m tests.pipeline_check OUT_DIR

Uses a synthetic poster (known element positions) and a real photo, and writes
each decomposition as a contact sheet so the result can be looked at.
Needs a CUDA GPU and `pip install scikit-image` for the sample photos.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import data

from app import pipeline
from app.models import engine


def poster() -> np.ndarray:
    """Gradient background, soft-edged sun, two overlapping shapes, a banner."""
    height, width = 768, 1024
    ramp = np.linspace(0, 1, height)[:, None, None]
    image = (np.array([250, 225, 190]) * (1 - ramp) + np.array([120, 170, 220]) * ramp).astype(np.float32)
    image = np.repeat(image, width, axis=1)

    yy, xx = np.mgrid[:height, :width]
    sun = np.clip(1 - (np.hypot(xx - 780, yy - 190) - 95) / 25, 0, 1)[..., None]
    image = image * (1 - sun) + np.array([250, 180, 40]) * sun
    image = image.astype(np.uint8).copy()
    cv2.rectangle(image, (120, 300), (480, 660), (40, 110, 90), -1, cv2.LINE_AA)
    cv2.circle(image, (470, 470), 150, (200, 60, 70), -1, cv2.LINE_AA)
    cv2.rectangle(image, (180, 400), (330, 470), (245, 240, 225), -1, cv2.LINE_AA)
    cv2.ellipse(image, (800, 600), (140, 70), 20, 0, 360, (70, 50, 140), -1, cv2.LINE_AA)
    return image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(255**2 / max(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2), 1e-6)))


def checker(height: int, width: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    value = 205 + 40 * (((xx // cell) + (yy // cell)) % 2)
    return np.repeat(value[..., None].astype(np.uint8), 3, axis=2)


def contact_sheet(rgb, background, layers, thumb=256, columns=5) -> np.ndarray:
    height, width = rgb.shape[:2]
    tiles = [rgb, background]
    for layer in layers:
        tile = checker(height, width)
        pipeline._composite(tile, layer)
        tiles.append(tile)
    size = (thumb, round(thumb * height / width))
    small = [cv2.resize(t, size, interpolation=cv2.INTER_AREA) for t in tiles]
    while len(small) % columns:
        small.append(np.full_like(small[0], 40))
    rows = [np.hstack(small[i : i + columns]) for i in range(0, len(small), columns)]
    return np.vstack(rows)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {"poster": poster(), "astronaut": data.astronaut(), "coffee": data.coffee()}
    for name, rgb in images.items():
        for detail in ("coarse", "medium"):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            embeddings = engine.embed(rgb)
            embedded = time.perf_counter()
            layers, background = pipeline.decompose(rgb, embeddings, detail=detail)
            done = time.perf_counter()
            flat = pipeline.flatten(background, layers)
            print(
                f"{name:10s} {detail:7s} layers={len(layers):2d} embed={embedded - start:.2f}s "
                f"decompose={done - embedded:.2f}s psnr={psnr(flat, rgb):.2f} "
                f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB",
                flush=True,
            )
            sheet = contact_sheet(rgb, background, layers)
            cv2.imwrite(str(out_dir / f"{name}_{detail}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # Click mode: centre of the red circle on the poster.
    rgb = images["poster"]
    embeddings = engine.embed(rgb)
    start = time.perf_counter()
    mask, score = engine.predict_mask(rgb, embeddings, [(520.0, 470.0)], [1], None)
    decoded = time.perf_counter()
    layer, background = pipeline.extract_element(rgb, rgb.copy(), mask, "circle")
    print(
        f"click: score={score:.3f} area={mask.sum()} decode={decoded - start:.3f}s "
        f"extract={time.perf_counter() - decoded:.2f}s box=({layer.left},{layer.top},{layer.rgba.shape[1]},{layer.rgba.shape[0]}) "
        f"psnr={psnr(pipeline.flatten(background, [layer]), rgb):.2f}",
        flush=True,
    )
    # Red must not leak into the repaired background.
    disc = np.hypot(*np.mgrid[:rgb.shape[0], :rgb.shape[1]][::-1] - np.array([470, 470])[:, None, None]) < 130
    print("background mean inside former circle (RGB):", background[disc].mean(0).round(1).tolist())
    sheet = contact_sheet(rgb, background, [layer], columns=3)
    cv2.imwrite(str(out_dir / "click.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
