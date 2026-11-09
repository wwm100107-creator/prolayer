"""Turns flat-coloured layers into SVG paths. Photographic ones stay raster."""
from __future__ import annotations

import re

import cv2
import numpy as np
import vtracer

from .pipeline import Layer

FLAT_TOP_COLOURS = 12
FLAT_COVERAGE = 0.88
MAX_SVG_CHARS = 4_000_000
_INNER = re.compile(r"<svg[^>]*>(.*)</svg>", re.DOTALL)


def is_flat(rgba: np.ndarray) -> bool:
    """True for logos, icons and shapes: a few colours cover almost everything."""
    opaque = rgba[rgba[..., 3] >= 128][:, :3]
    if not len(opaque):
        return False
    quantized = opaque.astype(np.uint32) >> 3
    codes = (quantized[:, 0] << 10) | (quantized[:, 1] << 5) | quantized[:, 2]
    counts = np.sort(np.bincount(codes))[::-1]
    return float(counts[:FLAT_TOP_COLOURS].sum()) / len(opaque) >= FLAT_COVERAGE


def layer_svg(layer: Layer) -> str | None:
    """The layer's SVG paths at its own pixel size, or None to keep it raster."""
    if layer.kind != "raster" or not is_flat(layer.rgba):
        return None
    # VTracer has no notion of partial alpha: edges become either in or out.
    pixels = cv2.cvtColor(layer.rgba, cv2.COLOR_RGBA2BGRA)
    pixels[..., 3] = np.where(pixels[..., 3] >= 128, 255, 0)
    pixels[pixels[..., 3] == 0] = 0
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        return None
    svg = vtracer.convert_raw_image_to_svg(
        encoded.tobytes(),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=4,
        color_precision=6,
        layer_difference=16,
    )
    inner = _INNER.search(svg)
    if inner is None or len(inner.group(1)) > MAX_SVG_CHARS:
        return None
    return inner.group(1).strip()
