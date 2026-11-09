"""Finds text, rebuilds each line as an editable layer, erases it from the picture."""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import engine
from .pipeline import Layer, _bbox, _dilate, _pad_box, _trim, inpaint_region

log = logging.getLogger("prolayer")

MIN_CONFIDENCE = 0.4
MIN_INK_OVERLAP = 0.55
MIN_LINE_HEIGHT = 10
MIN_CONTRAST = 40.0
MAX_TEXT_LAYERS = 60
FONT_DIR = Path(os.environ.get("PROLAYER_FONT_DIR", r"C:\Windows\Fonts"))
FIT_SIZE = 64


class Font(NamedTuple):
    file: str
    family: str
    postscript: str
    weight: int


FONTS = [
    Font("arial.ttf", "Arial", "ArialMT", 400),
    Font("arialbd.ttf", "Arial", "Arial-BoldMT", 700),
    Font("times.ttf", "Times New Roman", "TimesNewRomanPSMT", 400),
    Font("timesbd.ttf", "Times New Roman", "TimesNewRomanPS-BoldMT", 700),
    Font("georgia.ttf", "Georgia", "Georgia", 400),
    Font("georgiab.ttf", "Georgia", "Georgia-Bold", 700),
    Font("verdana.ttf", "Verdana", "Verdana", 400),
    Font("verdanab.ttf", "Verdana", "Verdana-Bold", 700),
    Font("tahoma.ttf", "Tahoma", "Tahoma", 400),
    Font("tahomabd.ttf", "Tahoma", "Tahoma-Bold", 700),
    Font("segoeui.ttf", "Segoe UI", "SegoeUI", 400),
    Font("segoeuib.ttf", "Segoe UI", "SegoeUI-Bold", 700),
    Font("calibri.ttf", "Calibri", "Calibri", 400),
    Font("calibrib.ttf", "Calibri", "Calibri-Bold", 700),
    Font("impact.ttf", "Impact", "Impact", 400),
]


def _render_ink(font: ImageFont.FreeTypeFont, text: str) -> tuple[np.ndarray, int, int] | None:
    """Ink of `text` as drawn, plus where it sits relative to the baseline start.

    Measured from pixels because getbbox reports the origin, not the first
    glyph's ink, which would shift every line by its left bearing.
    """
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    pad = 4
    origin = (pad - left, pad - top)
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(canvas).text(origin, text, font=font, fill=255, anchor="ls")
    ink = np.array(canvas) > 127
    box = _bbox(ink)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return ink[y0:y1, x0:x1], x0 - origin[0], y1 - origin[1]


def match_font(ink: np.ndarray, text: str) -> tuple[Font, float, float, float, float]:
    """Best installed font for this line of ink.

    Returns (font, size, x, y, overlap) with x/y the baseline start in `ink`
    coordinates. Each candidate is sized to the width of the ink, then scored.
    """
    x0, y0, x1, y1 = _bbox(ink)
    width, height = x1 - x0, y1 - y0
    target = ink[y0:y1, x0:x1]
    best: tuple[float, Font, float, float, float, float] | None = None
    for candidate in FONTS:
        path = FONT_DIR / candidate.file
        if not path.exists():
            continue
        fitted = _render_ink(ImageFont.truetype(str(path), FIT_SIZE), text)
        if fitted is None:
            continue
        size = FIT_SIZE * width / fitted[0].shape[1]
        rendered = _render_ink(ImageFont.truetype(str(path), max(1, round(size))), text)
        if rendered is None:
            continue
        glyphs, bearing, descent = rendered
        resized = cv2.resize(glyphs.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        overlap = (resized & target).sum() / max((resized | target).sum(), 1)
        # Overlap alone favours wrong fonts stretched into the right box.
        score = overlap - 0.5 * abs(math.log(glyphs.shape[0] / height))
        if best is None or score > best[0]:
            best = (score, candidate, size, x0 - bearing, y1 - descent, float(overlap))
    if best is None:  # no known font on this machine: estimate from the ink
        return FONTS[0], height / 0.72, float(x0), float(y1), 1.0
    return best[1], best[2], best[3], best[4], best[5]


def _separate(crop: np.ndarray, rim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Splits a text crop into (text colour, background colour, alpha)."""
    samples = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(samples, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(crop.shape[:2])
    border = np.ones(labels.shape, bool)
    border[rim:-rim, rim:-rim] = False
    background_label = int(np.bincount(labels[border], minlength=2).argmax())
    is_text = labels != background_label
    share = float(is_text.mean())
    if not 0.03 <= share <= 0.75:
        return None
    foreground = np.median(crop[is_text], axis=0).astype(np.float32)
    background = np.median(crop[~is_text], axis=0).astype(np.float32)
    direction = foreground - background
    if float(np.linalg.norm(direction)) < MIN_CONTRAST:
        return None
    # How far along the background -> text colour line each pixel sits.
    progress = ((crop.astype(np.float32) - background) * direction).sum(axis=2) / float((direction**2).sum())
    return foreground, background, np.clip((progress - 0.15) / 0.7, 0, 1).astype(np.float32)


def extract_text(rgb: np.ndarray) -> tuple[list[Layer], np.ndarray]:
    """Returns (text layers, the picture with that text erased)."""
    height, width = rgb.shape[:2]
    layers: list[Layer] = []
    hole = np.zeros((height, width), bool)

    for corners, content, confidence in engine.read_text(rgb):
        content = content.strip()
        if confidence < MIN_CONFIDENCE or len(content) < 2 or len(layers) >= MAX_TEXT_LAYERS:
            continue
        quad = np.array(corners, np.float32)
        line_width = round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        line_height = round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        if line_height < MIN_LINE_HEIGHT or line_width < 2:
            continue

        # Straighten the line, keeping a rim of background around it.
        pad = max(3, round(line_height * 0.15))
        target = np.array(
            [[pad, pad], [pad + line_width, pad], [pad + line_width, pad + line_height], [pad, pad + line_height]],
            np.float32,
        )
        forward = cv2.getPerspectiveTransform(quad, target)
        size = (line_width + 2 * pad, line_height + 2 * pad)
        crop = cv2.warpPerspective(rgb, forward, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        separated = _separate(crop, pad)
        if separated is None:
            continue
        foreground, _, alpha = separated
        if not (alpha > 0.5).any():
            continue

        font, font_size, baseline_x, baseline_y, overlap = match_font(alpha > 0.5, content)
        # Misread words and display fonts render nothing like the ink. Replacing
        # them would corrupt the picture, so they stay graphics for the segmenter.
        if overlap < MIN_INK_OVERLAP:
            log.info("Keeping %r as a graphic: rendered ink overlap %.2f", content, overlap)
            continue
        inverse = np.linalg.inv(forward)
        origin = cv2.perspectiveTransform(np.array([[[baseline_x, baseline_y]]], np.float32), inverse)[0, 0]
        angle = math.degrees(math.atan2(float(quad[1][1] - quad[0][1]), float(quad[1][0] - quad[0][0])))

        # Back into picture space, only over this line's bounding box.
        x_min, y_min = np.floor(quad.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(quad.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = _pad_box((int(x_min), int(y_min), int(x_max), int(y_max)), pad, width, height)
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        placed = cv2.warpPerspective(alpha, shift @ inverse, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)

        rgba = np.zeros((*placed.shape, 4), np.uint8)
        rgba[..., :3] = np.clip(foreground, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(placed * 255 + 0.5, 0, 255).astype(np.uint8)
        layer = _trim(rgba, x0, y0, content)
        layer.kind = "text"
        layer.text = {
            "content": content,
            "fontFamily": font.family,
            "postScriptName": font.postscript,
            "fontWeight": font.weight,
            "fontSize": round(font_size, 2),
            "color": [int(c) for c in np.clip(foreground, 0, 255)],
            "angle": round(angle, 2),
            "x": round(float(origin[0]), 2),
            "y": round(float(origin[1]), 2),
        }
        layers.append(layer)

        grow = max(2, round(line_height * 0.1))
        region = hole[y0:y1, x0:x1]
        region |= _dilate(placed > 0.15, grow)

    if not layers:
        return [], rgb
    return layers, inpaint_region(rgb, hole)
