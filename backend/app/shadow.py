"""Finds cast shadows around objects and lifts them into multiply layers.

There is no trained detector here. A shadow is modelled as what it physically
is: pixels next to an object that are darker than the lit surface around them
but keep that surface's hue. That is reliable on plain and softly textured
surfaces, and can miss or misread shadows on busy ones.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .pipeline import Layer, _bbox, _dilate, _smooth_fill

WORK_SIDE = 1024
MIN_DARKNESS = 0.08
MIN_MEAN_DARKNESS = 0.12
MAX_CHROMA_SHIFT = 16.0
MAX_ALPHA = 0.85
# A dark area covering this much of the search limit is scenery, not a shadow.
MAX_OUTER_CONTACT = 0.3
ALPHA_EPS = 0.02
SOFT_EDGE_RATIO = 0.2
HARD_EDGE_PERCENTILE = 80
MIN_SAME_HUE = 0.9
MAX_SURFACE_DETAIL = 6.0


def _gradient(lightness: np.ndarray) -> np.ndarray:
    """Brightness change per pixel."""
    grad_x = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3) / 8
    grad_y = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3) / 8
    return np.hypot(grad_x, grad_y)


def _masked_blur(image: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores invalid pixels, so objects cannot bleed in."""
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    blurred = cv2.GaussianBlur(image.astype(np.float32) * valid[..., None], (0, 0), sigma)
    return blurred / np.maximum(weight[..., None], 1e-3)


def estimate(rgb: np.ndarray, objects: np.ndarray) -> np.ndarray:
    """Shadow opacity in [0, 1] at full resolution; zero on the objects."""
    height, width = rgb.shape[:2]
    none = np.zeros((height, width), np.float32)
    scale = min(1.0, WORK_SIDE / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    body = cv2.resize(objects.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.3
    if not body.any():
        return none

    # Sized by the largest single object: the union of many would search everywhere.
    count, _, stats, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), connectivity=8)
    reach = int(np.clip(math.sqrt(stats[1:, cv2.CC_STAT_AREA].max()) * 0.3, 8, 240))
    near = _dilate(body, reach)
    far = ~near
    if far.mean() < 0.05:  # nothing left to tell what the lit surface looks like
        return none

    # On wood grain or foliage, surface variation looks exactly like shadow.
    detail = np.abs(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_32F))
    if float(detail[far].mean()) > MAX_SURFACE_DETAIL:
        return none

    sigma = max(1.5, reach / 10)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    observed = _masked_blur(lab, ~_dilate(body, 1), sigma)
    lit = _smooth_fill(np.clip(_masked_blur(lab, far, sigma), 0, 255).astype(np.uint8), far).astype(np.float32)

    darkness = np.clip(1 - (observed[..., 0] + 1) / (lit[..., 0] + 1), 0, 1).astype(np.float32)
    chroma = np.hypot(observed[..., 1] - lit[..., 1], observed[..., 2] - lit[..., 2])
    zone = near & ~_dilate(body, 1)
    candidate = zone & (darkness >= MIN_DARKNESS) & (chroma <= MAX_CHROMA_SHIFT)

    contact = _dilate(body, 2 + int(sigma))
    outer = near & ~_dilate(body, max(2, reach - 2))
    outer_total = max(int(outer.sum()), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    min_area = max(30, int(body.sum() * 0.01))
    gradient = _gradient(lab[..., 0].astype(np.float32))
    band = 3 + int(sigma)
    clear_of_objects = ~_dilate(body, 3)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        part = labels == label
        if not (part & contact).any() or darkness[part].mean() < MIN_MEAN_DARKNESS:
            continue
        if (part & outer).sum() / outer_total > MAX_OUTER_CONTACT:
            continue
        # A drawn shape next to the object has a crisp outline; a cast shadow fades.
        rim = _dilate(part, band) & _dilate(~part, band) & clear_of_objects
        step = float(np.percentile((lit[..., 0] - observed[..., 0])[part], 75))
        keep[label] = not rim.any() or float(np.percentile(gradient[rim], HARD_EDGE_PERCENTILE)) < step * SOFT_EDGE_RATIO
    if not keep.any():
        return none

    soft = cv2.GaussianBlur(keep[labels].astype(np.float32), (0, 0), max(1.0, sigma / 2))
    alpha = np.clip(darkness * soft, 0, MAX_ALPHA).astype(np.float32)
    full = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR) if scale < 1 else alpha
    full[objects] = 0
    full[full < ALPHA_EPS] = 0
    return full


def without_shadow_masks(rgb: np.ndarray, masks: list[np.ndarray]) -> list[np.ndarray]:
    """Drops masks that are really shadows: SAM outlines them like any object.

    A shadow is darker than the surface beside it, keeps that surface's hue,
    has a soft outline where it meets that surface, and touches another object.
    """
    height, width = rgb.shape[:2]
    scale = min(1.0, WORK_SIDE / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    lab = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32)
    gradient = _gradient(lab[..., 0])
    small = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
    union_count = np.sum(small, axis=0)

    kept = []
    for mask, small_mask in zip(masks, small):
        others = (union_count - small_mask.astype(np.int32)) > 0
        keep = True
        area = int(small_mask.sum())
        if area and (_dilate(small_mask, 3) & others).any():
            reach = max(4, int(math.sqrt(area) * 0.15))
            ring = _dilate(small_mask, reach) & ~_dilate(small_mask, 2) & ~_dilate(others, 2)
            # Both sides of the outline: SAM's edge can sit a few pixels off the real one.
            outline = _dilate(small_mask, 3) & _dilate(~small_mask, 3) & ~_dilate(others, 4)
            if ring.sum() >= 20 and outline.sum() >= 10:
                inside, around = lab[small_mask].mean(axis=0), lab[ring].mean(axis=0)
                drop = float(around[0] - inside[0])
                # Per pixel: a mean hides a figure whose colours happen to average grey.
                same_hue = np.hypot(*np.moveaxis(lab[small_mask][:, 1:] - around[1:], 1, 0)) <= MAX_CHROMA_SHIFT
                if drop >= around[0] * MIN_MEAN_DARKNESS and float(same_hue.mean()) >= MIN_SAME_HUE:
                    # Hard edges cross the whole brightness step in about two pixels.
                    keep = float(np.percentile(gradient[outline], HARD_EDGE_PERCENTILE)) >= drop * SOFT_EDGE_RATIO
        if keep:
            kept.append(mask)
    return kept


def relight(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Removes a shadow by undoing its darkening, which keeps surface texture."""
    return np.clip(rgb.astype(np.float32) / (1 - alpha[..., None]) + 0.5, 0, 255).astype(np.uint8)


def make_layer(alpha: np.ndarray, name: str) -> Layer | None:
    box = _bbox(alpha > ALPHA_EPS)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = alpha[y0:y1, x0:x1]
    rgba = np.zeros((*crop.shape, 4), np.uint8)
    rgba[..., 3] = np.clip(crop * 255 + 0.5, 0, 255).astype(np.uint8)
    return Layer(rgba, x0, y0, name, blend_mode="multiply", kind="shadow")
