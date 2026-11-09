"""Turns SAM masks into clean RGBA layers plus a restored background.

Everything here works on full-resolution numpy arrays and only calls the
models through `engine`. The order of operations matters:

    mask -> trimap -> ViTMatte alpha -> inpaint background -> decontaminate

because decontamination solves I = aF + (1-a)B and needs B from the inpaint.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

import cv2
import numpy as np

from .models import engine

Detail = Literal["coarse", "medium", "fine"]
Progress = Callable[[str, float], None]

ALPHA_MIN = 0.02
# Fraction of the image perimeter a mask must cover to count as background.
BACKGROUND_PERIMETER = 0.4
BACKGROUND_AREA = 0.6
# Mean absolute Laplacian below which a surface counts as flat colour or gradient.
SMOOTH_DETAIL = 2.0
SMOOTH_FILL_SIDE = 1024


@dataclass
class Layer:
    rgba: np.ndarray
    left: int
    top: int
    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    blend_mode: str = "normal"
    kind: Literal["raster", "shadow", "text"] = "raster"
    text: dict | None = None


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 3:
        return cv2.dilate(mask.astype(np.uint8), _ellipse(radius)) > 0
    # Morphology cost grows with the kernel; this stays linear on 16MP images.
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Exclusive (x0, y0, x1, y1) bounds, or None for an empty mask."""
    xs = np.flatnonzero(mask.any(axis=0))
    ys = np.flatnonzero(mask.any(axis=1))
    if not len(xs):
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _pad_box(box, pad: int, width: int, height: int):
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def edge_radius(area: int, edge: float) -> int:
    """Half-width of the trimap's unknown band.

    The source paper's fixed 15px kernel x 5 iterations (about 75px) destroys
    small elements, so the band scales with the element instead.
    """
    return int(np.clip(math.sqrt(max(area, 1)) * 0.025 * edge, 2, 64))


def make_trimap(mask: np.ndarray, radius: int, transparent: bool = False) -> np.ndarray:
    """0 = background, 128 = unknown, 255 = foreground."""
    trimap = np.zeros(mask.shape, np.uint8)
    trimap[_dilate(mask, radius)] = 128
    if not transparent:
        trimap[~_dilate(~mask, radius)] = 255
    return trimap


def _drop_islands(alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove matte speckles that are not connected to the selected object."""
    count, labels = cv2.connectedComponents((alpha > 0.05).astype(np.uint8), connectivity=8)
    if count <= 2:
        return alpha
    keep = np.zeros(count, bool)
    keep[np.unique(labels[mask])] = True
    keep[0] = False
    alpha = alpha.copy()
    alpha[~keep[labels]] = 0
    return alpha


def matte(
    rgb: np.ndarray, mask: np.ndarray, edge: float = 1.0, transparent: bool = False
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Soft alpha for `mask`. Returns (alpha crop, crop box, trimap radius)."""
    height, width = rgb.shape[:2]
    box = _bbox(mask)
    if box is None:
        raise ValueError("empty mask")
    radius = edge_radius(int(mask.sum()), edge)
    # The crop must hold the band plus some context for ViTMatte.
    box = _pad_box(box, radius + max(16, radius), width, height)
    x0, y0, x1, y1 = box
    crop_mask = mask[y0:y1, x0:x1]
    trimap = make_trimap(crop_mask, radius, transparent)
    alpha = engine.matte(rgb[y0:y1, x0:x1], trimap)
    alpha[trimap == 0] = 0
    if not transparent:
        alpha[trimap == 255] = 1
    alpha[alpha < ALPHA_MIN] = 0
    return _drop_islands(alpha, crop_mask), box, radius


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((values - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def decontaminate(rgb: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Recover true foreground colour, returning an RGBA uint8 image.

    Solving F = (I - (1-a)B) / a blows up as a -> 0, so faint pixels fall back
    to the colour of the nearest solid pixel of the object.
    """
    a = alpha[..., None].astype(np.float32)
    observed = rgb.astype(np.float32)
    solved = np.clip((observed - (1 - a) * background.astype(np.float32)) / np.maximum(a, 0.05), 0, 255)

    solid = alpha >= 0.95
    if not solid.any():
        solid = alpha >= max(0.5, float(alpha.max()) * 0.9)
    if solid.any():
        _, labels = cv2.distanceTransformWithLabels(
            (~solid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        by_label = np.zeros((int(labels.max()) + 1, 3), np.float32)
        by_label[labels[solid]] = observed[solid]
        nearest = by_label[labels]
    else:
        nearest = observed

    weight = _smoothstep(a, 0.1, 0.55)
    foreground = weight * solved + (1 - weight) * nearest
    foreground[solid] = observed[solid]

    out = np.empty((*alpha.shape, 4), np.uint8)
    out[..., :3] = np.clip(foreground + 0.5, 0, 255)
    out[..., 3] = np.clip(alpha * 255 + 0.5, 0, 255)
    return out


def _smooth_fill(image: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Membrane-like fill from the known pixels (pull-push, then relaxation)."""
    scale = SMOOTH_FILL_SIDE / max(image.shape[:2])
    if scale < 1:
        # The fill has no detail to lose, so large images solve at low resolution.
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        small_known = cv2.resize(known.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.999
        small = _smooth_fill(cv2.resize(image, size, interpolation=cv2.INTER_AREA), small_known)
        filled = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = image.copy()
        out[~known] = filled[~known]
        return out

    def pull_push(colour: np.ndarray, weight: np.ndarray) -> np.ndarray:
        height, width = weight.shape
        if min(height, width) <= 2:
            total = max(float(weight.sum()), 1e-6)
            mean = (colour * weight[..., None]).sum((0, 1)) / total
            return np.broadcast_to(mean, colour.shape).astype(np.float32).copy()
        size = (max(1, width // 2), max(1, height // 2))
        small_weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
        premultiplied = cv2.resize(colour * weight[..., None], size, interpolation=cv2.INTER_AREA)
        small = premultiplied / np.maximum(small_weight[..., None], 1e-6)
        filled = pull_push(small, (small_weight > 0).astype(np.float32))
        up = cv2.resize(filled, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.where(weight[..., None] > 0, colour, up)

    weight = known.astype(np.float32)
    out = pull_push(image.astype(np.float32) * weight[..., None], weight)
    hole = ~known
    for _ in range(40):
        out[hole] = cv2.blur(out, (5, 5))[hole]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _surroundings_are_smooth(gray_detail: np.ndarray, component: np.ndarray, reach: int) -> bool:
    """True when a hole sits on flat colour or a gradient rather than texture."""
    ring = _dilate(component, reach * 3) & ~_dilate(component, reach)
    return bool(ring.any()) and float(gray_detail[ring].mean()) < SMOOTH_DETAIL


def inpaint_region(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill `hole` in `image`.

    LaMa hallucinates ghosts on flat design backgrounds, where a smooth fill is
    exact, so each separate hole gets whichever suits what surrounds it.
    """
    if _bbox(hole) is None:
        return image
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    count, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    out = image.copy()
    needs_lama = np.zeros(hole.shape, bool)
    for label in range(1, count):
        part = labels == label
        part_box = _bbox(part)
        reach = max(3, int(max(part_box[2] - part_box[0], part_box[3] - part_box[1]) * 0.02))
        x0, y0, x1, y1 = _pad_box(part_box, reach * 4, width, height)
        if _surroundings_are_smooth(detail[y0:y1, x0:x1], part[y0:y1, x0:x1], reach):
            region = out[y0:y1, x0:x1]
            filled = _smooth_fill(region, ~hole[y0:y1, x0:x1])
            local = part[y0:y1, x0:x1]
            region[local] = filled[local]
        else:
            needs_lama |= part

    box = _bbox(needs_lama)
    if box is not None:
        context = max(64, int(max(box[2] - box[0], box[3] - box[1]) * 0.6))
        x0, y0, x1, y1 = _pad_box(box, context, width, height)
        out[y0:y1, x0:x1] = engine.inpaint(out[y0:y1, x0:x1], needs_lama[y0:y1, x0:x1])
    return out


def _hole_radius(width: int, height: int, radius: int) -> int:
    # Beyond the visible alpha: LaMa would otherwise copy any halo at the rim.
    return max(radius, max(4, int(max(width, height) * 0.008)))


def _background_hole(alpha: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    return _dilate(alpha > ALPHA_MIN, _hole_radius(width, height, radius))


def extract_element(
    rgb: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    name: str,
    edge: float = 1.0,
    transparent: bool = False,
    with_shadow: bool = False,
) -> tuple[list[Layer], np.ndarray]:
    """Lift one element out of the picture. Returns (layers, new background)."""
    height, width = rgb.shape[:2]
    alpha, (x0, y0, x1, y1), radius = matte(rgb, mask, edge, transparent)
    full = np.zeros((height, width), bool)
    full[y0:y1, x0:x1] = alpha > ALPHA_MIN
    layers: list[Layer] = []
    if with_shadow:
        from . import shadow

        darkening = shadow.estimate(background, full)
        shadow_layer = shadow.make_layer(darkening, f"Bóng đổ của {name}")
        if shadow_layer is not None:
            layers.append(shadow_layer)
            background = shadow.relight(background, darkening)
    new_background = inpaint_region(background, _background_hole(full, width, height, radius))
    rgba = decontaminate(rgb[y0:y1, x0:x1], new_background[y0:y1, x0:x1], alpha)
    layers.append(_trim(rgba, x0, y0, name))
    return layers, new_background


def _trim(rgba: np.ndarray, left: int, top: int, name: str, blend_mode: str = "normal") -> Layer:
    box = _bbox(rgba[..., 3] > 0)
    if box is None:
        return Layer(rgba[:1, :1], left, top, name, blend_mode=blend_mode)
    x0, y0, x1, y1 = box
    return Layer(rgba[y0:y1, x0:x1].copy(), left + x0, top + y0, name, blend_mode=blend_mode)


# --------------------------------------------------------------------------
# Full-scene decomposition
# --------------------------------------------------------------------------

_DETAIL = {
    # min area, smallest share of its parent a nested mask may have,
    # how many of each grid point's valid masks to keep, largest first
    "coarse": (0.004, None, 1),
    "medium": (0.0015, 0.04, 2),
    "fine": (0.0005, 0.0, 3),
}


def select_masks(
    logits: np.ndarray, scores: np.ndarray, point_ids: np.ndarray, detail: Detail, max_layers: int
) -> list[int]:
    """Pick the indices of the candidate masks worth keeping, biggest first."""
    min_area, min_share, per_point = _DETAIL[detail]
    masks = logits > 0
    size = masks.shape[-1]
    area = masks.sum(axis=(1, 2))
    stability = (logits > 1).sum(axis=(1, 2)) / np.maximum((logits > -1).sum(axis=(1, 2)), 1)

    perimeter = (
        masks[:, 0, :].sum(1) + masks[:, -1, :].sum(1) + masks[:, :, 0].sum(1) + masks[:, :, -1].sum(1)
    ) / (4.0 * size)
    is_background = (perimeter >= BACKGROUND_PERIMETER) | (area >= BACKGROUND_AREA * size * size)
    valid = (scores >= 0.7) & (stability >= 0.85) & (area >= min_area * size * size) & ~is_background

    # The largest mask a point yields is the whole object, unless that mask was
    # the background: then the next one down is the object.
    by_size = np.lexsort((-area, point_ids))
    taken: dict[int, int] = {}
    for index in by_size:
        if valid[index]:
            count = taken.get(int(point_ids[index]), 0)
            taken[int(point_ids[index])] = count + 1
            valid[index] = count < per_point

    # Non-maximum suppression, best predicted quality first. Thousands of
    # candidates are compared, so this runs on quarter-resolution masks.
    coarse = masks[:, ::4, ::4].reshape(len(masks), -1)
    coarse_area = coarse.sum(1)
    kept_coarse = np.zeros((int(valid.sum()), coarse.shape[1]), bool)
    kept: list[int] = []
    for index in np.argsort(-scores):
        if not valid[index]:
            continue
        if kept:
            overlap = (kept_coarse[: len(kept)] & coarse[index]).sum(1)
            union = coarse_area[kept] + coarse_area[index] - overlap
            if (overlap / np.maximum(union, 1)).max() > 0.7:
                continue
        kept_coarse[len(kept)] = coarse[index]
        kept.append(int(index))

    # Resolve nesting, starting from the largest mask.
    flat = masks.reshape(len(masks), -1)
    kept.sort(key=lambda i: -area[i])
    chosen: list[int] = []
    for index in kept:
        reject = False
        for parent in chosen:
            contained = (flat[parent] & flat[index]).sum() / area[index]
            if contained < 0.9:
                continue
            share = area[index] / area[parent]
            # Almost the whole parent again, or a part too small for this level.
            if share > 0.75 or min_share is None or share < min_share:
                reject = True
                break
        if not reject:
            chosen.append(index)

    if len(chosen) > max_layers:
        # Keep the most prominent, then restore bottom-to-top order.
        chosen = sorted(sorted(chosen, key=lambda i: -area[i] * scores[i])[:max_layers], key=lambda i: -area[i])
    return chosen


def _upsample(logits: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0


def _clean_visible(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop slivers left behind when upper layers cut this one apart."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= min_pixels
    keep[0] = False
    return keep[labels]


def _fill_interior(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def _occluded_hull(mask: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """The hull of a shape cut by something on top, if the shape is near convex.

    Rectangles, discs and plates complete well this way. For anything else the
    hull would invent area, so the test requires that what is hidden explains
    nearly all of the difference between the shape and its hull.
    """
    points = cv2.findNonZero(mask.astype(np.uint8))
    hull_mask = np.zeros(mask.shape, np.uint8)
    if points is None:
        return hull_mask > 0
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(points), 1)
    hull = hull_mask > 0
    completed = (mask | (hull & upper)).sum()
    return hull if completed >= 0.95 * hull.sum() else np.zeros(mask.shape, bool)


def decompose(
    rgb: np.ndarray,
    embeddings,
    detail: Detail = "medium",
    max_layers: int = 24,
    edge: float = 1.0,
    with_shadows: bool = False,
    progress: Progress = lambda stage, value: None,
) -> tuple[list[Layer], np.ndarray]:
    """Split a whole picture into layers, returned bottom to top."""
    height, width = rgb.shape[:2]
    pixels = width * height

    progress("Đang tìm các phần tử", 0.02)
    logits, scores, point_ids = engine.auto_candidates(
        rgb, embeddings, progress=lambda v: progress("Đang tìm các phần tử", 0.02 + v * 0.28)
    )
    chosen = select_masks(logits, scores, point_ids, detail, max_layers)

    # Every pixel belongs to exactly one layer: the top-most mask covering it.
    # Masks are ordered big to small, so small elements end up on top.
    owner = np.full((height, width), -1, np.int16)
    for z, index in enumerate(chosen):
        owner[_upsample(logits[index], width, height)] = z
    min_pixels = max(32, int(pixels * 0.0003))
    visible = [_clean_visible(owner == z, min_pixels) for z in range(len(chosen))]
    visible = [mask for mask in visible if mask.sum() >= min_pixels]
    if with_shadows and len(visible) > 1:
        from . import shadow

        visible = shadow.without_shadow_masks(rgb, visible)

    if not visible:
        return [], rgb.copy()

    # Matte everything first: the background hole needs every alpha.
    alphas: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    hole = np.zeros((height, width), bool)
    for i, mask in enumerate(visible):
        progress("Đang làm mịn biên", 0.3 + 0.45 * i / len(visible))
        alpha, box, radius = matte(rgb, mask, edge)
        alphas.append((alpha, box, radius))
        x0, y0, x1, y1 = box
        hole[y0:y1, x0:x1] |= alpha > ALPHA_MIN
    # Shadows come off first, or LaMa would extend them across the hole.
    source = rgb
    shadow_parts: dict[int, Layer] = {}
    if with_shadows:
        from . import shadow

        progress("Đang tách bóng đổ", 0.75)
        darkening = shadow.estimate(rgb, hole)
        if darkening.any():
            source = shadow.relight(rgb, darkening)
            # Each shadow belongs to the object nearest to it.
            owners = np.full((height, width), -1, np.int32)
            for z, mask in enumerate(visible):
                owners[mask] = z
            _, labels = cv2.distanceTransformWithLabels(
                (owners < 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
            )
            by_label = np.zeros(int(labels.max()) + 1, np.int32)
            by_label[labels[owners >= 0]] = owners[owners >= 0]
            nearest = by_label[labels]
            for z in range(len(visible)):
                layer = shadow.make_layer(np.where(nearest == z, darkening, 0).astype(np.float32), f"Bóng đổ {z + 1}")
                if layer is not None:
                    shadow_parts[z] = layer

    progress("Đang vá nền", 0.77)
    background = inpaint_region(source, _dilate(hole, _hole_radius(width, height, 0)))

    # Union of everything above layer z, so lower layers can extend under it.
    above = np.zeros((height, width), bool)
    uppers: list[np.ndarray] = []
    for mask in reversed(visible):
        uppers.append(above.copy())
        above |= mask
    uppers.reverse()

    # Bottom to top: `under` is the picture so far, the B for decontamination.
    layers: list[Layer] = []
    under = background.copy()
    for i, (mask, (alpha, box, radius), upper) in enumerate(zip(visible, alphas, uppers)):
        progress("Đang dựng các layer", 0.8 + 0.2 * i / len(visible))
        if i in shadow_parts:
            layers.append(shadow_parts[i])
            _composite(under, shadow_parts[i])
        x0, y0, x1, y1 = box
        crop_rgb = rgb[y0:y1, x0:x1]
        crop_upper = upper[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        rgba = decontaminate(crop_rgb, under[y0:y1, x0:x1], alpha)

        if crop_upper.any():
            rgba = _extend_under(rgba, crop_rgb, crop_mask, crop_upper, radius)

        layer = _trim(rgba, x0, y0, f"Phần tử {i + 1}")
        layers.append(layer)
        _composite(under, layer)

    return layers, background


def _extend_under(
    rgba: np.ndarray, rgb: np.ndarray, mask: np.ndarray, upper: np.ndarray, radius: int
) -> np.ndarray:
    """Make a layer whole where the layers above hide it.

    Closes the seam along upper layers' soft edges, and paints enclosed holes
    so that moving the top element does not expose a hole in this one.
    """
    rgba = rgba.copy()
    grow = max(2, radius)
    seam = _dilate(mask, grow) & _dilate(upper, grow)
    interior = (_fill_interior(mask | seam) | _occluded_hull(mask, upper)) & upper & ~mask

    if interior.any():
        # Filled from this layer's own pixels only, never from what lies on top.
        painted = _smooth_fill(rgb, mask)
        rgba[..., :3][interior] = painted[interior]
        rgba[..., 3][interior] = 255

    # Seam pixels outside the layer take the nearest colour the layer owns.
    outside = seam & ~mask & ~interior
    if outside.any() and mask.any():
        _, labels = cv2.distanceTransformWithLabels(
            (~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        by_label = np.zeros((int(labels.max()) + 1, 3), np.uint8)
        by_label[labels[mask]] = rgb[mask]
        rgba[..., :3][outside] = by_label[labels[outside]]
    rgba[..., 3][seam] = 255
    return rgba


def _composite(under: np.ndarray, layer: Layer) -> None:
    height, width = layer.rgba.shape[:2]
    target = under[layer.top : layer.top + height, layer.left : layer.left + width]
    alpha = layer.rgba[..., 3:].astype(np.float32) / 255
    target[...] = np.clip(alpha * layer.rgba[..., :3] + (1 - alpha) * target + 0.5, 0, 255)


def flatten(background: np.ndarray, layers: list[Layer]) -> np.ndarray:
    out = background.copy()
    for layer in layers:
        _composite(out, layer)
    return out
