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
# Lab distance within which a pixel still counts as the plain page.
PAGE_COLOUR_DISTANCE = 7.0

# Finding what the point grid missed.
GAP_ROUNDS = 6
SHADOW_CLEARANCE = 12
MAX_GAP_LAYERS = 24
GAP_MARGIN = 2
GAP_MIN_AREA = 0.001
SHAPE_EDGE = 6.0
SHAPE_SAME_COLOUR = 4.0
PAGE_EDGE = 3.0
# Drawn art has crisp edges, so the matting band can be much narrower.
FLAT_EDGE_SCALE = 0.4
CRUMB_MIN_PIXELS = 16
GAP_MIN_SCORE = 0.5
GAP_MIN_NEW = 0.6
GAP_MAX_PAGE = 0.1

# Whole objects from parts, on flat artwork.
SNAP_SHARE = 0.5
SNAP_RIM = 2
NESTED_SHARE = 0.9
CONTAINER_SHARE = 0.45
BLOCK_SHARE = 0.3
BLOCK_FILL = 0.75
PART_MAX_SIZE = 0.5
PART_MAX_ASPECT = 3.0
PART_MIN_BORDER = 0.1
PALE_PART_MIN_BORDER = 0.3
PALE_DISTANCE = 40.0
# How far a shape is assumed to continue under what covers it, relative to its size.
CLOSING_REACH = 0.15
FIGURE_REACH = 0.06
LEARNED_FILL_MIN = 64
FILL_MIN_PIXELS = 64
FILL_MIN_SHARE = 0.01
EVEN_AREA_STD = 8.0
CONTACT_SHADE = 0.06
LEARNED_COLOUR_SCALE = 0.04
# Holes at least this share of the picture get their colour from a smooth fill.
LARGE_HOLE = 0.03
CONTACT_KEEP = 0.5
FRINGE = 2

# What a run of letters looks like, relative to its line height.
GLYPH_MIN_HEIGHT = 12
GLYPH_MAX_ASPECT = 2.5
GLYPH_MIN_HEIGHT_RATIO = 0.6
GLYPH_TOP_ALIGN = 0.45
GLYPH_BASE_ALIGN = 0.15
GLYPH_MAX_GAP = 0.35
WORD_MAX_GAP = 1.2
# Looser than for letters: a glyph with a drop shadow has a different median.
WORD_COLOUR_DISTANCE = 60.0
STRAY_REACH = 6
LINE_MARGIN = 0.12
GLYPH_COLOUR_DISTANCE = 12.0


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
    # Letters joined into a word: the OCR gets a second, cleaner look at these.
    is_word: bool = False


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
        region, local = out[y0:y1, x0:x1], needs_lama[y0:y1, x0:x1]
        filled = engine.inpaint(region, local)
        if local.sum() >= LARGE_HOLE * hole.size:
            filled = _keep_colour(filled, _smooth_fill(region, ~hole[y0:y1, x0:x1]), local)
        out[y0:y1, x0:x1] = filled
    return out


def _keep_colour(learned: np.ndarray, base: np.ndarray, area: np.ndarray) -> np.ndarray:
    """LaMa's texture over the colour a smooth fill gives.

    Across a large hole LaMa drifts toward grey and dark; the smooth fill from
    the surrounding pixels holds the colour those pixels actually have.
    """
    sigma = max(3.0, max(learned.shape[:2]) * LEARNED_COLOUR_SCALE)
    learned_f = learned.astype(np.float32)
    corrected = learned_f + cv2.GaussianBlur(base.astype(np.float32), (0, 0), sigma) - cv2.GaussianBlur(learned_f, (0, 0), sigma)
    out = learned.copy()
    out[area] = np.clip(corrected[area] + 0.5, 0, 255).astype(np.uint8)
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


def _colour_edges(rgb: np.ndarray) -> np.ndarray:
    """Strongest change per pixel across the three Lab channels."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.max(np.abs(cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)), axis=2) / 8


def _find_page(rgb: np.ndarray, logits: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    """The plain surface a design sits on, or None for a photographic scene.

    Grown by colour so that thin slivers SAM left out still count as surface.
    """
    height, width = rgb.shape[:2]
    masks = logits > 0
    size = masks.shape[-1]
    perimeter = (
        masks[:, 0, :].sum(1) + masks[:, -1, :].sum(1) + masks[:, :, 0].sum(1) + masks[:, :, -1].sum(1)
    ) / (4.0 * size)
    surfaces = np.flatnonzero(perimeter >= BACKGROUND_PERIMETER)
    if not len(surfaces):
        return None
    area = masks[surfaces].sum(axis=(1, 2))
    page = _upsample(logits[surfaces[int(np.argmax(area * scores[surfaces]))]], width, height)
    # SAM often lumps pale shapes in with the page, but a drawn edge separates them.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (page & (_colour_edges(rgb) < PAGE_EDGE)).astype(np.uint8), connectivity=4
    )
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.zeros(count, bool)
    keep[border[border > 0]] = True
    keep &= stats[:, cv2.CC_STAT_AREA] >= 0.02 * height * width
    page = keep[labels]
    core = ~_dilate(~page, 3)
    if not core.any():
        return None
    detail = np.abs(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F))
    if float(detail[core].mean()) >= SMOOTH_DETAIL:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    surface = cv2.cvtColor(_smooth_fill(rgb, core), cv2.COLOR_RGB2LAB)
    difference = np.linalg.norm(lab.astype(np.float32) - surface.astype(np.float32), axis=2)
    # Must reach the image border: an eye or a shirt can share the page's colour.
    count, labels = cv2.connectedComponents((difference <= PAGE_COLOUR_DISTANCE).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.zeros(count, bool)
    keep[border[border > 0]] = True
    return keep[labels]


def _flat_shapes(rgb: np.ndarray, region: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    """Cuts a leftover region along its colour edges, for drawn art SAM cannot outline."""
    flat = region & (_colour_edges(rgb) < SHAPE_EDGE)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8), connectivity=4)
    shapes = [labels == label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_pixels]
    # The anti-aliased rim that was cut away goes back to the shape it surrounds.
    return [_dilate(shape, 2) & region for shape in shapes] or [region]


def _fill_gaps(
    rgb: np.ndarray, embeddings, selected: list[np.ndarray], page: np.ndarray, min_pixels: int
) -> list[np.ndarray]:
    """Masks for whatever is neither surface nor an element yet.

    The grid can miss an object whose masks all look uncertain. Prompting at the
    middle of each leftover region finds it, so nothing stays baked into the
    background. What SAM cannot outline is cut along its colour edges instead,
    and any crumbs left at the end share one layer.
    """
    height, width = rgb.shape[:2]
    covered = np.logical_or.reduce(selected) if selected else np.zeros((height, width), bool)
    # The margin absorbs the slivers along SAM's slightly-off outlines.
    residual = ~page & ~_dilate(covered, GAP_MARGIN)
    added: list[np.ndarray] = []
    shapes: list[np.ndarray] = []

    for _ in range(GAP_ROUNDS):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), connectivity=8)
        regions = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_pixels]
        regions = sorted(regions, key=lambda label: -stats[label, cv2.CC_STAT_AREA])[
            : MAX_GAP_LAYERS - len(added) - len(shapes)
        ]
        if not regions:
            break
        points = []
        for label in regions:
            # The deepest point cannot fall on the region's edge.
            depth = cv2.distanceTransform((labels == label).astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
            points.append((float(x), float(y)))
        logits, scores, point_ids = engine.candidates_at(rgb, embeddings, points)

        for point, label in enumerate(regions):
            region = labels == label
            best: np.ndarray | None = None
            indices = np.flatnonzero(point_ids == point)
            for index in indices[np.argsort(-(logits[indices] > 0).sum(axis=(1, 2)))]:
                if scores[index] < GAP_MIN_SCORE:
                    continue
                mask = _upsample(logits[index], width, height)
                area = int(mask.sum())
                if area < min_pixels:
                    continue
                if (mask & residual).sum() >= GAP_MIN_NEW * area and (mask & page).sum() <= GAP_MAX_PAGE * area:
                    best = mask
                    break
            if best is not None:
                added.append(best)
                residual &= ~best
            else:
                shapes += _flat_shapes(rgb, region, min_pixels)
                residual &= ~region
        if len(added) + len(shapes) >= MAX_GAP_LAYERS:
            break
    return added + _merge_same_colour(rgb, shapes)


def _merge_same_colour(rgb: np.ndarray, shapes: list[np.ndarray]) -> list[np.ndarray]:
    """One shape that objects on top cut into pieces becomes one layer again."""
    if len(shapes) < 2:
        return shapes
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    merged: list[tuple[np.ndarray, np.ndarray]] = []
    for shape in sorted(shapes, key=lambda mask: -int(mask.sum())):
        colour = np.median(lab[shape], axis=0)
        for index, (mask, seen) in enumerate(merged):
            if float(np.linalg.norm(colour - seen)) <= SHAPE_SAME_COLOUR:
                merged[index] = (mask | shape, seen)
                break
        else:
            merged.append((shape, colour))
    return [mask for mask, _ in merged]


def _crumbs(page: np.ndarray, visible: list[np.ndarray]) -> np.ndarray | None:
    """Strokes and specks too small to stand alone, kept off the background."""
    leftovers = ~page & ~_dilate(np.logical_or.reduce(visible), GAP_MARGIN)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(leftovers.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= CRUMB_MIN_PIXELS
    keep[0] = False
    return keep[labels] if keep.any() else None


def _flat_regions(rgb: np.ndarray, page: np.ndarray) -> np.ndarray:
    """Connected areas of even colour; 0 marks edges and anything page-like.

    White on a pale page has so faint an edge that the two flood together, and
    the inside of an outlined icon is just the page again. Neither can tell
    where an element ends, so those pixels are left to SAM's outline.
    """
    flat = (_colour_edges(rgb) < SHAPE_EDGE).astype(np.uint8)
    count, labels = cv2.connectedComponents(flat, connectivity=4)
    sizes = np.maximum(np.bincount(labels.ravel(), minlength=count), 1)
    on_page = np.bincount(labels[page], minlength=count) / sizes
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = np.stack([np.bincount(labels.ravel(), lab[..., c].ravel(), minlength=count) / sizes for c in range(3)], axis=1)
    surface = np.median(lab[page], axis=0)
    page_like = (on_page >= 0.5) | (np.linalg.norm(mean - surface, axis=1) <= PAGE_COLOUR_DISTANCE)
    page_like[0] = True
    labels[page_like[labels]] = 0
    return labels


def _snap(mask: np.ndarray, regions: np.ndarray, sizes: np.ndarray, page: np.ndarray) -> np.ndarray:
    """Moves SAM's soft 256px outline onto the crisp colour edges of drawn art."""
    inside = np.bincount(regions[mask], minlength=len(sizes))
    keep = inside >= SNAP_SHARE * sizes
    keep[0] = False
    solid = keep[regions]
    edges = regions == 0
    snapped = (solid | (edges & (mask | _dilate(solid, SNAP_RIM)))) & ~page
    return snapped if snapped.any() else mask


def _is_container(mask: np.ndarray, regions: np.ndarray) -> bool:
    """True for a surface that things sit on: mostly one area of even colour."""
    values = regions[mask]
    values = values[values > 0]
    if not len(values):
        return False
    counts = np.bincount(values)
    largest, share = int(counts.argmax()), float(counts.max()) / float(mask.sum())
    if share >= CONTAINER_SHARE:
        return True
    if share < BLOCK_SHARE:
        return False
    # A panel stays a solid block even when its contents punch holes in it;
    # the biggest area of a figure, such as its hair, is an irregular patch.
    region = regions == largest
    x0, y0, x1, y1 = _bbox(region)
    return float(_fill_interior(region[y0:y1, x0:x1]).mean()) >= BLOCK_FILL


def _is_surface(mask: np.ndarray, regions: np.ndarray, found_in_gaps: np.ndarray) -> bool:
    """Something objects stand on. What the object grid missed is never a figure."""
    return bool((mask & found_in_gaps).sum() >= 0.5 * mask.sum()) or _is_container(mask, regions)


def _absorb_nested(masks: list[np.ndarray], regions: np.ndarray, found_in_gaps: np.ndarray) -> list[np.ndarray]:
    """Parts inside an object stay with it; only surfaces keep separate children.

    A skirt belongs to the figure wearing it, but a chart stands on its board.
    """
    kept: list[np.ndarray] = []
    filled: list[np.ndarray] = []
    surface: list[bool] = []
    for mask in sorted(masks, key=lambda m: -int(m.sum())):
        area = max(int(mask.sum()), 1)
        parent = next(
            (i for i in range(len(kept) - 1, -1, -1) if (filled[i] & mask).sum() >= NESTED_SHARE * area), None
        )
        if parent is not None and not surface[parent]:
            kept[parent] = kept[parent] | mask
            continue
        kept.append(mask)
        filled.append(_fill_interior(mask))
        surface.append(_is_surface(mask, regions, found_in_gaps))
    return kept


def _merge_parts(
    rgb: np.ndarray, masks: list[np.ndarray], regions: np.ndarray, page: np.ndarray, found_in_gaps: np.ndarray
) -> list[np.ndarray]:
    """Joins a piece to the object it clearly belongs to: legs, a held clipboard.

    Pale pieces need a far longer shared border, or the ellipse under a figure's
    feet and a speech bubble touching its head would be swallowed too.
    """
    masks = list(masks)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    surface_lab = np.median(lab[page], axis=0) if page.any() else lab.mean((0, 1))
    alive = [True] * len(masks)
    surfaces = [_is_surface(mask, regions, found_in_gaps) for mask in masks]
    for index in sorted(range(len(masks)), key=lambda i: int(masks[i].sum())):
        mask = masks[index]
        area = int(mask.sum())
        x0, y0, x1, y1 = _bbox(mask)
        # Long strips are lines of text, bars and rules, not limbs.
        if max(x1 - x0, y1 - y0) > PART_MAX_ASPECT * min(x1 - x0, y1 - y0):
            continue
        rim = _dilate(mask, 2) & ~mask
        perimeter = max(int(rim.sum()), 1)
        best, shared = -1, 0
        for other in range(len(masks)):
            if other == index or not alive[other] or surfaces[other] or area > PART_MAX_SIZE * int(masks[other].sum()):
                continue
            count = int((rim & masks[other]).sum())
            if count > shared:
                best, shared = other, count
        if best < 0:
            continue
        pale = float(np.linalg.norm(np.median(lab[mask], axis=0) - surface_lab)) < PALE_DISTANCE
        if shared / perimeter >= (PALE_PART_MIN_BORDER if pale else PART_MIN_BORDER):
            masks[best] = masks[best] | mask
            alive[index] = False
    return [mask for mask, live in zip(masks, alive) if live]


def _group_words(rgb: np.ndarray, masks: list[np.ndarray]) -> tuple[list[np.ndarray], set[int]]:
    """Joins glyphs that SAM outlined one by one into a single word layer.

    A run of letters is the only common thing that shares colour, height, top
    and bottom while standing closer together than a fraction of its height.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    boxes = [_bbox(mask) for mask in masks]
    glyphs: list[int] = []
    for i, (mask, (x0, y0, x1, y1)) in enumerate(zip(masks, boxes)):
        if y1 - y0 >= GLYPH_MIN_HEIGHT and (x1 - x0) <= GLYPH_MAX_ASPECT * (y1 - y0):
            glyphs.append(i)
    colours = {i: np.median(lab[masks[i]], axis=0) for i in glyphs}

    parent = list(range(len(masks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def same_line(i: int, j: int, max_gap: float, max_colour: float = GLYPH_COLOUR_DISTANCE) -> bool:
        ax0, ay0, ax1, ay1 = boxes[i]
        bx0, by0, bx1, by1 = boxes[j]
        line = max(ay1 - ay0, by1 - by0)
        # Tops may differ more than baselines: accents make some glyphs taller.
        return (
            min(ay1 - ay0, by1 - by0) >= GLYPH_MIN_HEIGHT_RATIO * line
            and abs(ay0 - by0) <= GLYPH_TOP_ALIGN * line
            and abs(ay1 - by1) <= GLYPH_BASE_ALIGN * line
            and max(ax0, bx0) - min(ax1, bx1) <= max_gap * line
            and float(np.linalg.norm(colours[i] - colours[j])) <= max_colour
        )

    pairs = [(i, j) for position, i in enumerate(glyphs) for j in glyphs[position + 1 :]]
    for i, j in pairs:
        if same_line(i, j, GLYPH_MAX_GAP):
            parent[find(j)] = find(i)
    # Words of one line, but only once letters have been recognised on a side:
    # two lone speech bubbles side by side must stay separate.
    joined = True
    while joined:
        joined = False
        sizes = np.bincount([find(i) for i in range(len(masks))], minlength=len(masks))
        for i, j in pairs:
            if (
                find(i) != find(j)
                and max(sizes[find(i)], sizes[find(j)]) >= 2
                and same_line(i, j, WORD_MAX_GAP, WORD_COLOUR_DISTANCE)
            ):
                parent[find(j)] = find(i)
                joined = True
                break

    # Each group keeps the place of its first member: the list is the z-order.
    groups: dict[int, np.ndarray] = {}
    members: dict[int, int] = {}
    for i, mask in enumerate(masks):
        root = find(i)
        groups[root] = groups[root] | mask if root in groups else mask
        members[root] = members.get(root, 0) + 1
    words = [root for root, count in members.items() if count >= 2]
    for root in words:
        lines = np.array([boxes[i] for i in range(len(masks)) if find(i) == root])
        groups[root] = _trim_to_line(groups, root, lines)
    return list(groups.values()), {id(groups[root]) for root in words}


def _trim_to_line(groups: dict[int, np.ndarray], root: int, lines: np.ndarray) -> np.ndarray:
    """Hands pieces of the next text line back: SAM's glyph masks can overlap it."""
    word = groups[root]
    # Low percentile: several glyphs may have flooded into the line below.
    top, bottom = float(np.median(lines[:, 1])), float(np.percentile(lines[:, 3], 25))
    height = bottom - top
    below = word.copy()
    below[: min(word.shape[0], int(bottom + height * LINE_MARGIN))] = False
    if not below.any():
        return word
    # Find the other line this piece belongs to, by where it sits.
    count, labels = cv2.connectedComponents(below.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        part = labels == label
        px0, py0, px1, py1 = _bbox(part)
        for other, mask in groups.items():
            if other == root:
                continue
            x0, y0, x1, y1 = _bbox(mask)
            if px0 < x1 + STRAY_REACH and x0 - STRAY_REACH < px1 and py0 < y1 + STRAY_REACH and y0 - STRAY_REACH < py1:
                groups[other] = mask | part
                break
    return word & ~below


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
    max_layers: int = 40,
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
    selected = [_upsample(logits[index], width, height) for index in chosen]

    page = _find_page(rgb, logits, scores)
    if page is not None:
        progress("Đang tìm phần còn sót", 0.3)
        gaps = _fill_gaps(rgb, embeddings, selected, page, max(32, int(pixels * GAP_MIN_AREA)))
        selected += gaps
        found_in_gaps = np.logical_or.reduce(gaps) if gaps else np.zeros((height, width), bool)
        regions = _flat_regions(rgb, page)
        sizes = np.bincount(regions.ravel())
        selected = [_snap(mask, regions, sizes, page) for mask in selected]
        if detail != "fine":
            selected = _absorb_nested(selected, regions, found_in_gaps)
        selected.sort(key=lambda mask: -int(mask.sum()))

    from . import scene

    progress("Đang nhận diện vật thể", 0.3)
    understood = scene.understand(rgb)
    min_pixels = max(32, int(pixels * 0.0003))
    names: dict[int, str] = {}
    word_ids: set[int] = set()
    photo = page is None and understood is not None
    if photo:
        objects = [_upsample(logits[i], width, height) for i in select_masks(logits, scores, point_ids, "medium", 80)]
        pieces = select_masks(logits, scores, point_ids, "fine", 200)
        owner = np.full((height, width), -1, np.int16)
        for z, index in enumerate(pieces):
            owner[_upsample(logits[index], width, height)] = z
        cells = [owner == z for z in range(len(pieces))]
        masks, layer_names = scene.photo_layers(understood, objects, cells, detail, max_layers)
        visible = []
        for mask, name in zip(masks, layer_names):
            mask = _clean_visible(mask, min_pixels)
            if mask.sum() >= min_pixels:
                visible.append(mask)
                names[id(mask)] = name
    else:
        # Every pixel belongs to exactly one layer: the top-most mask covering it.
        # Masks are ordered big to small, so small elements end up on top.
        owner = np.full((height, width), -1, np.int16)
        for z, mask in enumerate(selected):
            owner[mask] = z
        visible = [_clean_visible(owner == z, min_pixels) for z in range(len(selected))]
        visible = [mask for mask in visible if mask.sum() >= min_pixels]
    if visible and not photo:
        visible, word_ids = _group_words(rgb, visible)
        if page is not None and detail != "fine":
            visible = _merge_parts(rgb, visible, regions, page, found_in_gaps)
            # Too small to be worth a row in the layer list: crumbs picks them up.
            visible = [mask for mask in visible if mask.sum() >= pixels * GAP_MIN_AREA]
        if page is not None:
            # Figures stand in front of the panels and ellipses around them.
            visible.sort(key=lambda mask: not _is_surface(mask, regions, found_in_gaps))
        crumbs = None if page is None else _crumbs(page, visible)
        if crumbs is not None:
            visible.append(crumbs)
        names = scene.name_flat_layers(understood, visible)
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
        alpha, box, radius = matte(rgb, mask, edge * FLAT_EDGE_SCALE if page is not None else edge)
        alphas.append((alpha, box, radius))
        x0, y0, x1, y1 = box
        hole[y0:y1, x0:x1] |= alpha > ALPHA_MIN
    # Shadows come off first, or LaMa would extend them across the hole.
    source = rgb
    darkening = None
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
    clear_hole = _dilate(hole, _hole_radius(width, height, 0))
    # On a plain page the true background is known: only guessing could spoil it.
    if page is None:
        background = inpaint_region(source, clear_hole)
    else:
        # Soft shadows fade below the page's colour tolerance, so keep well clear.
        if darkening is not None and darkening.any():
            clear_hole |= _dilate(darkening > 0, SHADOW_CLEARANCE)
        background = _smooth_fill(source, page & ~clear_hole)

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

        # Letters have no hidden part worth guessing; a word stops where it ends.
        if crop_upper.any() and id(mask) not in word_ids:
            rgba = _extend_under(
                rgba, crop_rgb, crop_mask, crop_upper, radius,
                flat=page is not None,
                surface=page is not None and _is_surface(mask, regions, found_in_gaps),
            )

        layer = _trim(rgba, x0, y0, names.get(id(mask), f"Phần tử {i + 1}"))
        layer.is_word = id(mask) in word_ids
        layers.append(layer)
        _composite(under, layer)

    return layers, background


def _continue_colours(rgb: np.ndarray, source: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Carries each area of even colour straight on under what hides it.

    Only clean pixels are sources: an anti-aliased edge pixel would otherwise
    be drawn out into a streak across the hidden part.
    """
    clean = source & (_colour_edges(rgb) < SHAPE_EDGE)
    # Specks left over from other objects would be drawn out into stains too.
    count, parts, stats, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=4)
    large = stats[:, cv2.CC_STAT_AREA] >= max(FILL_MIN_PIXELS, FILL_MIN_SHARE * float(source.sum()))
    large[0] = False
    clean = large[parts] if large.any() else clean
    if not clean.any():
        clean = source
    _, labels = cv2.distanceTransformWithLabels((~clean).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # Compression ringing tints the pixels around every edge. Drawn outward, one
    # tinted pixel becomes a whole ray, so an even area gives its typical colour.
    colours = rgb.copy()
    for label in np.flatnonzero(large):
        part = parts == label
        if float(rgb[part].std(axis=0).max()) <= EVEN_AREA_STD:
            colours[part] = np.median(rgb[part], axis=0)
    by_label = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    by_label[labels[clean]] = colours[clean]
    painted = rgb.copy()
    painted[area] = by_label[labels[area]]
    # The cells of the nearest-colour map meet in hard steps; soften them only.
    softened = cv2.GaussianBlur(painted, (0, 0), 0.7)
    painted[area] = softened[area]
    return painted


def _learned_fill(rgb: np.ndarray, source: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Texture for a photographed object, learned from that object alone."""
    if int(area.sum()) < LEARNED_FILL_MIN or not source.any():
        return _continue_colours(rgb, source, area)
    # Everything that is not this object is unknown, so nothing else can leak in.
    return _keep_colour(engine.inpaint(rgb, ~source), _smooth_fill(rgb, source), area)


def _extend_under(
    rgba: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    upper: np.ndarray,
    radius: int,
    flat: bool = False,
    surface: bool = False,
) -> np.ndarray:
    """Make a layer whole where the layers above hide it.

    Closes the seam along upper layers' soft edges, and paints enclosed holes
    so that moving the top element does not expose a hole in this one.
    """
    rgba = rgba.copy()
    box = _bbox(mask)
    grow = max(2, radius)
    seam = _dilate(mask, grow) & _dilate(upper, grow)
    hull = _occluded_hull(mask, upper)
    interior = (_fill_interior(mask | seam) | hull) & upper & ~mask
    pure = mask
    if flat:
        reach = max(4, int(max(box[2] - box[0], box[3] - box[1]) * (CLOSING_REACH if surface else FIGURE_REACH)))
        if not hull.any():
            # Closing bridges what covers the shape without growing it elsewhere.
            closed = ~_dilate(~_dilate(mask, reach), reach)
            interior |= closed & upper & ~mask
    if surface:
        # Everything enclosed, including the hairline left between two outlines.
        interior = (interior | _fill_interior(mask | interior)) & ~mask
        # Colour from the object on top bleeds a little past its outline.
        near_upper = _dilate(upper, FRINGE)
        if (mask & ~near_upper).any():
            pure = mask & ~near_upper
            interior |= mask & near_upper

    if interior.any():
        # Filled from this layer's own pixels only, never from what lies on top.
        if not flat:
            # In a photo the object on top shades what it rests on: learn from beyond that.
            shaded = _dilate(upper, max(3, int(max(box[2] - box[0], box[3] - box[1]) * CONTACT_SHADE)))
            if (mask & ~shaded).sum() >= CONTACT_KEEP * mask.sum():
                pure = mask & ~shaded
        painted = _continue_colours(rgb, pure, interior) if flat else _learned_fill(rgb, pure, interior)
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
