"""Redraws the part of an object that other objects hide.

A cut-out ends where something stood in front of it. Stable Diffusion has
seen whole cups, people and chairs, so it can draw the missing part: the
object goes alone onto a plain backdrop, the area that hid it is left open,
and the model is asked for the whole object. SAM then outlines the object in
the result, and only what it finds inside the open area is kept. The pixels
that were visible in the original are never touched.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .models import engine

log = logging.getLogger("prolayer")

WORK_SIDE = 512
MIN_SIDE = 256
# How far past the object's hull the hidden part may reach, as a share of its size.
REACH = 0.12
PAD = 0.1
# Too little hidden to be worth drawing: the seam fill already covers it.
MIN_HIDDEN_SHARE = 0.02
MIN_HIDDEN_PIXELS = 256
MIN_FOUND_SHARE = 0.01
# SAM must find again nearly all of what was visible, or its outline is not trusted.
MIN_VISIBLE_KEPT = 0.85
BACKDROP_DISTANCE = 22.0
# A hull explains a shape when what is hidden accounts for nearly all the
# difference between the two: plates, balls and boxes, but not a person.
HULL_EXPLAINS = 0.9
# With a first guess painted in, the model starts from it rather than from noise.
GUESS_STRENGTH = 0.92
# Where the shape is sure there is nothing to imagine: asked to draw under a cup,
# the model draws another object on the saucer. Plain filling does better there.
SURE_SHARE = 0.8
ELLIPSE_MIN_POINTS = 40
ELLIPSE_MIN_FREE = 0.35
ELLIPSE_FIT = 0.03
ELLIPSE_HOLDS = 0.95
RECTANGLE_MIN_POINTS = 64
RECTANGLE_MIN_MISSING = 0.01
RECTANGLE_OPEN = 0.03
RECTANGLE_EDGE = 0.6
# A concave shape likely carries on across a bay this wide, as a share of its size.
BAY = 0.2
# What stands in front shades the object next to it: the guess takes its colours from beyond.
SHADE = 0.06
SHADE_KEEP = 0.5
RIM = 4
GAP = 3
BACKDROPS = (("white", (255, 255, 255)), ("grey", (128, 128, 128)), ("dark", (36, 36, 36)))
AVOID = "cropped, cut off, text, watermark, frame, border, extra objects, clutter, shadow, low quality, blurry"
STYLES = {
    "photo": "photo of {what}, whole object, complete, isolated on a plain {backdrop} background, studio lighting",
    "flat": "{what}, flat vector illustration, simple cartoon style, whole, complete, isolated on a plain {backdrop} background",
}
UNKNOWN = {"photo": "an object", "flat": "a cartoon figure"}


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _touching(parts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The connected pieces of `parts` that touch `mask`."""
    count, labels = cv2.connectedComponents(parts.astype(np.uint8), connectivity=8)
    touched = np.zeros(count, bool)
    touched[np.unique(labels[_dilate(mask, 2) & parts])] = True
    touched[0] = False
    return touched[labels]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def fit_ellipse(mask: np.ndarray, occluders: np.ndarray) -> np.ndarray | None:
    """The ellipse a plate, wheel or ball would complete to, if the part of
    its outline that nothing hides lies on one."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)[:, 0, :]
    height, width = mask.shape
    near = _dilate(occluders, GAP + 1)
    xs, ys = outline[:, 0], outline[:, 1]
    # The frame cuts shapes off too: its edge says nothing about the outline.
    free = ~near[ys, xs] & (xs > 0) & (ys > 0) & (xs < width - 1) & (ys < height - 1)
    if free.sum() < max(ELLIPSE_MIN_POINTS, ELLIPSE_MIN_FREE * len(outline)):
        return None
    points = outline[free].astype(np.float32)
    (cx, cy), (major, minor), angle = cv2.fitEllipse(points)
    if not (np.isfinite([cx, cy, major, minor]).all() and minor > 4 and major < 2 * max(height, width)):
        return None
    # Distance of each free point from the ellipse, relative to its size.
    theta = np.deg2rad(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = (dx * np.cos(theta) + dy * np.sin(theta)) / (major / 2)
    v = (-dx * np.sin(theta) + dy * np.cos(theta)) / (minor / 2)
    if float(np.abs(np.hypot(u, v) - 1).mean()) > ELLIPSE_FIT:
        return None
    shape = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(shape, ((cx, cy), (major, minor), angle), 1, -1)
    shape = shape > 0
    # It must hold the object, and the object must fill what is not hidden.
    if (mask & shape).sum() < ELLIPSE_HOLDS * mask.sum():
        return None
    if (shape & ~mask & ~occluders).sum() > (1 - ELLIPSE_HOLDS) * shape.sum():
        return None
    return shape


def fit_rectangle(mask: np.ndarray, occluders: np.ndarray) -> np.ndarray | None:
    """The rectangle a card or panel would complete to, if all that is missing
    from it lies under something else."""
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < RECTANGLE_MIN_POINTS:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(points))
    shape = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(shape, np.round(box).astype(np.int32), 1)
    shape = shape > 0
    missing = shape & ~mask
    if missing.sum() < RECTANGLE_MIN_MISSING * shape.sum():
        return None
    # Rounded corners and soft edges leave a little of the rectangle open.
    if (missing & ~occluders).sum() > RECTANGLE_OPEN * shape.sum():
        return None
    # A card runs along most of its rectangle's edge; a figure only touches its box here and there.
    edge = shape & ~cv2.erode(shape.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if (edge & mask).sum() < RECTANGLE_EDGE * edge.sum():
        return None
    return shape


def known_shape(mask: np.ndarray, occluders: np.ndarray, hull_explains: float | None) -> np.ndarray:
    """What the object surely covers, hidden parts included: an ellipse, a
    rectangle, or its hull when the hull explains its shape (None: never the hull)."""
    shape = np.zeros(mask.shape, bool)
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or not occluders.any():
        return shape
    hull_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(points), 1)
    hull = hull_mask > 0
    for fitted in (fit_ellipse(mask, occluders), fit_rectangle(mask, occluders)):
        if fitted is not None:
            shape |= fitted
    if not shape.any() and hull_explains is not None and (mask | (hull & occluders)).sum() >= hull_explains * hull.sum():
        shape |= hull
    return shape


def hidden_area(mask: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Where the rest of the object may lie, where it surely does, and where it likely does.

    It may lie under whatever touches it, not far from its hull. It surely
    lies in the holes it encloses, and inside its hull when the hull explains
    its shape.
    """
    occluders = _touching(upper, mask)
    if not occluders.any():
        return None
    x0, y0, x1, y1 = _bbox(mask)
    reach = max(4, int(max(x1 - x0, y1 - y0) * REACH))
    hull_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8))), 1)
    hull = hull_mask > 0
    # The hairline between two outlines belongs to neither: it is hidden too.
    area = (occluders | (_dilate(occluders, GAP) & _dilate(mask, GAP))) & ~mask & _dilate(hull, reach)
    if area.sum() < max(MIN_HIDDEN_PIXELS, MIN_HIDDEN_SHARE * mask.sum()):
        return None
    sure = _fill_holes(mask | (area & _dilate(mask, GAP))) & area
    sure |= known_shape(mask, occluders, HULL_EXPLAINS) & area
    # Likely, though not sure: the bays that closing the shape bridges.
    bay = max(4, int(max(x1 - x0, y1 - y0) * BAY))
    pad = bay + 1
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bay + 1, 2 * bay + 1))
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[pad:-pad, pad:-pad] > 0
    likely = sure | (closed & area)
    return area, sure, likely


def _backdrop(rgb: np.ndarray, mask: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """The plain colour least like the object, so its outline stays clear."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
    sample = lab[:: max(1, len(lab) // 4000)]
    best, best_distance = BACKDROPS[0], -1.0
    for name, colour in BACKDROPS:
        target = cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        # The tenth percentile: a few pixels near the backdrop colour do not matter.
        distance = float(np.percentile(np.linalg.norm(sample - target, axis=1), 10))
        if distance > best_distance:
            best, best_distance = (name, colour), distance
    return best


def _work_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Pad the box, and widen its short side so the shape is at most 2:1."""
    x0, y0, x1, y1 = box
    pad = int(max(x1 - x0, y1 - y0) * PAD) + 4
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w, h = x1 - x0, y1 - y0
    if w < h / 2:
        x0 -= (h // 2 - w) // 2
        x1 = x0 + h // 2
    if h < w / 2:
        y0 -= (w // 2 - h) // 2
        y1 = y0 + w // 2
    return x0, y0, x1, y1


def hidden_size(mask: np.ndarray, upper: np.ndarray) -> int:
    """How much of the object the model would be asked to draw; 0 for none."""
    areas = hidden_area(mask, upper)
    if areas is None or areas[1].sum() >= SURE_SHARE * areas[0].sum():
        return 0
    return int(areas[0].sum())


def complete(
    rgb: np.ndarray,
    mask: np.ndarray,
    upper: np.ndarray,
    what: str = "",
    style: str = "photo",
    in_front: str = "",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The hidden part of the object in `mask`, as (where, picture) at full size.

    `upper` is everything in front of it, and `in_front` names what must not
    be drawn again. Returns None when nothing worth drawing is hidden, or when
    the drawing cannot be told from the backdrop.
    """
    from .pipeline import _smooth_fill

    height, width = mask.shape
    areas = hidden_area(mask, upper)
    if areas is None:
        return None
    area, sure, likely = areas
    if sure.sum() >= SURE_SHARE * area.sum():
        return None
    backdrop_name, backdrop = _backdrop(rgb, mask)

    # The work canvas may reach past the picture: the backdrop fills the rest.
    x0, y0, x1, y1 = _work_box(_bbox(mask | area), width, height)
    w, h = x1 - x0, y1 - y0
    canvas = np.empty((h, w, 3), np.uint8)
    canvas[...] = backdrop
    seen = np.zeros((h, w), bool)
    open_area = np.zeros((h, w), bool)
    guess = np.zeros((h, w), bool)
    forced = np.zeros((h, w), bool)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    inside = (slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0))
    source = (slice(sy0, sy1), slice(sx0, sx1))
    seen[inside] = mask[source]
    open_area[inside] = area[source]
    guess[inside] = likely[source]
    forced[inside] = sure[source]
    canvas[seen] = rgb[source][mask[source]]
    if guess.any():
        # A first guess: the object's own colours carried into where it likely is.
        unshaded = seen & ~_dilate(open_area, max(3, int(max(w, h) * SHADE)))
        known = unshaded if unshaded.sum() >= SHADE_KEEP * seen.sum() else seen
        canvas[guess] = _smooth_fill(canvas, known)[guess]

    scale = WORK_SIDE / max(w, h)
    size = (max(MIN_SIDE, round(w * scale / 8) * 8), max(MIN_SIDE, round(h * scale / 8) * 8))
    small = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    small_seen = cv2.resize(seen.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    small_open = cv2.resize(open_area.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    small_guess = cv2.resize(guess.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    small_forced = cv2.resize(forced.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    # The rim of the cut-out still carries a little of what stood in front.
    hole = _dilate(small_open, RIM)

    prompt = STYLES[style].format(what=what or UNKNOWN[style], backdrop=backdrop_name)
    avoid = f"{in_front}, {AVOID}" if in_front else AVOID
    drawn = engine.paint(small, hole, prompt, avoid, GUESS_STRENGTH if small_guess.any() else 1.0, seed)

    # Which of the drawn pixels are the object: SAM's outline, and unlike the backdrop.
    # Stray threads of the cut-out neither guide SAM nor anchor what was drawn.
    core = cv2.morphologyEx(small_seen.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(core & ~_dilate(~core, 4) & ~hole)
    if len(xs) < 8:
        ys, xs = np.nonzero(small_seen)
    picks = np.linspace(0, len(xs) - 1, 6).astype(int)
    points = [(float(xs[i]), float(ys[i])) for i in picks]
    whole = small_seen | small_open
    bx0, by0, bx1, by1 = _bbox(whole)
    outline, _ = engine.predict_mask(
        drawn, engine.embed(drawn), points, [1] * len(points), (float(bx0), float(by0), float(bx1), float(by1))
    )
    difference = np.linalg.norm(
        cv2.cvtColor(drawn, cv2.COLOR_RGB2LAB).astype(np.float32)
        - cv2.cvtColor(np.uint8([[backdrop]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32),
        axis=2,
    )
    unlike = difference > BACKDROP_DISTANCE
    kept = float((outline & small_seen).sum()) / max(int(small_seen.sum()), 1)
    found = small_open & ((unlike & (outline if kept >= MIN_VISIBLE_KEPT else True)) | small_forced)
    found = cv2.morphologyEx(found.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    # The hairline next to the cut-out always closes; the rest must hang on to it.
    found |= small_open & _dilate(core, GAP + 1)
    found = _touching(found, core)
    # A highlight the colour of the backdrop is still part of the object around it.
    found |= _fill_holes(found | small_seen) & small_open
    log.debug("complete %r area=%d kept=%.2f found=%d", what, int(small_open.sum()), kept, int(found.sum()))
    if found.sum() < MIN_FOUND_SHARE * small_seen.sum():
        return None

    back = cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC
    drawn_full = cv2.resize(drawn, (w, h), interpolation=back)
    found_full = cv2.resize(found.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    where = np.zeros((height, width), bool)
    picture = rgb.copy()
    # Only under what stood in front: everything that shows stays as it was.
    where[source] = found_full[inside] & (area & upper)[source]
    picture[source][where[source]] = drawn_full[inside][where[source]]
    return where, picture
