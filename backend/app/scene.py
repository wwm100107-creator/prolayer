"""Whole objects in a photograph: what they are, and which one is in front.

Mask2Former knows 133 kinds of things and surfaces, so a person comes out as
one person rather than a face, a sleeve and a collar. Its outlines are
coarse, so they are snapped to SAM's. Objects it has no name for, like a
saucer on a table or a helmet held in front of a suit, stand out in depth:
Depth Anything shows them nearer than what surrounds them. Depth also gives
the stacking order, read where two objects meet.
"""
from __future__ import annotations

import heapq
import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .models import engine

log = logging.getLogger("prolayer")

NAMES = {
    "person": "Người", "bicycle": "Xe đạp", "car": "Ô tô", "motorcycle": "Xe máy", "airplane": "Máy bay",
    "bus": "Xe buýt", "train": "Tàu hỏa", "truck": "Xe tải", "boat": "Thuyền", "traffic light": "Đèn giao thông",
    "fire hydrant": "Trụ cứu hỏa", "stop sign": "Biển dừng", "parking meter": "Đồng hồ đỗ xe", "bench": "Ghế dài",
    "bird": "Chim", "cat": "Mèo", "dog": "Chó", "horse": "Ngựa", "sheep": "Cừu", "cow": "Bò", "elephant": "Voi",
    "bear": "Gấu", "zebra": "Ngựa vằn", "giraffe": "Hươu cao cổ", "backpack": "Ba lô", "umbrella": "Ô",
    "handbag": "Túi xách", "tie": "Cà vạt", "suitcase": "Va li", "frisbee": "Đĩa ném", "skis": "Ván trượt tuyết",
    "snowboard": "Ván trượt tuyết", "sports ball": "Quả bóng", "kite": "Diều", "baseball bat": "Gậy bóng chày",
    "baseball glove": "Găng bóng chày", "skateboard": "Ván trượt", "surfboard": "Ván lướt sóng",
    "tennis racket": "Vợt tennis", "bottle": "Chai", "wine glass": "Ly rượu", "cup": "Cốc", "fork": "Nĩa",
    "knife": "Dao", "spoon": "Thìa", "bowl": "Bát", "banana": "Chuối", "apple": "Táo", "sandwich": "Bánh mì kẹp",
    "orange": "Cam", "broccoli": "Bông cải xanh", "carrot": "Cà rốt", "hot dog": "Xúc xích", "pizza": "Pizza",
    "donut": "Bánh vòng", "cake": "Bánh ngọt", "chair": "Ghế", "couch": "Sofa", "potted plant": "Chậu cây",
    "bed": "Giường", "dining table": "Bàn", "toilet": "Bồn cầu", "tv": "Tivi", "laptop": "Laptop",
    "mouse": "Chuột máy tính", "remote": "Điều khiển", "keyboard": "Bàn phím", "cell phone": "Điện thoại",
    "microwave": "Lò vi sóng", "oven": "Lò nướng", "toaster": "Máy nướng bánh", "sink": "Bồn rửa",
    "refrigerator": "Tủ lạnh", "book": "Sách", "clock": "Đồng hồ", "vase": "Bình hoa", "scissors": "Kéo",
    "teddy bear": "Gấu bông", "hair drier": "Máy sấy tóc", "toothbrush": "Bàn chải đánh răng",
    "banner": "Cờ, băng rôn", "blanket": "Chăn", "bridge": "Cầu", "cardboard": "Bìa các tông", "counter": "Quầy",
    "curtain": "Rèm", "door-stuff": "Cửa", "floor-wood": "Sàn gỗ", "flower": "Hoa", "fruit": "Trái cây",
    "gravel": "Sỏi", "house": "Nhà", "light": "Đèn", "mirror-stuff": "Gương", "net": "Lưới", "pillow": "Gối",
    "platform": "Sân ga", "playingfield": "Sân chơi", "railroad": "Đường ray", "river": "Sông", "road": "Đường",
    "roof": "Mái nhà", "sand": "Cát", "sea": "Biển", "shelf": "Kệ", "snow": "Tuyết", "stairs": "Cầu thang",
    "tent": "Lều", "towel": "Khăn", "wall-brick": "Tường gạch", "wall-stone": "Tường đá", "wall-tile": "Tường ốp",
    "wall-wood": "Tường gỗ", "water-other": "Nước", "window-blind": "Rèm cửa sổ", "window-other": "Cửa sổ",
    "tree-merged": "Cây", "fence-merged": "Hàng rào", "ceiling-merged": "Trần nhà", "sky-other-merged": "Bầu trời",
    "cabinet-merged": "Tủ", "table-merged": "Bàn", "floor-other-merged": "Sàn", "pavement-merged": "Vỉa hè",
    "mountain-merged": "Núi", "grass-merged": "Cỏ", "dirt-merged": "Đất", "paper-merged": "Giấy",
    "food-other-merged": "Đồ ăn", "building-other-merged": "Tòa nhà", "rock-merged": "Đá",
    "wall-other-merged": "Tường", "rug-merged": "Thảm",
}

UNNAMED = "Vật thể"
# A Mask2Former outline is trusted only this far from SAM's.
SNAP_REACH = 0.015
SNAP_SHARE = 0.6
# A surface becomes a layer when it stands this much nearer than the backdrop.
SURFACE_STEP = 0.15
SURFACE_MAX_AREA = 0.35
# A thing this large that runs along this much of the frame and holds other
# things is the backdrop, like a table photographed from above with a cup on it.
BACKDROP_THING_AREA = 0.4
BACKDROP_THING_BORDER = 0.5
BACKDROP_HOLDS = 0.5
# Objects Mask2Former has no name for: how much nearer than what surrounds them.
# On the backdrop a saucer stands only a little above its table. Inside a named
# thing a far larger step is needed: a cup's handle is nearer than its side.
EXTRA_STEP = {"coarse": 0.08, "medium": 0.05, "fine": 0.035}
PART_STEP = {"coarse": 0.3, "medium": 0.2, "fine": 0.12}
# Ground and ceiling run along the frame and always look nearer than the far wall.
EXTRA_MAX_FRAME = 0.25
EXTRA_MIN_AREA = {"coarse": 0.01, "medium": 0.003, "fine": 0.0015}
EXTRA_INSIDE = 0.85
EXTRA_MIN_SHARE = 0.02
EXTRA_MAX_SHARE = 0.7
EXTRA_MAX_TAKEN = 0.3
# Most of an object's outline runs along what holds it. A cup's rim is nearer
# than the coffee inside it, but the rim's outline runs along the table.
EXTRA_MIN_HELD = 0.25
RING = 0.01
RING_MIN_PIXELS = 12
# Two objects that meet: the side that is nearer by this much is in front.
FRONT_STEP = 0.02
SAME_DEPTH = 0.08
PART_MIN_AREA = 0.002
PART_INSIDE = 0.8


@dataclass
class Scene:
    labels: np.ndarray  # segment index per pixel
    segments: list[dict]
    depth: np.ndarray  # 0 far, 1 near


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return ~_dilate(~mask, radius)


def understand(rgb: np.ndarray) -> Scene | None:
    segmentation, found = engine.panoptic(rgb)
    if not found:
        return None
    depth = engine.depth(rgb)
    low, high = np.percentile(depth, (2, 98))
    depth = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)

    index = {segment["id"]: i for i, segment in enumerate(found)}
    lookup = np.full(int(segmentation.max()) + 2, -1, np.int32)
    for segment_id, i in index.items():
        lookup[segment_id + 1] = i
    labels = lookup[segmentation + 1]
    # Pixels no class was sure of go to the nearest segment.
    missing = labels < 0
    if missing.any():
        _, nearest = cv2.distanceTransformWithLabels(
            missing.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        by_label = np.full(int(nearest.max()) + 1, -1, np.int32)
        by_label[nearest[~missing]] = labels[~missing]
        labels = np.where(missing, by_label[nearest], labels)
    segments = [
        {**segment, "name": NAMES.get(segment["label"], UNNAMED)} for segment in found
    ]
    return Scene(labels, segments, depth)


def name_flat_layers(scene: Scene | None, masks: list[np.ndarray]) -> dict[int, str]:
    """Names for layers of flat artwork, only where a thing matches closely."""
    if scene is None:
        return {}
    names = {}
    for i, segment in enumerate(scene.segments):
        if not segment["thing"] or segment["score"] < 0.85:
            continue
        region = scene.labels == i
        best, best_iou = None, 0.5
        for mask in masks:
            overlap = int((mask & region).sum())
            iou = overlap / max(int((mask | region).sum()), 1)
            if iou > best_iou:
                best, best_iou = mask, iou
        if best is not None:
            names[id(best)] = segment["name"]
    return names


def _boundary(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    return edge


def _snap(labels: np.ndarray, cells: list[np.ndarray], reach: int) -> np.ndarray:
    """Move Mask2Former's outlines onto SAM's where they are close."""
    band = _dilate(_boundary(labels), reach)
    out = labels.copy()
    for cell in cells:
        values = labels[cell]
        if not values.size:
            continue
        counts = np.bincount(values[values >= 0])
        best = int(counts.argmax())
        if counts[best] >= SNAP_SHARE * values.size:
            out[cell & band] = best
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def _step(depth: np.ndarray, mask: np.ndarray, around: np.ndarray, width: int) -> tuple[float, float] | None:
    """How much nearer the outline's inside is than the outside next to it, and
    what share of the outside belongs to `around`."""
    outline = _fill_holes(mask)
    inner = mask & ~_erode(outline, width)
    ring = _dilate(outline, width) & ~outline
    # The frame's edge holds nothing, so it counts for neither side.
    ring[:width] = ring[-width:] = False
    ring[:, :width] = ring[:, -width:] = False
    outer = ring & around
    if inner.sum() < RING_MIN_PIXELS or outer.sum() < RING_MIN_PIXELS:
        return None
    held = float(outer.sum()) / float(ring.sum())
    return float(np.median(depth[inner]) - np.median(depth[outer])), held


def _order(depth: np.ndarray, layers: list[np.ndarray], width: int) -> list[int]:
    """Bottom-to-top order: where two layers meet, the nearer side is in front."""
    count = len(layers)
    rings = [_dilate(mask, width) & ~mask for mask in layers]
    edges = []
    for i in range(count):
        for j in range(i + 1, count):
            j_side = rings[i] & layers[j]
            i_side = rings[j] & layers[i]
            if j_side.sum() < RING_MIN_PIXELS or i_side.sum() < RING_MIN_PIXELS:
                continue
            difference = float(np.median(depth[i_side]) - np.median(depth[j_side]))
            if abs(difference) >= FRONT_STEP:
                below, above = (j, i) if difference > 0 else (i, j)
                edges.append((abs(difference), below, above))
    # Strongest evidence first; a contradiction of stronger evidence is dropped.
    above_of: list[set[int]] = [set() for _ in range(count)]

    def reaches(start: int, goal: int) -> bool:
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            for nxt in above_of[node] - seen:
                seen.add(nxt)
                stack.append(nxt)
        return False

    for _, below, above in sorted(edges, reverse=True):
        if not reaches(above, below):
            above_of[below].add(above)
    waiting = [0] * count
    for node in range(count):
        for above in above_of[node]:
            waiting[above] += 1
    medians = [float(np.median(depth[mask])) for mask in layers]
    ready = [(medians[i], i) for i in range(count) if waiting[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for above in above_of[node]:
            waiting[above] -= 1
            if waiting[above] == 0:
                heapq.heappush(ready, (medians[above], above))
    return order


def _plain(label: str) -> str:
    """A COCO label as plain words: 'wall-other-merged' is a wall."""
    for suffix in ("-merged", "-other", "-stuff"):
        label = label.replace(suffix, "")
    return "a " + label.replace("-", " ")


def what_is(scene: Scene | None, mask: np.ndarray) -> str:
    """What a layer of flat artwork shows, when one thing covers most of it."""
    if scene is None:
        return ""
    counts = np.bincount(scene.labels[mask], minlength=len(scene.segments))
    best = int(counts.argmax())
    segment = scene.segments[best]
    if segment["thing"] and counts[best] >= 0.5 * mask.sum() and counts[best] >= 0.3 * (scene.labels == best).sum():
        return _plain(segment["label"])
    return ""


def _numbered(names: list[str]) -> list[str]:
    totals = Counter(names)
    seen: Counter = Counter()
    out = []
    for name in names:
        seen[name] += 1
        out.append(f"{name} {seen[name]}" if totals[name] > 1 else name)
    return out


def photo_layers(
    scene: Scene,
    objects: list[np.ndarray],
    cells: list[np.ndarray],
    detail: str,
    max_layers: int,
) -> tuple[list[np.ndarray], list[str], list[str | None]]:
    """Layer masks bottom to top, their names, and what each one is in
    English for the model that redraws hidden parts (None: not a whole object).

    `objects` are SAM's whole-object masks, `cells` its finest pieces.
    Whatever is left belongs to the background.
    """
    height, width = scene.labels.shape
    pixels = height * width
    side = max(height, width)
    ring = max(2, int(side * RING))
    labels = _snap(scene.labels, cells, max(3, int(side * SNAP_REACH)))
    depth = scene.depth
    segments = scene.segments

    # The backdrop is the largest surface; with none, the farthest segment.
    areas = np.bincount(labels.ravel(), minlength=len(segments))
    medians = [float(np.median(depth[labels == i])) if areas[i] else 1.0 for i in range(len(segments))]
    frame = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    on_frame = np.bincount(frame, minlength=len(segments)) / frame.size

    def holds_things(i: int) -> bool:
        for j, other in enumerate(segments):
            if j != i and other["thing"] and areas[j]:
                outline = _dilate(labels == j, ring) & (labels != j)
                if (labels[outline] == i).mean() >= BACKDROP_HOLDS:
                    return True
        return False

    surfaces = [
        i
        for i, segment in enumerate(segments)
        if areas[i]
        and (
            not segment["thing"]
            or (
                areas[i] >= BACKDROP_THING_AREA * pixels
                and on_frame[i] >= BACKDROP_THING_BORDER
                and holds_things(i)
            )
        )
    ]
    if surfaces:
        backdrop = max(surfaces, key=lambda i: areas[i])
    else:
        backdrop = min((i for i in range(len(segments)) if areas[i]), key=lambda i: medians[i])
    layer_ids = []
    for i, segment in enumerate(segments):
        if not areas[i] or i == backdrop:
            continue
        if segment["thing"] and i not in surfaces:
            layer_ids.append(i)
        elif areas[i] <= SURFACE_MAX_AREA * pixels and medians[i] - medians[backdrop] >= SURFACE_STEP:
            layer_ids.append(i)

    masks = [labels == i for i in layer_ids]
    names = [segments[i]["name"] for i in layer_ids]
    whats: list[str | None] = [_plain(segments[i]["label"]) for i in layer_ids]
    background = ~np.logical_or.reduce(masks) if masks else np.ones((height, width), bool)

    # Objects without a class name, standing out from what holds them.
    taken = np.zeros((height, width), bool)
    for candidate in sorted(objects, key=lambda mask: -int(mask.sum())):
        area = int(candidate.sum())
        if area < EXTRA_MIN_AREA[detail] * pixels or (candidate & taken).sum() > EXTRA_MAX_TAKEN * area:
            continue
        holders = [background] + masks
        shares = [int((candidate & holder).sum()) for holder in holders]
        best = int(np.argmax(shares))
        holder = holders[best]
        if shares[best] < EXTRA_INSIDE * area:
            continue
        holder_area = int(holder.sum())
        if not EXTRA_MIN_SHARE * holder_area <= area <= EXTRA_MAX_SHARE * holder_area:
            continue
        part = candidate & holder
        measured = _step(depth, part, holder, ring)
        if measured is None:
            continue
        step, held = measured
        frame = np.concatenate([part[0], part[-1], part[:, 0], part[:, -1]]).sum() / (height + width)
        log.debug("extra area=%.4f holder=%d step=%.3f held=%.2f frame=%.2f", area / pixels, best, step, held, frame)
        if held < EXTRA_MIN_HELD:
            continue
        if best == 0 and (step < EXTRA_STEP[detail] or frame > EXTRA_MAX_FRAME):
            continue
        if best > 0 and step < PART_STEP[detail]:
            continue
        if best == 0:
            background &= ~part
        else:
            masks[best - 1] = masks[best - 1] & ~part
        masks.append(part)
        names.append(UNNAMED)
        whats.append("")
        taken |= part

    # SAM may see a rim and the dome inside it as two objects. Unnamed objects
    # that meet at the same depth are one.
    merged = True
    while merged:
        merged = False
        unnamed = [i for i, name in enumerate(names) if name == UNNAMED]
        for a in unnamed:
            for b in unnamed:
                if a >= b:
                    continue
                a_side = _dilate(masks[b], ring) & masks[a]
                b_side = _dilate(masks[a], ring) & masks[b]
                if a_side.sum() < RING_MIN_PIXELS or b_side.sum() < RING_MIN_PIXELS:
                    continue
                if abs(float(np.median(depth[a_side]) - np.median(depth[b_side]))) < SAME_DEPTH:
                    masks[a] = masks[a] | masks[b]
                    del masks[b], names[b], whats[b]
                    merged = True
                    break
            if merged:
                break

    # Fine detail: each object also splits into the pieces SAM sees in it.
    if detail == "fine":
        split_masks, split_names, split_whats = [], [], []
        for mask, name, what in zip(masks, names, whats):
            pieces = []
            rest = mask.copy()
            for cell in cells:
                piece = cell & rest
                piece_area = int(piece.sum())
                if piece_area >= PART_MIN_AREA * pixels and piece_area >= PART_INSIDE * int(cell.sum()):
                    pieces.append(piece)
                    rest &= ~piece
            if len(pieces) < 2:
                split_masks.append(mask)
                split_names.append(name)
                split_whats.append(what)
                continue
            # What no piece claimed stays with the largest piece.
            pieces.sort(key=lambda piece: -int(piece.sum()))
            pieces[0] = pieces[0] | rest
            split_masks += pieces
            split_names += [f"{name} – phần" for _ in pieces]
            # A piece of an object has no hidden part of its own to redraw.
            split_whats += [None for _ in pieces]
        masks, names, whats = split_masks, split_names, split_whats

    keep = [i for i, mask in enumerate(masks) if mask.any()]
    if len(keep) > max_layers:
        keep = sorted(keep, key=lambda i: -int(masks[i].sum()))[:max_layers]
    masks = [masks[i] for i in keep]
    names = [names[i] for i in keep]
    whats = [whats[i] for i in keep]
    order = _order(depth, masks, ring)
    return [masks[i] for i in order], _numbered([names[i] for i in order]), [whats[i] for i in order]
