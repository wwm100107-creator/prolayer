"""Loads SAM 2, ViTMatte and LaMa, and serialises access to the GPU."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
    Sam2ImageProcessor,
    Sam2Model,
    VitMatteForImageMatting,
    VitMatteImageProcessor,
)

log = logging.getLogger("prolayer")

SAM2_ID = os.environ.get("PROLAYER_SAM2", "facebook/sam2.1-hiera-large")
VITMATTE_ID = os.environ.get("PROLAYER_VITMATTE", "hustvl/vitmatte-base-composition-1k")
PANOPTIC_ID = os.environ.get("PROLAYER_PANOPTIC", "facebook/mask2former-swin-large-coco-panoptic")
DEPTH_ID = os.environ.get("PROLAYER_DEPTH", "depth-anything/Depth-Anything-V2-Small-hf")
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
LAMA_PATH = Path.home() / ".cache" / "prolayer" / "models" / "big-lama.pt"

SAM_SIZE = 1024
MATTE_MAX_SIDE = 1280
INPAINT_MAX_SIDE = 1536
MIN_RETRY_SIDE = 512
# Mask2Former was evaluated at a shortest side of 800.
PANOPTIC_SIZE = {"shortest_edge": 800, "longest_edge": 1333}
PANOPTIC_THRESHOLD = 0.5
GRID_SIDE = 32
GRID_BATCH = 64


def _download_lama() -> Path:
    if LAMA_PATH.exists():
        return LAMA_PATH
    LAMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = LAMA_PATH.with_suffix(".part")
    log.info("Downloading LaMa weights to %s", LAMA_PATH)
    digest = hashlib.md5()
    with urllib.request.urlopen(LAMA_URL) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != LAMA_MD5:
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for {LAMA_URL}: got {digest.hexdigest()}")
    partial.replace(LAMA_PATH)
    return LAMA_PATH


class Engine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # One inference at a time: 8GB of VRAM does not fit concurrent crops.
        self.gpu = threading.RLock()
        self._sam = None
        self._sam_processor = None
        self._matte = None
        self._matte_processor = None
        self._lama = None
        self._ocr = None
        self._panoptic = None
        self._panoptic_processor = None
        self._depth = None
        self._depth_processor = None
        self.error: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def preload(self) -> None:
        try:
            self._load_sam()
            self._load_matte()
            self._load_lama()
            self._load_panoptic()
            self._load_depth()
            self._load_ocr()
        except Exception as error:  # surfaced to the UI through /api/health
            log.exception("Model preload failed")
            self.error = str(error)

    def status(self) -> dict:
        return {
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ready": all(m is not None for m in (self._sam, self._matte, self._lama)),
            "error": self.error,
        }

    def _load_sam(self):
        with self.gpu:
            if self._sam is None:
                self._sam_processor = Sam2ImageProcessor.from_pretrained(SAM2_ID)
                self._sam = Sam2Model.from_pretrained(SAM2_ID).to(self.device).eval()
        return self._sam

    def _load_matte(self):
        with self.gpu:
            if self._matte is None:
                self._matte_processor = VitMatteImageProcessor.from_pretrained(VITMATTE_ID)
                self._matte = VitMatteForImageMatting.from_pretrained(VITMATTE_ID).to(self.device).eval()
        return self._matte

    def _load_lama(self):
        with self.gpu:
            if self._lama is None:
                self._lama = torch.jit.load(str(_download_lama()), map_location=self.device).eval()
        return self._lama

    def _load_panoptic(self):
        with self.gpu:
            if self._panoptic is None:
                self._panoptic_processor = AutoImageProcessor.from_pretrained(PANOPTIC_ID)
                model = Mask2FormerForUniversalSegmentation.from_pretrained(PANOPTIC_ID).to(self.device).eval()
                self._panoptic = model.half() if self.device.type == "cuda" else model
        return self._panoptic

    def _load_depth(self):
        with self.gpu:
            if self._depth is None:
                self._depth_processor = AutoImageProcessor.from_pretrained(DEPTH_ID)
                self._depth = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(self.device).eval()
        return self._depth

    def _load_ocr(self):
        with self.gpu:
            if self._ocr is None:
                import easyocr  # slow import, and only text extraction needs it

                self._ocr = easyocr.Reader(["vi", "en"], gpu=self.device.type == "cuda", verbose=False)
        return self._ocr

    def read_text(self, rgb: np.ndarray) -> list:
        """EasyOCR lines as (four corners, text, confidence)."""
        reader = self._load_ocr()
        with self.gpu:
            return reader.readtext(rgb, paragraph=False)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda")

    def _to_gpu(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.to(self.device) for tensor in embeddings]

    # ------------------------------------------------------------------
    # SAM 2
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def embed(self, rgb: np.ndarray) -> list[torch.Tensor]:
        """Image features, kept on the CPU so idle sessions cost no VRAM."""
        sam = self._load_sam()
        with self.gpu, self._autocast():
            pixels = self._sam_processor(images=rgb, return_tensors="pt").pixel_values.to(self.device)
            return [tensor.float().cpu() for tensor in sam.get_image_embeddings(pixels)]

    @torch.inference_mode()
    def predict_mask(
        self,
        rgb: np.ndarray,
        embeddings: list[torch.Tensor],
        points: Sequence[tuple[float, float]],
        labels: Sequence[int],
        box: tuple[float, float, float, float] | None,
        level: int | None = None,
    ) -> tuple[np.ndarray, float]:
        sam = self._load_sam()
        height, width = rgb.shape[:2]
        # SAM 2 squashes the image to a square, so x and y scale independently.
        scale = torch.tensor([SAM_SIZE / width, SAM_SIZE / height], device=self.device)
        kwargs: dict = {}
        if len(points):
            xy = torch.tensor(points, dtype=torch.float32, device=self.device) * scale
            kwargs["input_points"] = xy[None, None]
            kwargs["input_labels"] = torch.tensor(labels, device=self.device)[None, None]
        if box is not None:
            kwargs["input_boxes"] = (
                torch.tensor(box, dtype=torch.float32, device=self.device) * scale.repeat(2)
            )[None, None]
        # Only a lone click is ambiguous enough to need three candidates.
        multimask = len(points) == 1 and box is None
        with self.gpu, self._autocast():
            output = sam(image_embeddings=self._to_gpu(embeddings), multimask_output=multimask, **kwargs)
        logits = output.pred_masks[0, 0].float().cpu().numpy()
        scores = output.iou_scores[0, 0].float().cpu().numpy()
        if level is None or not multimask:
            index = int(np.argmax(scores))
        else:
            # Ordered by size so that level 0 is always the smallest part.
            index = int(np.argsort((logits > 0).sum(axis=(1, 2)))[level])
        mask = cv2.resize(logits[index], (width, height), interpolation=cv2.INTER_LINEAR) > 0
        return mask, float(scores[index])

    @torch.inference_mode()
    def auto_candidates(
        self,
        rgb: np.ndarray,
        embeddings: list[torch.Tensor],
        progress: Callable[[float], None] = lambda value: None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prompt a point grid; returns (N x 256 x 256 logits, N scores, N point ids)."""
        step = SAM_SIZE / GRID_SIDE
        axis = torch.arange(GRID_SIDE, dtype=torch.float32, device=self.device) * step + step / 2
        grid = torch.cartesian_prod(axis, axis)
        # Cheap first cut; most of the ~3000 masks are junk or duplicates.
        return self._decode_points(embeddings, grid, 0.6, progress)

    @torch.inference_mode()
    def candidates_at(
        self, rgb: np.ndarray, embeddings: list[torch.Tensor], points: Sequence[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Three candidate masks for each (x, y) image point, in the same format."""
        height, width = rgb.shape[:2]
        scale = torch.tensor([SAM_SIZE / width, SAM_SIZE / height], device=self.device)
        xy = torch.tensor(points, dtype=torch.float32, device=self.device) * scale
        return self._decode_points(embeddings, xy, 0.0)

    def _decode_points(
        self,
        embeddings: list[torch.Tensor],
        grid: torch.Tensor,
        min_score: float,
        progress: Callable[[float], None] = lambda value: None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sam = self._load_sam()
        kept_logits, kept_scores, kept_points = [], [], []
        total = len(grid)
        features = self._to_gpu(embeddings)
        for start in range(0, total, GRID_BATCH):
            batch = grid[start : start + GRID_BATCH]
            with self.gpu, self._autocast():
                output = sam(
                    image_embeddings=features,
                    input_points=batch[None, :, None],
                    input_labels=torch.ones((1, len(batch), 1), dtype=torch.int64, device=self.device),
                    multimask_output=True,
                )
            logits = output.pred_masks[0].flatten(0, 1).float()
            scores = output.iou_scores[0].flatten().float()
            point_ids = torch.arange(start, start + len(batch), device=self.device).repeat_interleave(3)
            keep = scores >= min_score
            kept_logits.append(logits[keep].cpu().numpy())
            kept_scores.append(scores[keep].cpu().numpy())
            kept_points.append(point_ids[keep].cpu().numpy())
            progress(min(1.0, (start + len(batch)) / total))
        return np.concatenate(kept_logits), np.concatenate(kept_scores), np.concatenate(kept_points)

    # ------------------------------------------------------------------
    # Scene understanding
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def panoptic(self, rgb: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Segment ids per pixel (-1 where nothing was found) and, per id,
        its COCO label, whether it is a countable thing, and the confidence."""
        model = self._load_panoptic()
        height, width = rgb.shape[:2]
        with self.gpu:
            inputs = self._panoptic_processor(images=rgb, size=PANOPTIC_SIZE, return_tensors="pt").to(self.device)
            outputs = model(pixel_values=inputs.pixel_values.to(model.dtype), pixel_mask=inputs.pixel_mask)
            outputs.class_queries_logits = outputs.class_queries_logits.float()
            outputs.masks_queries_logits = outputs.masks_queries_logits.float()
            result = self._panoptic_processor.post_process_panoptic_segmentation(
                outputs, threshold=PANOPTIC_THRESHOLD, target_sizes=[(height, width)], label_ids_to_fuse=set()
            )[0]
        segmentation = result["segmentation"].cpu().numpy().astype(np.int32)
        segments = [
            {
                "id": int(info["id"]),
                "label": model.config.id2label[info["label_id"]],
                # COCO panoptic lists the 80 things before the 53 stuff classes.
                "thing": info["label_id"] < 80,
                "score": float(info["score"]),
            }
            for info in result["segments_info"]
        ]
        return segmentation, segments

    @torch.inference_mode()
    def depth(self, rgb: np.ndarray) -> np.ndarray:
        """Relative inverse depth: larger is nearer."""
        model = self._load_depth()
        height, width = rgb.shape[:2]
        with self.gpu:
            inputs = self._depth_processor(images=rgb, return_tensors="pt").to(self.device)
            predicted = model(**inputs).predicted_depth[0].float().cpu().numpy()
        return cv2.resize(predicted, (width, height), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # ViTMatte
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def matte(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Float alpha in [0, 1] at the size of `rgb`."""
        matte = self._load_matte()
        height, width = rgb.shape[:2]
        max_side = MATTE_MAX_SIDE
        while True:
            scale = min(1.0, max_side / max(height, width))
            size = (max(32, round(width * scale)), max(32, round(height * scale)))
            small_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
            small_trimap = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST) if scale < 1 else trimap
            try:
                with self.gpu, self._autocast():
                    inputs = self._matte_processor(images=small_rgb, trimaps=small_trimap, return_tensors="pt")
                    # The processor pads bottom/right to a multiple of 32: crop, never resize.
                    alpha = matte(pixel_values=inputs.pixel_values.to(self.device)).alphas[0, 0]
                    alpha = alpha[: small_rgb.shape[0], : small_rgb.shape[1]].float().clamp(0, 1).cpu().numpy()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if max_side <= MIN_RETRY_SIDE:
                    raise
                max_side = max(MIN_RETRY_SIDE, int(max_side * 0.7))
                log.warning("ViTMatte out of memory, retrying at %d", max_side)
        return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR) if scale < 1 else alpha

    # ------------------------------------------------------------------
    # LaMa
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inpaint(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """Copy of `rgb` with `hole` filled; pixels outside it stay exact."""
        lama = self._load_lama()
        height, width = rgb.shape[:2]
        max_side = INPAINT_MAX_SIDE
        while True:
            scale = min(1.0, max_side / max(height, width))
            size = (max(8, round(width * scale)), max(8, round(height * scale)))
            small_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
            small_hole = (
                cv2.resize(hole.astype(np.uint8), size, interpolation=cv2.INTER_LINEAR) if scale < 1 else hole.astype(np.uint8)
            )
            # LaMa's FFC layers need sides divisible by 8.
            pad_w, pad_h = -size[0] % 8, -size[1] % 8
            padded_rgb = cv2.copyMakeBorder(small_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            padded_hole = cv2.copyMakeBorder(small_hole, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            image = torch.from_numpy(padded_rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
            mask = torch.from_numpy(padded_hole > 0)[None, None].float().to(self.device)
            try:
                with self.gpu:  # no autocast: the Fourier layers need full precision
                    result = lama(image, mask)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if max_side <= MIN_RETRY_SIDE:
                    raise
                max_side = max(MIN_RETRY_SIDE, int(max_side * 0.7))
                log.warning("LaMa out of memory, retrying at %d", max_side)
        result = result[: size[1], : size[0]]
        out = rgb.copy()
        if scale >= 1:
            out[hole] = result[hole]
            return out
        # Upscaled fill is softer than its surroundings: feather inside the hole only.
        upscaled = cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)
        weight = np.clip(cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3) / 4, 0, 1)[..., None]
        blended = weight * upscaled + (1 - weight) * rgb
        out[hole] = np.clip(blended[hole] + 0.5, 0, 255)
        return out


engine = Engine()
