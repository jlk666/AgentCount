"""SAM3-based colony detector using the exact facebook/sam3 package API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class ColonyDetector:
    """Wrapper around Meta SAM3 (`sam3.model_builder`) for colony detection.

    The SAM3 image model is loaded once at initialisation.  Inference uses
    ``Sam3Processor.set_text_prompt`` with a configurable text prompt, so no
    manual point grids are required.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        backbone: str = "sam3",
        model_type: str = "vit_b",
        model_config: str = "",
        text_prompt: str = "bacterial colonies",
        text_prompts_raw: str = "",
        enable_image_enhancement: bool = True,
        checkpoint_url: str = "",
        auto_download_checkpoint: bool = True,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 20,
        min_colony_area_ratio: float = 0.00002,
        max_colony_area_ratio: float = 0.015,
        min_colony_circularity: float = 0.2,
        nms_iou_threshold: float = 0.3,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.backbone = backbone.lower()
        self.model_type = model_type
        self.model_config = model_config
        self.text_prompt = text_prompt
        self.text_prompts = self._parse_prompts(text_prompt, text_prompts_raw)
        self.enable_image_enhancement = enable_image_enhancement
        self.checkpoint_url = checkpoint_url
        self.auto_download_checkpoint = auto_download_checkpoint
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.min_colony_area_ratio = min_colony_area_ratio
        self.max_colony_area_ratio = max_colony_area_ratio
        self.min_colony_circularity = min_colony_circularity
        self.nms_iou_threshold = nms_iou_threshold

        checkpoint_path = Path(self.model_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path

        if not checkpoint_path.exists() and self.auto_download_checkpoint and self.checkpoint_url:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading SAM checkpoint from %s …", self.checkpoint_url)
            try:
                urlretrieve(self.checkpoint_url, str(checkpoint_path))
            except Exception as exc:
                raise FileNotFoundError(
                    "SAM checkpoint was not found and auto-download failed.\n"
                    f"Tried to download from: {self.checkpoint_url}\n"
                    f"Target path: {checkpoint_path}\n"
                    "Set SAM_CHECKPOINT to an existing local path."
                ) from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{self.backbone.upper()} checkpoint not found.\n"
                f"Expected: {checkpoint_path}\n"
                "Place your downloaded checkpoint at the path above or set the "
                "SAM_CHECKPOINT environment variable."
            )

        if self.backbone == "sam1":
            self.mask_generator = self._load_sam1(checkpoint_path)
        else:
            self.mask_generator = self._load_sam3(checkpoint_path)

    @staticmethod
    def _parse_prompts(primary_prompt: str, prompts_raw: str) -> list[str]:
        prompts: list[str] = []
        seen: set[str] = set()

        def _add(p: str) -> None:
            candidate = p.strip()
            if not candidate or candidate in seen:
                return
            seen.add(candidate)
            prompts.append(candidate)

        _add(primary_prompt)
        for token in prompts_raw.split("|"):
            _add(token)
        return prompts

    @staticmethod
    def _enhance_for_sam(image_rgb: np.ndarray) -> np.ndarray:
        """Improve local contrast for low-contrast colony imagery."""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0.0)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_sam1(self, checkpoint_path: Path):
        """Load SAM1 automatic mask generator."""
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Could not import `segment_anything` for SAM1 backend.\n"
                "Install it with:\n"
                "  pip install segment-anything"
            ) from exc

        logger.info(
            "Loading SAM1 model (%s) from %s on device %s …",
            self.model_type,
            checkpoint_path,
            self.device,
        )
        model = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
        model.to(device=self.device)
        logger.info("SAM1 model loaded successfully.")
        return SamAutomaticMaskGenerator(
            model=model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def _load_sam3(self, checkpoint_path: Path) -> _Sam3MaskGeneratorAdapter:
        """Load the SAM3 image model using the exact facebook/sam3 package API."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            missing_mod = getattr(exc, "name", "")
            if missing_mod == "triton":
                raise ImportError(
                    "SAM3 import failed because dependency `triton` is missing.\n"
                    "The facebook/sam3 package currently targets Linux + CUDA workflows.\n"
                    "On macOS/CPU-only environments, use a different backend (e.g. SAM1/YOLO) "
                    "or run this project inside a Linux CUDA environment with Triton available."
                ) from exc
            raise ImportError(
                "Could not import `sam3`. Make sure the facebook/sam3 package is "
                "installed in your environment:\n"
                "  pip install git+https://github.com/facebookresearch/sam3.git\n"
                "or:\n"
                "  pip install sam3"
            ) from exc

        logger.info("Loading SAM3 model from %s on device %s …", checkpoint_path, self.device)

        # build_sam3_image_model(checkpoint_path=..., device=..., load_from_HF=False)
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            load_from_HF=False,
        )
        model = model.to(device=self.device)
        model.eval()

        # Sam3Processor(model, device=...) — device must match model
        processor = Sam3Processor(model, device=self.device)
        logger.info("SAM3 model loaded successfully.")
        return _Sam3MaskGeneratorAdapter(processor=processor, prompt=self.text_prompt)

    @staticmethod
    def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(1, (xa2 - xa1) * (ya2 - ya1))
        area_b = max(1, (xb2 - xb1) * (yb2 - yb1))
        return float(inter_area / max(area_a + area_b - inter_area, 1))

    def _nms(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        kept: list[dict[str, Any]] = []
        for det in sorted_dets:
            if all(self._bbox_iou(det["bbox"], prev["bbox"]) < self.nms_iou_threshold for prev in kept):
                kept.append(det)
        return kept

    def _masks_to_detections(self, masks: list[dict[str, Any]], image: np.ndarray) -> list[dict[str, Any]]:
        """Convert raw SAM masks to filtered detection objects."""
        detections: list[dict[str, Any]] = []
        if not masks:
            return detections

        h, w = image.shape[:2]
        image_area = float(h * w)
        min_area = image_area * self.min_colony_area_ratio
        max_area = image_area * self.max_colony_area_ratio

        for ann in masks:
            area = float(ann.get("area", 0.0))
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = ann["bbox"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w - 1, int(x + bw))
            y2 = min(h - 1, int(y + bh))
            if x2 <= x1 or y2 <= y1:
                continue

            mask = ann.get("segmentation")
            if mask is None:
                continue
            contour_mask = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            if perimeter <= 0:
                continue
            circularity = (4.0 * np.pi * contour_area) / (perimeter * perimeter)
            if circularity < self.min_colony_circularity:
                continue

            confidence = float(max(ann.get("predicted_iou", 0.0), ann.get("stability_score", 0.0)))
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": 0,
                    "class_name": "colony",
                    "mask_area": area,
                    "circularity": float(circularity),
                }
            )
        return self._nms(detections)

    def detect_with_intermediate(self, image: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run SAM segmentation and return both detections and raw SAM masks."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_input = self._enhance_for_sam(image_rgb) if self.enable_image_enhancement else image_rgb
        if hasattr(self.mask_generator, "generate_multi"):
            raw_masks = self.mask_generator.generate_multi(sam_input, self.text_prompts)
        else:
            raw_masks = self.mask_generator.generate(sam_input)
        detections = self._masks_to_detections(raw_masks, image)
        return detections, raw_masks

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run SAM segmentation and return normalized detection-style outputs."""
        detections, _ = self.detect_with_intermediate(image)
        return detections


class _Sam3MaskGeneratorAdapter:
    """Adapts Sam3Processor text-prompt output to the AMG-style list used downstream.

    ``set_text_prompt`` returns the mutated *state* dict with keys:
        - ``masks``   : BoolTensor  [N, 1, H, W]
        - ``boxes``   : FloatTensor [N, 4]  in absolute xyxy pixel coords
        - ``scores``  : FloatTensor [N]
    """

    def __init__(self, processor: Any, prompt: str = "bacterial colonies") -> None:
        self.processor = processor
        self.prompt = prompt

    @staticmethod
    def _to_numpy(t: Any) -> np.ndarray:
        if hasattr(t, "detach"):
            if t.dtype == torch.bfloat16:
                t = t.float()
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _state_to_anns(self, state: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
        masks_t = state.get("masks")    # BoolTensor [N, 1, H, W]
        boxes_t = state.get("boxes")    # FloatTensor [N, 4] xyxy absolute
        scores_t = state.get("scores")  # FloatTensor [N]

        if masks_t is None or boxes_t is None or len(boxes_t) == 0:
            return []

        masks_np = self._to_numpy(masks_t)   # [N, 1, H, W] bool
        boxes_np = self._to_numpy(boxes_t)   # [N, 4]
        scores_np = self._to_numpy(scores_t) if scores_t is not None else np.ones(len(boxes_np))

        anns: list[dict[str, Any]] = []
        for i in range(len(boxes_np)):
            mask_i = masks_np[i]
            if mask_i.ndim > 2:
                mask_i = mask_i[0]              # squeeze channel dim → [H, W]
            mask_bool = mask_i.astype(bool)
            area = int(np.sum(mask_bool))

            x1, y1, x2, y2 = boxes_np[i].tolist()
            # bbox in SAM1-AMG format: [x, y, w, h]
            bbox = [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]
            score = float(scores_np[i]) if i < len(scores_np) else 1.0

            anns.append(
                {
                    "segmentation": mask_bool,
                    "area": area,
                    "bbox": bbox,
                    "predicted_iou": score,
                    "stability_score": score,
                    "prompt": prompt,
                }
            )
        return anns

    def generate(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        pil_img = Image.fromarray(image_rgb)

        # set_image returns state dict; set_text_prompt mutates and returns same dict
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.processor.device.startswith("cuda")):
            state = self.processor.set_image(pil_img)
            state = self.processor.set_text_prompt(prompt=self.prompt, state=state)
        return self._state_to_anns(state=state, prompt=self.prompt)

    def generate_multi(self, image_rgb: np.ndarray, prompts: list[str]) -> list[dict[str, Any]]:
        pil_img = Image.fromarray(image_rgb)
        if not prompts:
            prompts = [self.prompt]

        all_anns: list[dict[str, Any]] = []
        for prompt in prompts:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.processor.device.startswith("cuda")):
                state = self.processor.set_image(pil_img)
                state = self.processor.set_text_prompt(prompt=prompt, state=state)
            all_anns.extend(self._state_to_anns(state=state, prompt=prompt))
        return all_anns
