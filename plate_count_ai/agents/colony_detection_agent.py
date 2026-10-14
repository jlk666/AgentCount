"""Agent for colony object detection."""

from __future__ import annotations

from typing import Any

from plate_count_ai.models.colony_detector import ColonyDetector


class ColonyDetectionAgent:
    """Run model inference and attach detections to state."""

    def __init__(self, detector: ColonyDetector) -> None:
        self.detector = detector

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        plate_image = state.get("plate_image")
        if plate_image is None:
            raise ValueError("ColonyDetectionAgent requires 'plate_image' in state.")

        if self.detector.enable_tiling:
            detections, raw_count, diagnostics, raw_overlay = self.detector.detect_tiled_with_intermediate(plate_image)
            return {
                **state,
                "detections": detections,
                "raw_sam_masks": [],
                "raw_sam_mask_count": raw_count,
                "tile_diagnostics": diagnostics,
                "raw_sam_overlay": raw_overlay,
            }

        detections, raw_sam_masks = self.detector.detect_with_intermediate(plate_image)
        return {
            **state,
            "detections": detections,
            "raw_sam_masks": raw_sam_masks,
            "raw_sam_mask_count": len(raw_sam_masks),
            "tile_diagnostics": [],
        }
