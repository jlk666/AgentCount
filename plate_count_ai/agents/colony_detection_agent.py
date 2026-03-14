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

        detections = self.detector.detect(plate_image)
        return {
            **state,
            "detections": detections,
        }
