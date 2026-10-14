"""Agent for agar plate localization and cropping."""

from __future__ import annotations

from typing import Any

from plate_count_ai.utils.image_utils import crop_to_circle, detect_plate_circle


class PlateSegmentationAgent:
    """Extract plate ROI from the full image."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        image = state.get("processed_image")
        if image is None:
            raise ValueError("PlateSegmentationAgent requires 'processed_image' in state.")

        plate = detect_plate_circle(image)
        warnings = list(state.get("warnings", []))

        if plate["found"] and plate["radius"] is not None:
            plate_image = crop_to_circle(
                image=image,
                center_x=plate["center_x"],
                center_y=plate["center_y"],
                radius=plate["radius"],
            )
        else:
            # Fallback to full image if segmentation cannot reliably localize plate.
            warnings.append("Using full image as plate ROI (segmentation fallback).")
            plate_image = image.copy()

        return {
            **state,
            "plate_image": plate_image,
            "warnings": warnings,
        }
