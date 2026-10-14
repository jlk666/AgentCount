"""Agent responsible for input image quality checks."""

from __future__ import annotations

from typing import Any

from plate_count_ai.config.settings import Settings
from plate_count_ai.utils.image_utils import detect_plate_circle, exposure_stats, laplacian_variance, load_image


class ImageQCAgent:
    """Evaluate blur, exposure, and plate presence before inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        image_path = state["image_path"]
        image = load_image(image_path)
        blur = laplacian_variance(image)
        exposure = exposure_stats(image)
        underexposed = exposure["mean_brightness"] < self.settings.underexposed_mean_threshold
        overexposed = exposure["mean_brightness"] > self.settings.overexposed_mean_threshold
        plate = detect_plate_circle(image)

        warnings = list(state.get("warnings", []))
        if blur < self.settings.blur_threshold:
            warnings.append("Image may be blurry (low Laplacian variance).")
        if underexposed:
            warnings.append("Image appears underexposed.")
        if overexposed:
            warnings.append("Image appears overexposed.")
        if not plate["found"]:
            warnings.append("No clear plate boundary detected during QC.")

        qc_status = {
            "passed": bool(
                blur >= self.settings.blur_threshold
                and not underexposed
                and not overexposed
                and plate["score"] >= self.settings.plate_presence_score_threshold
            ),
            "blur_score": blur,
            "exposure": {
                **exposure,
                "underexposed": underexposed,
                "overexposed": overexposed,
            },
            "plate_presence": plate,
        }

        return {
            **state,
            "processed_image": image,
            "qc_status": qc_status,
            "warnings": warnings,
        }
