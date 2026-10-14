"""Agent for translating detections into microbiology-aware counts."""

from __future__ import annotations

from typing import Any

from plate_count_ai.config.settings import Settings
from plate_count_ai.utils.image_utils import dense_growth_texture_fraction


class ColonyCountingAgent:
    """Count detections and flag plates outside ideal counting range."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        detections = state.get("detections", [])
        colony_count = len(detections)
        warnings = list(state.get("warnings", []))
        plate_image = state.get("plate_image")
        texture_fraction = dense_growth_texture_fraction(plate_image) if plate_image is not None else 0.0
        control = state.get("metadata", {}).get("is_negative_control") is True

        if control:
            count_status = "negative_control"
            if colony_count > 0:
                warnings.append("Negative control has detected growth; review for contamination.")
            elif texture_fraction >= self.settings.tntc_texture_fraction_threshold:
                warnings.append("Negative control has dense texture despite zero detections; review for contamination.")
        elif colony_count > self.settings.max_recommended_colonies:
            count_status = "TNTC"
            warnings.append("Colony count above recommended range; plate recorded as TNTC.")
        elif colony_count < self.settings.min_recommended_colonies and texture_fraction >= self.settings.tntc_texture_fraction_threshold:
            count_status = "TNTC"
            warnings.append(
                f"Dense plate texture ({texture_fraction:.2f}) suggests overgrowth; "
                f"automated count {colony_count} recorded as TNTC pending review."
            )
        elif colony_count < self.settings.min_recommended_colonies:
            count_status = "below_counting_range"
            warnings.append(
                f"Colony count {colony_count} below recommended range "
                f"({self.settings.min_recommended_colonies}-{self.settings.max_recommended_colonies})."
            )
        else:
            count_status = "countable_candidate"

        return {
            **state,
            "colony_count": colony_count,
            "count_status": count_status,
            "tntc_texture_fraction": texture_fraction,
            "warnings": warnings,
        }
