"""Agent for translating detections into microbiology-aware counts."""

from __future__ import annotations

from typing import Any

from plate_count_ai.config.settings import Settings


class ColonyCountingAgent:
    """Count detections and flag plates outside ideal counting range."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        detections = state.get("detections", [])
        colony_count = len(detections)
        warnings = list(state.get("warnings", []))

        if colony_count < self.settings.min_recommended_colonies:
            warnings.append(
                f"Colony count {colony_count} below recommended range "
                f"({self.settings.min_recommended_colonies}-{self.settings.max_recommended_colonies})."
            )
        if colony_count > self.settings.max_recommended_colonies:
            warnings.append(
                f"Colony count {colony_count} above recommended range "
                f"({self.settings.min_recommended_colonies}-{self.settings.max_recommended_colonies})."
            )

        return {
            **state,
            "colony_count": colony_count,
            "warnings": warnings,
        }
