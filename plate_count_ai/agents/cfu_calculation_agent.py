"""Agent for colony-forming unit concentration calculation."""

from __future__ import annotations

from typing import Any


class CFUCalculationAgent:
    """Compute CFU/mL from colony count and plating metadata."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        colony_count = state.get("colony_count")
        metadata = state.get("metadata", {})
        warnings = list(state.get("warnings", []))

        if metadata.get("is_negative_control") is True:
            return {
                **state,
                "cfu_per_ml": None,
                "warnings": warnings + ["Negative control: sample CFU/mL is not applicable."],
            }

        if state.get("count_status") == "TNTC":
            return {
                **state,
                "cfu_per_ml": None,
                "warnings": warnings + ["TNTC plate: a numeric CFU/mL estimate is not reported."],
            }

        dilution = float(metadata.get("dilution", 0.0))
        volume = float(metadata.get("volume", 0.0))

        cfu_per_ml = None
        if colony_count is None:
            warnings.append("CFU calculation skipped: colony_count is missing.")
        elif dilution <= 0 or volume <= 0:
            warnings.append("CFU calculation skipped: dilution and volume must be positive.")
        else:
            cfu_per_ml = float(colony_count / (dilution * volume))

        return {
            **state,
            "cfu_per_ml": cfu_per_ml,
            "warnings": warnings,
        }
