"""Agent for heuristic validation checks on detections and plate quality."""

from __future__ import annotations

from typing import Any

from plate_count_ai.config.settings import Settings


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
    if inter_area == 0:
        return 0.0
    area_a = max(1, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(1, (xb2 - xb1) * (yb2 - yb1))
    union = area_a + area_b - inter_area
    return float(inter_area / max(union, 1))


class ValidationAgent:
    """Detect merged colonies, edge artifacts, and overgrowth indicators."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        detections = state.get("detections", [])
        plate_image = state.get("plate_image")
        colony_count = int(state.get("colony_count", 0))
        warnings = list(state.get("warnings", []))

        merged_colony_pairs = 0
        for i, det_a in enumerate(detections):
            for det_b in detections[i + 1 :]:
                if _bbox_iou(det_a["bbox"], det_b["bbox"]) > self.settings.merge_iou_threshold:
                    merged_colony_pairs += 1

        edge_artifacts = 0
        if plate_image is not None:
            h, w = plate_image.shape[:2]
            mx = int(w * self.settings.edge_margin_ratio)
            my = int(h * self.settings.edge_margin_ratio)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                near_edge = x1 <= mx or y1 <= my or x2 >= (w - mx) or y2 >= (h - my)
                if near_edge:
                    edge_artifacts += 1

        overgrown = colony_count >= self.settings.overgrown_count_threshold
        if merged_colony_pairs > 0:
            warnings.append(f"Potential merged colonies detected: {merged_colony_pairs} overlapping pairs.")
        if edge_artifacts > 0:
            warnings.append(f"Potential edge artifacts detected: {edge_artifacts} colonies near border.")
        if overgrown:
            warnings.append("Plate may be overgrown based on high colony count.")

        passed = not overgrown and merged_colony_pairs == 0
        validation_status = {
            "passed": passed,
            "merged_colony_pairs": merged_colony_pairs,
            "edge_artifacts": edge_artifacts,
            "overgrown": overgrown,
        }

        return {
            **state,
            "validation_status": validation_status,
            "warnings": warnings,
        }
