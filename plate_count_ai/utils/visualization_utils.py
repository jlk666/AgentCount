"""Visualization helpers for colony detection reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_annotations(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    colony_count: int | None,
    cfu_per_ml: float | None,
    qc_status: dict[str, Any] | None,
    validation_status: dict[str, Any] | None,
    warnings: list[str] | None,
) -> np.ndarray:
    """Overlay detections and status text onto an image."""
    canvas = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("confidence", 0.0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 200, 30), 2)
        cv2.putText(
            canvas,
            f"{conf:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (30, 200, 30),
            1,
            cv2.LINE_AA,
        )

    y = 22
    lines = [
        f"Colonies: {colony_count if colony_count is not None else 'N/A'}",
        f"CFU/mL: {cfu_per_ml:.2f}" if isinstance(cfu_per_ml, (int, float)) else "CFU/mL: N/A",
    ]
    if qc_status is not None:
        lines.append(f"QC pass: {qc_status.get('passed', False)}")
    if validation_status is not None:
        lines.append(f"Validation pass: {validation_status.get('passed', False)}")
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return canvas


def save_image(image: np.ndarray, output_path: str | Path) -> str:
    """Persist an image and return normalized path string."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to save image: {output_path}")
    return str(output_path)
