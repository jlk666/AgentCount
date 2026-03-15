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


def draw_sam_masks_overlay(
    image: np.ndarray,
    raw_masks: list[dict[str, Any]],
    max_masks: int = 512,
) -> np.ndarray:
    """Overlay raw SAM masks right after inference."""
    canvas = image.copy()
    overlay = np.zeros_like(canvas, dtype=np.uint8)

    for idx, ann in enumerate(raw_masks[:max_masks]):
        mask = ann.get("segmentation")
        if mask is None:
            continue
        mask_bool = mask.astype(bool)
        color = np.array(
            [
                (53 * (idx + 1)) % 255,
                (97 * (idx + 1)) % 255,
                (193 * (idx + 1)) % 255,
            ],
            dtype=np.uint8,
        )
        overlay[mask_bool] = color

        x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(int(c) for c in color), 1)

    canvas = cv2.addWeighted(canvas, 0.65, overlay, 0.35, 0.0)
    cv2.putText(
        canvas,
        f"SAM raw masks: {len(raw_masks)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_image(image: np.ndarray, output_path: str | Path) -> str:
    """Persist an image and return normalized path string."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to save image: {output_path}")
    return str(output_path)
