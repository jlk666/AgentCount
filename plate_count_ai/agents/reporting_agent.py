"""Agent for generating annotated outputs and structured reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from plate_count_ai.config.settings import Settings
from plate_count_ai.utils.visualization_utils import draw_annotations, draw_sam_masks_overlay, save_image


class ReportingAgent:
    """Persist image overlays plus JSON and CSV result artifacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(state.get("metadata", {}))
        sample_id = metadata.get("sample_id", "unknown_sample")
        replicate_id = metadata.get("replicate_id", "r0")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{sample_id}_{replicate_id}_{timestamp}"

        plate_image = state.get("plate_image")
        if plate_image is None:
            raise ValueError("ReportingAgent requires 'plate_image' in state.")

        annotated = draw_annotations(
            image=plate_image,
            detections=state.get("detections", []),
            colony_count=state.get("colony_count"),
            cfu_per_ml=state.get("cfu_per_ml"),
            qc_status=state.get("qc_status"),
            validation_status=state.get("validation_status"),
            warnings=state.get("warnings", []),
        )

        annotated_path = save_image(annotated, self.settings.output_dir / f"{base_name}_annotated.png")
        raw_sam_masks = state.get("raw_sam_masks", [])
        sam_masks_path = ""
        if raw_sam_masks:
            sam_overlay = draw_sam_masks_overlay(image=plate_image, raw_masks=raw_sam_masks)
            sam_masks_path = save_image(sam_overlay, self.settings.output_dir / f"{base_name}_sam_masks.png")
        result_payload = {
            "image_path": state.get("image_path"),
            "metadata": metadata,
            "qc_status": state.get("qc_status"),
            "colony_count": state.get("colony_count"),
            "cfu_per_ml": state.get("cfu_per_ml"),
            "validation_status": state.get("validation_status"),
            "warnings": state.get("warnings", []),
            "annotated_image_path": annotated_path,
            "sam_masks_image_path": sam_masks_path or None,
        }

        json_path = self.settings.output_dir / f"{base_name}_result.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        csv_path = self.settings.output_dir / "summary.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "sample_id",
                    "replicate_id",
                    "dilution",
                    "volume",
                    "colony_count",
                    "cfu_per_ml",
                    "qc_passed",
                    "validation_passed",
                    "warning_count",
                    "annotated_image_path",
                    "result_json_path",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "sample_id": sample_id,
                    "replicate_id": replicate_id,
                    "dilution": metadata.get("dilution"),
                    "volume": metadata.get("volume"),
                    "colony_count": state.get("colony_count"),
                    "cfu_per_ml": state.get("cfu_per_ml"),
                    "qc_passed": state.get("qc_status", {}).get("passed"),
                    "validation_passed": state.get("validation_status", {}).get("passed"),
                    "warning_count": len(state.get("warnings", [])),
                    "annotated_image_path": annotated_path,
                    "result_json_path": str(json_path),
                }
            )

        return {
            **state,
            "result_json_path": str(json_path),
            "summary_csv_path": str(csv_path),
            "annotated_image_path": annotated_path,
            "sam_masks_image_path": sam_masks_path,
            "result": result_payload,
        }
