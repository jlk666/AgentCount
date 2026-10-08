"""Run the frozen AgentCount workflow on one real-data split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plate_count_ai.config.settings import settings
from plate_count_ai.workflows.plate_workflow import PlateWorkflow


HERE = Path(__file__).resolve().parent
RESULT_FIELDS = [
    "plate_id", "image_path", "lab", "medium", "dilution_code", "dilution",
    "volume_ml", "sampling_time_h", "is_negative_control", "colony_count",
    "cfu_per_ml", "qc_passed", "validation_passed", "warnings",
    "annotated_image_path", "result_json_path", "error",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "holdout"], required=True)
    args = parser.parse_args()

    with (HERE / "real_data_manifest.csv").open(newline="", encoding="utf-8") as handle:
        plates = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if not plates:
        raise ValueError(f"No plates in {args.split} split")

    output_dir = ROOT / "plate_count_ai" / "outputs" / "real_48h_v1" / args.split
    cfg = replace(settings, output_dir=output_dir, sam_backbone="sam3",
                  detector_model_path="checkpoints/sam3.pt", sam_use_tiling=False)
    workflow = PlateWorkflow(cfg)
    results_path = HERE / f"{args.split}_predictions.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for index, plate in enumerate(plates, 1):
            print(f"[{index}/{len(plates)}] {plate['plate_id']}", flush=True)
            control = plate["is_negative_control"] == "true"
            metadata = {
                "sample_id": plate["sample_id"],
                "replicate_id": plate["replicate_id"],
                "dilution": 1.0 if control else float(plate["dilution"]),
                "volume": float(plate["volume_ml"]),
                "is_negative_control": control,
                "sampling_time_h": int(plate["sampling_time_h"]),
                "lab": plate["lab"],
                "medium": plate["medium"],
            }
            record = {key: plate.get(key, "") for key in RESULT_FIELDS}
            try:
                state = workflow.run(str(ROOT / plate["image_path"]), metadata)
                record.update({
                    "colony_count": state.get("colony_count"),
                    "cfu_per_ml": state.get("cfu_per_ml"),
                    "qc_passed": state.get("qc_status", {}).get("passed"),
                    "validation_passed": state.get("validation_status", {}).get("passed"),
                    "warnings": json.dumps(state.get("warnings", [])),
                    "annotated_image_path": state.get("annotated_image_path"),
                    "result_json_path": state.get("result_json_path"),
                    "error": "",
                })
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(record)
            handle.flush()
    print(f"Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
