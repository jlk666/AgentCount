"""Create a reproducible, stratified manifest and blind review sheet."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "RealData" / "FST104L_PlateCount"
OUT = Path(__file__).resolve().parent
SEED = "AgentCount-real-v1"
VOLUME_ML = {"PCA": 0.1, "VRBA": 1.0, "MAC": 0.1, "MRS": 0.1}
FIELDS = [
    "image_path", "plate_id", "lab", "medium", "dilution_code",
    "dilution", "replicate_id", "sample_id", "volume_ml",
    "sampling_time_h", "is_negative_control", "split",
]


def main() -> None:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(DATA.glob("Lab_*/*/*.jpeg")):
        lab, medium, filename = path.relative_to(DATA).parts
        tokens = Path(filename).stem.split("_")
        if len(tokens) != 3 or tokens[0] != medium or medium not in VOLUME_ML:
            raise ValueError(f"Unexpected file name or medium: {path}")
        _, code, replicate = tokens
        if code not in {"0", "1", "2", "3", "NC"} or not replicate.isdigit():
            raise ValueError(f"Unexpected dilution or replicate: {path}")
        control = code == "NC"
        row = {
            "image_path": str(path.relative_to(ROOT)),
            "plate_id": f"{lab}_{medium}_{code}_{replicate}",
            "lab": lab,
            "medium": medium,
            "dilution_code": code,
            "dilution": "" if control else str(10 ** -int(code)),
            "replicate_id": replicate,
            "sample_id": f"{lab}_{medium}_48h_{code}",
            "volume_ml": str(VOLUME_ML[medium]),
            "sampling_time_h": "48",
            "is_negative_control": str(control).lower(),
            "split": "development",
        }
        groups[(lab, medium, code)].append(row)

    rows = [row for group in groups.values() for row in group]
    if len(rows) != 54 or len({row["plate_id"] for row in rows}) != len(rows):
        raise ValueError("Expected 54 uniquely identified plate images")

    for group in groups.values():
        selected = min(
            group,
            key=lambda row: hashlib.sha256(
                f'{SEED}:{row["image_path"]}'.encode()
            ).hexdigest(),
        )
        selected["split"] = "holdout"

    rows.sort(key=lambda row: row["image_path"])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "real_data_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    review_path = OUT / "holdout_manual_counts.csv"
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if any(row.get("manual_count") or row.get("count_status") or row.get("notes") for row in existing):
            raise ValueError("Holdout review sheet has entries; refusing to overwrite it")
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["plate_id", "image_path", "manual_count", "count_status", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["split"] == "holdout":
                writer.writerow({"plate_id": row["plate_id"], "image_path": row["image_path"]})

    batch_fields = ["image_path", "sample_id", "dilution", "volume", "replicate_id",
                    "is_negative_control", "sampling_time_h", "lab", "medium"]
    with (OUT / "development_input_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=batch_fields)
        writer.writeheader()
        for row in rows:
            if row["split"] != "development":
                continue
            writer.writerow({
                "image_path": row["image_path"],
                "sample_id": row["sample_id"],
                "dilution": row["dilution"],
                "volume": row["volume_ml"],
                "replicate_id": row["replicate_id"],
                "is_negative_control": row["is_negative_control"],
                "sampling_time_h": row["sampling_time_h"],
                "lab": row["lab"],
                "medium": row["medium"],
            })

    print(f"Wrote {len(rows)} plates: {sum(row['split'] == 'holdout' for row in rows)} holdout")


if __name__ == "__main__":
    main()
