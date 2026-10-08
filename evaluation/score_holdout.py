"""Compare frozen holdout predictions with completed blind manual counts."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
VALID_STATUSES = {"countable", "too_numerous", "confluent", "unreadable", "other"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    manual = rows(HERE / "holdout_manual_counts.csv")
    predictions = {r["plate_id"]: r for r in rows(HERE / "holdout_predictions.csv")}
    if len(manual) != 15 or len(predictions) != 15:
        raise ValueError("Expected 15 manual rows and 15 frozen predictions")
    if len({r["plate_id"] for r in manual}) != 15 or set(predictions) != {r["plate_id"] for r in manual}:
        raise ValueError("Manual and prediction plate IDs do not match one to one")

    incomplete = [r["plate_id"] for r in manual if r["count_status"] not in VALID_STATUSES]
    if incomplete:
        raise ValueError(f"Complete count_status for all plates: {', '.join(incomplete)}")

    eligible = []
    review_flags = []
    for row in manual:
        pred = predictions[row["plate_id"]]
        predicted_count = int(pred["colony_count"]) if pred["colony_count"] else None
        flagged = (
            bool(pred["error"])
            or predicted_count is None
            or not 30 <= predicted_count <= 300
            or pred["qc_passed"] != "True"
            or pred["validation_passed"] != "True"
        )
        review_flags.append((row["count_status"], flagged))
        if row["count_status"] != "countable":
            continue
        try:
            count = int(row["manual_count"])
        except ValueError as exc:
            raise ValueError(f"Invalid manual_count for {row['plate_id']}") from exc
        if count < 0:
            raise ValueError(f"Negative manual_count for {row['plate_id']}")
        if pred["error"] or pred["colony_count"] == "":
            continue
        actual = int(pred["colony_count"])
        eligible.append((row["plate_id"], pred["lab"], pred["medium"], pred["dilution_code"], count, actual))

    print(f"Holdout plates: {len(manual)}")
    print(f"Countable plates with predictions: {len(eligible)}")
    print(f"Other statuses: {len(manual) - len(eligible)}")
    for status in sorted(VALID_STATUSES):
        group = [flag for observed, flag in review_flags if observed == status]
        if group:
            print(f"Review flag for {status}: {sum(group)}/{len(group)}")
    if not eligible:
        return
    errors = [abs(a - p) for _, _, _, _, a, p in eligible]
    print(f"Mean absolute error: {mean(errors):.2f} colonies")
    print(f"Median absolute error: {median(errors):.2f} colonies")
    print(f"Within ±5 colonies or ±10%: {sum(e <= max(5, a * 0.1) for (_, _, _, _, a, _), e in zip(eligible, errors))}/{len(eligible)}")
    nonzero = [(abs(a - p) / a) for _, _, _, _, a, p in eligible if a > 0]
    if nonzero:
        print(f"Mean relative absolute error (nonzero manual counts): {mean(nonzero):.1%}")
    for lab, medium, dilution in sorted({(lab, m, d) for _, lab, m, d, _, _ in eligible}):
        subset = [(a, p) for _, lab_i, m, d, a, p in eligible if (lab_i, m, d) == (lab, medium, dilution)]
        print(f"{lab} {medium} dilution {dilution}: n={len(subset)}, MAE={mean(abs(a-p) for a,p in subset):.2f}")


if __name__ == "__main__":
    main()
