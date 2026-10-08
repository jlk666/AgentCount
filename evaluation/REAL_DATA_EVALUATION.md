# Real-data AgentCount evaluation (48 h)

## Input and confirmed metadata

- Dataset: `RealData/FST104L_PlateCount` (54 JPEG plates).
- Sampling time: 48 hours post inoculation for every plate.
- Plated volume: PCA, MAC, MRS = 0.1 mL; VRBA = 1 mL. The data owner wrote “VBRA”; the image folder and filenames use `VRBA`.
- Dilution codes: `0` = 1, `1` = 0.1, `2` = 0.01, `3` = 0.001; `NC` = negative control.
- Sample identity and biological independence across labs and dilutions are not established by these names. Do not claim a time trend from this dataset.

## Fixed split

Run `python evaluation/build_real_manifest.py` to regenerate `real_data_manifest.csv` and `holdout_manual_counts.csv`. One image from each lab/medium/dilution stratum is assigned to holdout using the SHA-256 ranking and fixed seed in that script. This yields **39 development** and **15 holdout** images, including one negative control in each split. Do not change the split after reviewing predictions.

## Agent task

1. Verify each image and parse metadata from its path. Report missing, duplicated, or unexpected files.
2. Run the same frozen model settings on all images. Preserve per-plate counts, image QC, validation flags, annotated images, and errors. Record backend, checkpoint, environment, and any fallback.
3. Treat negative controls as contamination checks. Report their counts but do not calculate sample CFU/mL for them.
4. For sample plates, compute `CFU/mL = count / (dilution × plated volume in mL)` and show the contributing metadata. Mark plates outside the configured 30–300 count range or failing QC/validation for human review; do not silently discard them.
5. Summarize only within justified lab/medium groups. Do not treat multiple dilutions as independent biological replicates or run pairwise tests across media/labs without an experimental design that supports them.
6. Give the reviewer a blind holdout sheet with image paths and empty manual-count fields. Do not put model counts in that sheet.

## Manual review and score

The reviewer fills `manual_count` for each of the 15 holdout images, uses `count_status` for `countable`, `too_numerous`, `confluent`, `unreadable`, or `other`, and notes ambiguous colonies. Counts should be made without opening `holdout_predictions.csv` or annotated model images. A second independent reviewer or adjudication is useful where counts differ substantially.

After the manual sheet is filled, compare only readable, countable holdout plates on absolute count error, relative count error (when manual count is nonzero), and the fraction within ±10% or ±5 colonies. Report negative-control counts separately; score review flags against the human count status. Show results by medium and dilution as well as overall. Preserve every failed or excluded image in the denominator and publish its reason. Do not tune model prompts or thresholds against holdout labels; any later changes require a new held-out evaluation.

Run `python evaluation/score_holdout.py` after filling the sheet. It refuses to score an incomplete or mismatched sheet.

## Reproduce inference

From the repository root, in the `agentcount` environment with a CUDA GPU:

```bash
python evaluation/build_real_manifest.py
python evaluation/run_real_data.py --split development
python evaluation/run_real_data.py --split holdout
```

The runner writes per-split predictions beside this document and annotated images and JSON outputs under `plate_count_ai/outputs/real_48h_v1/`. It does not use the generic CLI's default metadata or its cross-group t-test report.

Plate boundary localization runs on a 1024-pixel thumbnail and maps the circle back to the full-resolution image. Colony detection still uses the full-resolution plate crop. This fixed preprocessing step is applied to both splits.

## Baseline run status

The frozen SAM3 baseline processed all 54 images without execution errors. Its development predictions were 0 colonies on 35 of 39 plates; all 39 were below the configured 30-colony review threshold. The holdout predictions are saved separately and should stay unopened until manual counting is complete. The model's image QC and validation flags passed every development image, including one with visually dense growth and a zero model count. Treat this run as a diagnostic baseline, not a validated CFU result.

## Integrated tiled development run

The main CLI now runs overlapping SAM3 tiles by default and records a count status. A low detection count with strong image texture is marked `TNTC`; its raw detection count is retained and numeric CFU/mL is omitted. The texture rule is a development-set heuristic and needs human review.

The 39 development plates were run through the updated CLI using `development_input_parameters.csv`. Outputs are under `plate_count_ai/outputs/real_48h_tiled_v1/`: 11 `TNTC`, 22 `below_counting_range`, 5 `countable_candidate`, and 1 `negative_control`. The batch HTML report uses compact previews and omits pairwise tests. The integrated workflow was not run on the holdout split.
