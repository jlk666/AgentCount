# Overlapping-tile SAM3 experiment

This is a development-only diagnostic. No holdout image or manual holdout count was used.

## Method

- Same SAM3 checkpoint, `bacterial colonies` prompt, image enhancement, and mask-filter *ratios* as the frozen baseline. Area ratios are applied to each tile, so their absolute pixel cutoffs differ from the full-plate baseline. The change in detections cannot be attributed solely to image scale.
- Plate localization and crop are unchanged.
- Split each cropped plate into 1008 × 1008-pixel tiles with 252-pixel overlap. Edge tiles are shifted to cover the whole crop.
- Keep a detection only from the tile that owns its center in an overlap, then apply global box NMS. Record raw masks and accepted detections for each tile.
- A separate 512-pixel tile / 128-pixel overlap pilot was run on two development plates.

Reproduce with `python evaluation/try_tiled_sam3.py --all-development --skip-existing` in the `agentcount` environment with a CUDA GPU. Per-image tile diagnostics and overlays are in `evaluation/tiled_dev/`. The comparison table is `evaluation/tiled_dev_summary.csv`.

## Results

| Group | Development plates | Nonzero baseline | Nonzero tiled | Baseline detections | Tiled detections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lab_1 PCA | 16 | 0 | 4 | 0 | 18 |
| Lab_2 VRBA | 3 | 1 | 2 | 1 | 61 |
| Lab_5 MAC | 8 | 1 | 5 | 4 | 435 |
| Lab_5 MRS | 4 | 0 | 1 | 0 | 16 |
| Lab_5 PCA | 8 | 2 | 4 | 25 | 71 |
| **Total** | **39** | **4** | **16** | **30** | **601** |

These are model detections, not verified colony counts. Overlapping tiles increase the chance that SAM3 returns masks, but the improvement is uneven. The visibly dense `Lab_1_PCA_0_02` plate still produced zero raw masks in every 1008-pixel tile. In the 512-pixel pilot, it again produced zero masks; `Lab_5_PCA_1_05` fell from 20 accepted detections at 1008 pixels to 4 at 512 pixels.

Visual review of the overlays shows that many boxed objects on `Lab_5_MAC_0_02` are plausible colonies, while some boxes are near the rim or other artifacts and conspicuous colonies remain unboxed. `Lab_5_PCA_1_01` also has many visible colonies beyond its 14 tiled detections. A higher detection total therefore does not establish higher accuracy. Do not convert these results into reportable CFU/mL or apply this tiled configuration to the holdout as a final model without reference counts and further validation.
