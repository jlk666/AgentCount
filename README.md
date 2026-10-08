# AgentCount

<p align="center">
  <img src="AgentCountLogo.png" alt="AgentCount Logo" width="600"/>
</p>

AgentCount is a Python project for agar plate colony counting and CFU estimation.  
It uses a multi-agent workflow (LangGraph) with SAM-based segmentation/detection, QC, validation, and reporting.

## Abstract Overview

Rapid and reproducible colony enumeration remains a bottleneck in microbiology workflows, particularly for high-throughput plate-based assays where manual counting is labor-intensive and variable across operators. We developed AgentCount, a modular AI workflow for automated colony counting and CFU/mL estimation from agar plate images. The framework combines staged quality control, plate-region segmentation, prompt-guided segmentation with Segment Anything Model 3 (SAM3), morphology-aware filtering (area ratio, circularity, non-maximum suppression), counting, CFU computation, and report generation. AgentCount supports both single-image and batch inference and exports traceable intermediate artifacts, including raw SAM mask overlays, to improve interpretability and debugging.

To improve detection robustness across visually heterogeneous plates, we implemented prompt ensembling and image enhancement (contrast-limited adaptive histogram equalization plus mild sharpening) prior to SAM3 inference. In internal evaluations, prompt and preprocessing choices substantially influenced recall, while postprocessing reduced duplicate and spurious detections. The workflow produced machine-readable JSON and CSV outputs and enabled reproducible processing of synthetic and challenging test sets with configurable metadata (sample ID, dilution, plated volume, replicate ID). These results demonstrate that an agentic, modular vision pipeline can support scalable colony quantification and standardized CFU reporting in microbiology imaging workflows.

<p align="center">
  <img src="workFlow.jpg" alt="AgentCount Workflow" width="900"/>
</p>

## Quick Setup

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd AgentCount
```

### 2) Create a Python environment

Use either `venv` or Conda (both work; this repo is pip/requirements-based).

#### Option A: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Option B: Conda / Miniconda

```bash
conda create -n agentcount python=3.10 -y
conda activate agentcount
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Download model weights

```bash
python download_models.py
```

Notes:
- Default backend is SAM3, which requires HuggingFace access for `facebook/sam3`.
- If you do not have SAM3 access yet, you can skip it:

```bash
python download_models.py --skip-sam3
```

## Run the Project

### Single Image (CLI)

Minimal run:

```bash
python main.py --image /path/to/plate_image.png
```

Run with explicit metadata:

```bash
python main.py \
  --image /path/to/plate_image.png \
  --metadata '{"sample_id":"sample_001","dilution":0.01,"volume":0.1,"replicate_id":"r1"}'
```

Run with SAM3 backend options:

```bash
python main.py \
  --image /path/to/plate_image.png \
  --sam-backend sam3 \
  --checkpoint checkpoints/sam3.pt \
  --sam-text-prompt "small circular colonies on TSA agar"
```

### Batch Run (Folder)

Use one metadata template for all images in a folder:

```bash
python main.py --batch-dir /path/to/folder_with_images
```

### Batch Run (CSV - Recommended)

Use per-image metadata from a CSV file:

```csv
image_path,sample_id,dilution,volume,replicate_id
SyntheticData/Day14-TSA-r1.png,Day14-TSA,0.01,0.1,r1
SyntheticData/Day14-TSA-r2.png,Day14-TSA,0.01,0.1,r2
```

Run:

```bash
python main.py --batch-csv SyntheticData/input_parameters.csv
```

For the 48-hour real-data development split, use the reviewed manifest metadata:

```bash
python main.py \
  --batch-csv evaluation/development_input_parameters.csv \
  --output-dir plate_count_ai/outputs/real_48h_tiled_v1 \
  --sam-backend sam3
```

SAM3 uses overlapping 1008-pixel tiles with 252-pixel overlap by default. Disable with `--no-tiling`, or adjust using `--tile-size` and `--tile-overlap`. Each result includes raw tile-mask counts, tile diagnostics, and a mask overlay. The raw detection count is preserved even when the plate is classified `TNTC`.

The count status is `countable_candidate`, `below_counting_range`, `TNTC`, or `negative_control`. A low detected count plus strong image texture is marked `TNTC` for human review; TNTC and negative controls have no numeric CFU/mL result. This texture rule is a screening heuristic tuned on the development split, so its labels require review before scientific use. Batch reports omit pairwise tests by default; use `--pairwise-tests` only when the experiment supports those comparisons.

Optional SAM3 prompt ensemble (pipe-separated):

```bash
SAM_TEXT_PROMPTS='small circular light-brown colonies on TSA agar plate interior|discrete circular colonies on tryptic soy agar|round isolated colony spots on agar surface' \
SAM_ENABLE_IMAGE_ENHANCEMENT=1 \
python main.py \
  --batch-csv SyntheticData/input_parameters.csv \
  --sam-backend sam3 \
  --checkpoint checkpoints/sam3.pt \
  --sam-text-prompt "small circular light-brown colonies on TSA agar plate interior"
```

### CLI Parameters (`main.py`)

`main.py` supports the following arguments:

- `--image <path>`: run one image.
- `--metadata '<json>'`: metadata JSON for single-image mode.
- `--batch-dir <dir>`: run all images in a folder (same metadata template for all).
- `--batch-csv <csv>`: run batch from CSV where each row has `image_path` + metadata columns.
- `--output-dir <dir>`: write artifacts to a separate directory.
- `--no-tiling`, `--tile-size <pixels>`, `--tile-overlap <pixels>`: control SAM3 tiling.
- `--pairwise-tests`: add pairwise tests to the batch report when justified by the experimental design.
- `--sam-backend <sam1|sam3>`: choose segmentation backend.
- `--checkpoint <path>`: model checkpoint path for selected backend.
- `--sam-model-type <vit_b|vit_l|vit_h>`: SAM1 model type (used when `--sam-backend sam1`).
- `--sam-text-prompt "<text>"`: primary text prompt for SAM3.

Notes:
- If both `--batch-dir` and `--batch-csv` are provided, `--batch-dir` is used first.
- In CSV mode, missing metadata fields fall back to defaults from the code:
  - `sample_id=sample_001`
  - `dilution=0.01`
  - `volume=0.1`
  - `replicate_id=r1`

### Metadata Fields

The framework expects metadata with these keys:

- `sample_id` (string): sample identifier used in output filenames/reporting.
- `dilution` (float): dilution factor used for CFU/mL calculation.
- `volume` (float): plated volume (mL) used for CFU/mL calculation.
- `replicate_id` (string): replicate label (for example `r1`, `r2`).

Example:

```json
{
  "sample_id": "Day14-TSA",
  "dilution": 0.01,
  "volume": 0.1,
  "replicate_id": "r1"
}
```

### FastAPI server

Start server:

```bash
uvicorn plate_count_ai.api.server:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Analyze endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/analyze_plate" \
  -F 'image=@/path/to/plate_image.png' \
  -F 'metadata={"sample_id":"sample_001","dilution":0.01,"volume":0.1,"replicate_id":"r1"}'
```

## Output Files

Generated outputs are written to:
- `plate_count_ai/outputs/`

Typical artifacts:
- Annotated image (`*_annotated.png`)
- Raw SAM masks overlay (`*_sam_masks.png`)
- Per-run JSON result (`*_result.json`)
- Running CSV summary (`summary.csv`)
- Status-aware running CSV summary (`summary_v2.csv` for new runs)

## Configuration

Runtime config lives in `plate_count_ai/config/settings.py`.  
Useful environment variables include:

- `SAM_BACKBONE` (default: `sam3`)
- `SAM_CHECKPOINT` (default: `checkpoints/sam3.pt`)
- `SAM_MODEL_CONFIG`
- `SAM_TEXT_PROMPT`
- `SAM_TEXT_PROMPTS` (optional prompt ensemble separated by `|`)
- `SAM_ENABLE_IMAGE_ENHANCEMENT` (`1` to enable CLAHE+sharpen pre-processing)
- `SAM_AUTO_DOWNLOAD` (`1` to auto-download from URL if configured)

## Troubleshooting

- Missing checkpoint error:
  - Ensure model files exist (run `python download_models.py`).
  - Or set `SAM_CHECKPOINT` to a valid local path.
- SAM3 import/auth issues:
  - Install HuggingFace helper: `pip install huggingface-hub`
  - Authenticate: `hf auth login`
  - Request access to `facebook/sam3` on HuggingFace.
- CPU-only environment:
  - Prefer `sam1` backend if SAM3 dependencies are unavailable.
