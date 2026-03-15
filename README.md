# AgentCount

AgentCount is a Python project for agar plate colony counting and CFU estimation.  
It uses a multi-agent workflow (LangGraph) with SAM-based segmentation/detection, QC, validation, and reporting.

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
