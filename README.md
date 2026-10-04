# AreaMap: iPhone Capture to Dimensioned Floor Plan

AreaMap is a pipeline that turns an iPhone capture of a property (photos, video walkthroughs, or LiDAR scans) into a stitched, dimensioned whole-property floor plan, including per-surface damage regions, concealed-damage flags, scope line items, and confidence intervals on every measurement.

## Setup Instructions

AreaMap is designed to run locally on your machine with minimal configuration.

### 1. Prerequisites
- **Python 3.11** or higher.
- A virtual environment is highly recommended.

```bash
# Create and activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate
```

### 2. Auto-Setup
The pipeline includes a self-bootstrapping script. You only need to run `main.py` and it will automatically:
1. Install all required Python dependencies from `requirements.txt`.
2. Download and extract the local Ollama LLM server (if running on Windows).
3. Pull the required vision-language models (`llava`, etc.) automatically.
4. Download necessary Hugging Face model weights (Depth-Anything, OWLv2) on first run.

## How to Run

To run the pipeline, simply execute `main.py` and pass the path to your capture data. The pipeline will automatically detect whether the input is a Photo, Video, or LiDAR capture and route it accordingly.

```bash
# Run on a video walkthrough
python main.py Data/1BHKRoom/1bhKRoom.mp4

# Run on a folder containing LiDAR or Photo exports
python main.py Data/HOUSE1
```

### Custom Output Directory
By default, the pipeline outputs the generated JSON schema and SVG floor plan to an `out/` folder grouped by the capture name. You can override this using the `--out` flag:

```bash
python main.py Data/1BHKRoom/1bhKRoom.mp4 --out my_results/
```

### Disabling LLMs or ML Models
If you want to run purely geometric Structure-from-Motion (SfM) without any heavy AI models (Ollama, OWLv2, Depth-Anything), you can disable them:

```bash
# Disable Ollama cloud/local LLMs (Skips damage detection & room semantic naming)
python main.py Data/1BHKRoom/1bhKRoom.mp4 --no-llm

# Disable all heavy Hugging Face models (Disables OWLv2 window detection & Depth-Anything scale recovery)
python main.py Data/1BHKRoom/1bhKRoom.mp4 --no-local-models
```

## Generated Outputs

Once a run completes, you will find three primary artifacts in your output directory:

1. **`plan.json`**: A highly structured JSON file adhering to the AreaMap schema. It contains every wall length, ceiling height, opening (door/window), and detected damage region, fully quantified in metric units.
2. **`plan.svg`**: A vector graphic floor plan rendered directly from the JSON geometry, overlaying damage flags, openings, and calculated dimensions.
3. **`run_log.json`**: An execution audit log containing pipeline telemetry, registration ratios, scale methods, and timing metadata.
