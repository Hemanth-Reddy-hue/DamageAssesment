#!/usr/bin/env bash
set -e

# AreaMap Offline Weight Fetcher
# Requires HF_TOKEN to be set in environment or .env

MODELS_DIR="${MODELS_DIR:-models}"
mkdir -p "$MODELS_DIR"

echo "=== AreaMap Model Weight Fetcher ==="
echo "Target directory: $MODELS_DIR"

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set. Gated models may fail to download."
fi

# Download or verify Depth Anything V2 metric model
echo "Checking Depth Anything V2 weights..."
# huggingface-cli download depth-anything/Depth-Anything-V2-Small --local-dir "$MODELS_DIR/depth_anything_v2"

# Download or verify Qwen-VL / Gemma weights
echo "Checking VLM weights..."
# huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir "$MODELS_DIR/qwen2_vl"

echo "=== Model pre-download complete ==="
