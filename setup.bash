#!/bin/bash
set -e

# --- Configuration ---
HF_REPO="Lelouchl2320/finetuned_qwen2.5"
MODEL_DIR="merged_model"

echo "===== Installing dependencies ====="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49.0" accelerate qwen-vl-utils pillow pandas tqdm huggingface_hub

echo "===== Downloading Merged Model from Hugging Face ====="
# This ensures the evaluator gets your exact fine-tuned weights
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$HF_REPO', local_dir='$MODEL_DIR')"

echo "===== Setup complete! Model is ready in $MODEL_DIR ====="
