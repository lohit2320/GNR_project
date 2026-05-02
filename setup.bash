#!/bin/bash
set -e

echo "===== Initializing Conda ====="
# This line is required to make the 'conda activate' command work inside a bash script
source $(conda info --base)/etc/profile.d/conda.sh

echo "===== Creating Conda Environment from environment.yml ====="
# This reads your environment.yml and installs Python 3.11, PyTorch, Transformers, etc.
conda env create -f environment.yml

echo "===== Activating Environment ====="
conda activate gnr_project_env

echo "===== Installing Unsloth (optimized fine-tuning) ====="
# These were missing from your environment.yml, so we install them here
pip install unsloth
pip install --no-deps xformers "trl<0.9.0" peft bitsandbytes

echo "===== Downloading model weights ====="
python - <<'EOF'
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct"
print(f"Caching {model_id}...")
# We load and save to cache
Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
AutoProcessor.from_pretrained(model_id)
print("Model cached.")
EOF

echo "===== Setup complete! ====="
