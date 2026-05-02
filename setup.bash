#!/bin/bash
set -e

echo "===== Cloning Repository ====="
# 1. Clone the project and navigate into the folder
git clone https://github.com/lohit2320/GNR_project.git
cd GNR_project

echo "===== Initializing Conda ====="
# 2. Hook Conda into the bash script
source $(conda info --base)/etc/profile.d/conda.sh

echo "===== Creating Conda Environment ====="
# 3. Read the environment.yml from inside the cloned repo
conda env create -f environment.yml

echo "===== Activating Environment ====="
# 4. Switch to the new environment
conda activate gnr_project_env

echo "===== Installing Unsloth (optimized fine-tuning) ====="
# 5. Install the extra dependencies not covered by Conda
pip install unsloth
pip install --no-deps xformers "trl<0.9.0" peft bitsandbytes

echo "===== Downloading model weights ====="
# 6. Cache the Qwen3-VL model using the active Python environment
python - <<'EOF'
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct"
print(f"Caching {model_id}...")

# We load and save to cache
Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
AutoProcessor.from_pretrained(model_id)

print("Model cached successfully.")
EOF

echo "===== Setup complete! ====="
