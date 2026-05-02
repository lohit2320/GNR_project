#!/bin/bash
set -e

# Use current environment or create one
# If you are on a system like Lambda or RunPod, you might just want to use the base env
# But let's stick to the script's logic if the user wants a clean env.

echo "===== Installing core dependencies ====="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49.0" accelerate qwen-vl-utils pillow pandas tqdm

echo "===== Installing Unsloth (optimized fine-tuning) ====="
# Unsloth for Qwen3-VL
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
