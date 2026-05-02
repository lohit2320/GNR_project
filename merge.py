import os
import torch
from unsloth import FastVisionModel
from peft import PeftModel

# Constants
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR   = "lora_adapter"
MERGED_DIR    = "merged_model"

print(f"Loading base model: {BASE_MODEL_ID}...")
model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=False,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # Load on CPU first to save GPU memory during merge if needed, or "auto"
)

print(f"Loading adapter from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}...")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("Done! Standalone model is ready in 'merged_model' folder.")
