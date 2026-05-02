import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, required=True)
args = parser.parse_args()

TEST_DIR   = args.test_dir
IMAGE_DIR  = os.path.join(TEST_DIR, "images")
TEST_CSV   = os.path.join(TEST_DIR, "test.csv")
OUTPUT_CSV = "submission.csv"

MODEL_NAME = "merged_model"   # was "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Loading {MODEL_NAME} ...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "45GiB", 1: "45GiB"},
)
model.eval()
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)
print("Model loaded.\n")

def call_model(messages, max_new_tokens=512):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

# ── Stage 1 prompt: pure OCR, no reasoning ───────────────────────────────────
OCR_PROMPT = """Carefully read this image and extract ALL text exactly as written.
Output in this exact format:
QUESTION: <question text here>
OPTION 1: <option 1 text>
OPTION 2: <option 2 text>
OPTION 3: <option 3 text>
OPTION 4: <option 4 text>"""

# ── Stage 2 prompt: pure reasoning, no image ─────────────────────────────────
REASONING_PROMPT = """You are an expert deep learning professor.
Answer this multiple choice question about deep learning.

{extracted_text}

Deep learning topics: neural networks, backpropagation, CNNs, RNNs, LSTMs, 
transformers, self-attention, multi-head attention, batch normalization, dropout, 
layer normalization, ResNets, skip connections, optimizers (SGD, Adam, RMSprop, 
AdaGrad), learning rate schedulers, loss functions (cross entropy, MSE, KL divergence),
activation functions (ReLU, GELU, sigmoid, softmax, tanh), GANs, VAEs, autoencoders,
BERT, GPT, transfer learning, fine-tuning, regularization (L1, L2, weight decay),
gradient vanishing/exploding, Xavier/He initialization, data augmentation.

Think carefully about which option is correct.
Reply with ONLY one digit: 1, 2, 3, or 4."""

# ── Reflection Prompts ───────────────────────────────────────────────────────
STEP1_PROMPT = """Analyze this deep learning MCQ carefully.
1. Extract the core question.
2. Evaluate each option against deep learning principles.
3. Provide your reasoned choice and the final answer digit."""

STEP2_PROMPT = """You previously thought the answer was: {initial_answer}

Double-check the image one more time. 
- Did you miss any subtle details? 
- Is there a 'better' or more specific answer?
- Are there any 'trap' words (e.g., 'not', 'always', 'only')?

If you are 100% sure, restate your answer. If you found a mistake, correct it."""

STEP3_PROMPT = "Based on all your reasoning above, what is the final correct option? Reply with ONLY the digit: 1, 2, 3, or 4."

def predict_with_reflection(image_path: str) -> str:
    """Three-pass inference for maximum accuracy."""
    image = Image.open(image_path).convert("RGB")
    
    # --- Pass 1: Initial Analysis ---
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": STEP1_PROMPT}],
        }
    ]
    initial_response = call_model(messages, max_new_tokens=300)
    print(f"  [Pass 1 - Draft]\n{initial_response[:200]}...")

    # --- Pass 2: Self-Correction/Verification ---
    messages.append({"role": "assistant", "content": [{"type": "text", "text": initial_response}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": STEP2_PROMPT.format(initial_answer=initial_response)}]})
    
    reflection_response = call_model(messages, max_new_tokens=300)
    print(f"  [Pass 2 - Reflection]\n{reflection_response[:200]}...")

    # --- Pass 3: Final Extraction ---
    messages.append({"role": "assistant", "content": [{"type": "text", "text": reflection_response}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": STEP3_PROMPT}]})
    
    final_digit = call_model(messages, max_new_tokens=10)
    print(f"  [Pass 3 - Final] -> {final_digit}")
    
    match = re.search(r"[1-4]", final_digit)
    if match:
        return match.group(0)
    
    # Fallback to checking the reflection text if Pass 3 failed to give a clean digit
    match2 = re.search(r"[1-4]", reflection_response[-50:])
    return match2.group(0) if match2 else "5"

# ... in the main loop, change:
# option = predict_direct(image_path)
# to:
# option = predict_with_reflection(image_path)

if os.path.exists(TEST_CSV):
    df = pd.read_csv(TEST_CSV)
    print(f"Found {len(df)} test samples in CSV.")
else:
    print(f"CSV not found at {TEST_CSV}. Processing all images in {IMAGE_DIR}...")
    # Fallback: list all images in the directory
    if os.path.exists(IMAGE_DIR):
        img_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # remove extension for image_name to match your expected logic
        df = pd.DataFrame({"image_name": [os.path.splitext(f)[0] for f in img_list]})
    else:
        # Final fallback: if --test_dir IS the image directory
        img_list = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        df = pd.DataFrame({"image_name": [os.path.splitext(f)[0] for f in img_list]})
        IMAGE_DIR = TEST_DIR

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
    image_name = row["image_name"]
    image_path = os.path.join(IMAGE_DIR, f"{image_name}.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGE_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"  [WARN] Not found: {image_path} → 5")
        option = "5"
    else:
        # Using multi-pass reflection for higher accuracy
        option = predict_with_reflection(image_path)
        print(f"  {image_name} → {option}")
    results.append({"id": image_name, "image_name": image_name, "option": option})

submission = pd.DataFrame(results)
submission.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Saved {OUTPUT_CSV}")
print(submission["option"].value_counts().sort_index())
