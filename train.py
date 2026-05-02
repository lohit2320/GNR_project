import os
import json
from datasets import Dataset
from PIL import Image
import torch

# ── Install check ─────────────────────────────────────────────────────────────
try:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
except ImportError:
    raise SystemExit("Run: pip install unsloth")

from trl import SFTTrainer
try:
    from trl import SFTConfig
except ImportError:
    from transformers import TrainingArguments as SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_JSONL  = "dataset/train.jsonl"
OUTPUT_DIR   = "lora_adapter"

# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r                = 16,
    lora_alpha       = 16,
    lora_dropout     = 0,
    bias             = "none",
    random_state     = 42,
)

# ── Load dataset ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert deep learning researcher. "
    "Look at this multiple choice question image about deep learning. "
    "Reply with ONLY one digit: 1, 2, 3, or 4."
)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

# Robust path detection
if not os.path.exists(TRAIN_JSONL):
    if os.path.exists("../" + TRAIN_JSONL):
        TRAIN_JSONL = "../" + TRAIN_JSONL
        DATASET_ROOT = "../dataset"
    elif os.path.exists("qwen/" + TRAIN_JSONL):
        TRAIN_JSONL = "qwen/" + TRAIN_JSONL
        DATASET_ROOT = "qwen/dataset"
    else:
        # If still not found, keep default and let it fail with clear error later
        DATASET_ROOT = "dataset"
else:
    DATASET_ROOT = "dataset"

print(f"Loading data from: {TRAIN_JSONL}")
raw = load_jsonl(TRAIN_JSONL)

def make_conversation(item):
    # The image path in JSONL is "images/xxx.png"
    # We need to prepend the dataset root folder
    full_image_path = os.path.join(DATASET_ROOT, item["image"])
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path},
                    {"type": "text",  "text": SYSTEM_PROMPT},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["answer"]}
                ]
            }
        ]
    }

conversations = [make_conversation(r) for r in raw]
dataset = Dataset.from_list(conversations)

def convert_sample(sample):
    messages = sample["messages"]
    # load image for user turn
    for msg in messages:
        if msg["role"] == "user":
            for block in msg["content"]:
                if block["type"] == "image":
                    img_path = block["image"]
                    if not os.path.exists(img_path):
                        print(f"Warning: image not found at {img_path}")
                    block["image"] = Image.open(img_path).convert("RGB")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text, "messages": messages}

dataset = dataset.map(convert_sample, num_proc=1)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        warmup_steps                = 20,
        logging_steps               = 10,
        save_steps                  = 100,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        remove_unused_columns       = False,
        report_to                   = "none",
        dataset_text_field          = "",
        dataset_kwargs              = {"skip_prepare_dataset": True},
    ),
)

print("Starting training...")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training done!")