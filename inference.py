import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--image_dir", type=str, default=None,
                    help="Directory containing images (default: <test_dir>/images)")
parser.add_argument("--pdf_dir", type=str, default=None,
                    help="Optional: directory containing course PDF files for context retrieval")
args = parser.parse_args()

TEST_DIR   = args.test_dir
IMAGE_DIR  = args.image_dir if args.image_dir else os.path.join(TEST_DIR, "images")
TEST_CSV   = os.path.join(TEST_DIR, "test.csv")
OUTPUT_CSV = "submission.csv"

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
print(f"Loading {MODEL_NAME} ...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "45GiB"}
)
model.eval()

# High resolution enables the model to read small text perfectly
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256*28*28,
    max_pixels=1280*28*28,
)
print("Model loaded.\n")


def call_model(messages, max_new_tokens=2048, thinking=True):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if thinking:
        # Activate Qwen3 extended reasoning: the model generates a <think>...</think>
        # scratchpad before its answer. The thinking tokens (IDs 151667/151668) are in
        # the vocabulary but apply_chat_template doesn't expose enable_thinking in this
        # transformers version, so we inject the opening token manually.
        text = text + "<think>\n"

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
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    # skip_special_tokens=False preserves <think>/</think> tags for answer extraction
    return processor.batch_decode(generated, skip_special_tokens=False)[0].strip()


def extract_answer(response: str) -> str:
    # Isolate text after </think> so digits inside the thinking trace are ignored
    think_end = response.rfind("</think>")
    if think_end != -1:
        answer_text = response[think_end + len("</think>"):]
    else:
        answer_text = response

    # Strip Qwen special tokens (<|im_end|> etc.) from the answer region
    answer_text = re.sub(r"<\|[^|]+\|>", "", answer_text).strip()

    # 1. Strict requested format
    m = re.search(r"Final Answer:\s*([1-4])", answer_text, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2. Natural language patterns: "the answer is B", "option 2", "select A", etc.
    m = re.search(
        r"(?:answer is|correct (?:answer|option) is|choose|select)\s*[:\-]?\s*([A-D1-4])",
        answer_text, re.IGNORECASE
    )
    if m:
        val = m.group(1).upper()
        return str("ABCD".index(val) + 1) if val in "ABCD" else val

    # 3. Last standalone digit 1-4 in post-think text only
    digits = re.findall(r"\b([1-4])\b", answer_text)
    if digits:
        return digits[-1]

    return "5"


# ── PDF Context Helpers ───────────────────────────────────────────────────────

def load_pdf_chunks(pdf_dir: str, chunk_words: int = 250, overlap: int = 50) -> list:
    chunks = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[WARN] PyMuPDF (fitz) not installed — PDF context disabled")
        return chunks

    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(pdf_dir, fname)
        try:
            doc = fitz.open(fpath)
            text = " ".join(page.get_text() for page in doc)
            words = text.split()
            before = len(chunks)
            for i in range(0, len(words), chunk_words - overlap):
                chunk = " ".join(words[i:i + chunk_words])
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
            print(f"  {fname}: {len(words)} words → {len(chunks) - before} chunks")
        except Exception as e:
            print(f"  [WARN] Could not read {fname}: {e}")
    return chunks


def get_question_text(image) -> str:
    """Lightweight OCR pass to extract question text for chunk retrieval."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Transcribe the question and answer options from this image as plain text. Be exact and concise."}
    ]}]
    raw = call_model(messages, max_new_tokens=150, thinking=False)
    # Clean any residual special tokens from the short OCR response
    return re.sub(r"<[^>]+>", "", raw).strip()


def retrieve_chunks(query: str, chunks: list, top_k: int = 2) -> str:
    if not chunks:
        return ""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = [
        (sum(1 for w in query_words if w in chunk.lower()), chunk)
        for chunk in chunks
    ]
    scored.sort(reverse=True)
    # Require at least 3 keyword matches to avoid injecting irrelevant text
    top = [c for score, c in scored[:top_k] if score >= 3]
    return "\n---\n".join(top) if top else ""


# ── Load PDF Chunks at Startup ────────────────────────────────────────────────

pdf_chunks = []
if args.pdf_dir and os.path.isdir(args.pdf_dir):
    print(f"Loading PDFs from {args.pdf_dir} ...")
    pdf_chunks = load_pdf_chunks(args.pdf_dir)
    print(f"Loaded {len(pdf_chunks)} chunks from course PDFs.\n")
elif args.pdf_dir:
    print(f"[WARN] --pdf_dir '{args.pdf_dir}' not found, skipping PDF context.\n")


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert professor in Deep Learning and Computer Vision. This is a graduate-level Deep Learning examination.

=== DEEP LEARNING REFERENCE ===
CONV / POOL OUTPUT SIZE (applies to Conv2d AND MaxPool2d / AvgPool2d):
  out = floor((W + 2*P - F) / S) + 1   ← the +1 is MANDATORY; omitting it is the #1 error
  Equivalent shortcut when P=0 and S=F: out = floor(W / S)  (e.g. MaxPool k=2,s=2 on 64 → 32)
  Verify: always substitute actual numbers and confirm the +1 before selecting an option.

RECEPTIVE FIELD:
  Start with rf=1. For each layer l (conv or pool with kernel k_l, stride s_l):
    rf = rf + (k_l - 1) * product(s_i for i < l)
  Example: two Conv(k=3,s=1) layers: rf=1 → 1+(3-1)*1=3 → 3+(3-1)*1=5
  After a stride-2 layer, subsequent kernels cover a larger input region.

PYTORCH LAYERS & TENSOR SHAPES:
  nn.Linear(in, out): input (N, *, in) → output (N, *, out); params = in*out + out
  nn.Conv2d(in_ch, out_ch, k, s=1, p=0): input (N, in_ch, H, W) → output (N, out_ch, H', W')
  nn.Flatten(start_dim=1): (N, C, H, W) → (N, C*H*W)
  nn.BatchNorm2d(C): normalizes over (N, H, W) per channel; running stats at inference
  nn.LayerNorm(D): normalizes over feature dimension D per sample
  nn.Dropout(p): drops each neuron with prob p at TRAINING; at inference no dropout is applied
  nn.CrossEntropyLoss: expects RAW LOGITS (not softmax); internally applies log-softmax
  torch.view(N, -1) vs flatten: both reshape but view requires contiguous memory

PYTORCH CODE TRACING — for any code question, trace shapes layer by layer:
  1. Write down input shape (N, C, H, W) or (N, features)
  2. Apply each layer's transformation using the formulas above
  3. Check the final output shape or value matches the claimed option

DROPOUT (precise definition):
  Training: each unit kept with prob (1-p), scaled by 1/(1-p) [inverted dropout — no test change needed]
  Inference: dropout layer is a no-op (model.eval() disables it automatically in PyTorch)
  MC Dropout: keeping dropout ON at inference to approximate Bayesian uncertainty estimation
  Spatial Dropout (Dropout2d): drops entire channels rather than individual neurons

GAN (Generative Adversarial Network):
  Discriminator D maximizes: E[log D(x_real)] + E[log(1 - D(G(z)))]
  Generator G minimizes:     E[log(1 - D(G(z)))]  ≡ maximizes E[log D(G(z))]  (non-saturating)
  Mode collapse: G produces limited variety, ignoring parts of data distribution
  WGAN: uses Wasserstein distance, requires Lipschitz constraint (weight clipping or gradient penalty)
  cGAN / Pix2Pix: conditioning G and D on class label or paired image

VAE (Variational Autoencoder):
  Objective (ELBO): L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    → Reconstruction term + regularisation term (KL to standard normal prior)
  Reparameterization trick: z = μ + σ * ε,  ε ~ N(0, I)  — allows gradients to flow through z
  p(z) = N(0, I);  q(z|x) = N(μ(x), σ²(x))  — encoder outputs μ and σ
  VAE latent space is continuous and smooth; AE latent space is not regularised

ATTENTION (scaled dot-product):
  Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_k)) · V

ACTIVATIONS:
  ReLU, GELU, ELU, Leaky ReLU — mitigate vanishing gradient (do not saturate for x>0)
  Sigmoid, Tanh — saturate at extremes → cause vanishing gradient in deep networks

BACKPROP: loss gradient w.r.t. weight = upstream_grad × local_grad  (chain rule)
WEIGHT INIT: Xavier/Glorot for Sigmoid/Tanh; He/Kaiming for ReLU variants
=== END REFERENCE ===

INSTRUCTIONS:
1. The options in the image are labeled A, B, C, D — map them unconditionally to 1, 2, 3, 4.
2. Think step-by-step. For computation questions, apply the reference formulas above and substitute the actual numbers. Verify your result before concluding.
3. After your reasoning, state your final answer EXACTLY on a new line as:
Final Answer: <digit>
where <digit> is 1, 2, 3, or 4, or 5 if you are genuinely uncertain (wrong answers carry a penalty).
You MUST always output this line — never end without it."""


def build_prompt(context: str = "") -> str:
    if context:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== COURSE REFERENCE MATERIAL ===\n{context}\n=== END COURSE MATERIAL ==="
        )
    return SYSTEM_PROMPT


def predict(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # Retrieve course-specific context if PDFs were loaded
    context = ""
    if pdf_chunks:
        question_text = get_question_text(image)
        context = retrieve_chunks(question_text, pdf_chunks)
        if context:
            print(f"  [PDF context found, {len(context)} chars]")

    prompt = build_prompt(context)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]

    response = call_model(messages, max_new_tokens=2048, thinking=True)

    # Log enough to verify thinking mode is active and extraction is correct
    think_end = response.rfind("</think>")
    if think_end != -1:
        print(f"  [Think] {response[:200].strip()}...")
        print(f"  [Post-think] ...{response[think_end:think_end + 200].strip()}")
    else:
        print(f"  [Response] {response[:500]}")

    return extract_answer(response)


# ── Initialization & Resume Logic ─────────────────────────────────────────────

if os.path.exists(TEST_CSV):
    df = pd.read_csv(TEST_CSV)
    print(f"Found {len(df)} test samples in CSV.")
else:
    print(f"CSV not found at {TEST_CSV}. Processing all images in {IMAGE_DIR}...")
    if os.path.exists(IMAGE_DIR):
        img_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        df = pd.DataFrame({"image_name": [os.path.splitext(f)[0] for f in img_list]})
    else:
        img_list = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        df = pd.DataFrame({"image_name": [os.path.splitext(f)[0] for f in img_list]})
        IMAGE_DIR = TEST_DIR

if os.path.exists(OUTPUT_CSV):
    print(f"Found existing {OUTPUT_CSV}. Resuming from previous state...")
    existing_sub = pd.read_csv(OUTPUT_CSV)
    existing_sub["option"] = existing_sub["option"].astype(str)
    submission = df.copy()
    submission["option"] = "5"
    old_answers = dict(zip(existing_sub["image_name"], existing_sub["option"]))
    submission["option"] = submission["image_name"].apply(lambda x: old_answers.get(x, "5"))
else:
    print(f"Creating new {OUTPUT_CSV} initialized with '5'...")
    submission = df.copy()
    submission["option"] = "5"

submission = submission[["image_name", "option"]]
submission.to_csv(OUTPUT_CSV, index=False)


# ── Main Inference Loop ───────────────────────────────────────────────────────

for index, row in tqdm(submission.iterrows(), total=len(submission), desc="Predicting"):
    image_name = row["image_name"]
    current_option = str(row["option"])

    # Skip items we've already successfully predicted
    if current_option in ["1", "2", "3", "4"]:
        continue

    image_path = os.path.join(IMAGE_DIR, f"{image_name}.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"  [WARN] Not found: {image_path} → 5")
        option = "5"
    else:
        try:
            option = predict(image_path)
            print(f"  {image_name} → Option {option}")
        except Exception as e:
            print(f"  [ERROR] Failed processing {image_name}: {str(e)}")
            option = "5"

    submission.at[index, "option"] = option
    submission.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone! Final results saved to {OUTPUT_CSV}")
print(submission["option"].value_counts().sort_index())
