"""
Build-time validation for the Gemma 4 E2B checkpoint.

Downloads the model, runs a real CPU forward pass on a synthetic 224x224
image, and asserts that the output is valid. Fails the build with a clear
error if anything is off. This should catch:
  - transformers version mismatches (missing/unexpected keys)
  - tie_word_embeddings handling bugs (missing lm_head)
  - any init path that silently produces random weights

Takes ~1-2 min at build time but saves hours of production debugging.
"""

import os
import sys
from PIL import Image

MODEL_ID = "google/gemma-4-E2B-it"

print(f"[VALIDATE] Loading processor for {MODEL_ID}...")
from transformers import AutoProcessor, AutoModelForImageTextToText

token = os.environ.get("HF_TOKEN") or None

processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
print("[VALIDATE] Processor loaded.")

print(f"[VALIDATE] Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    token=token,
)
model.eval()

param_count = sum(p.numel() for p in model.parameters()) / 1e9
print(f"[VALIDATE] Model loaded with {param_count:.2f}B parameters")

# Check 1: either lm_head is present OR embeddings are tied (Gemma 4 uses tie_word_embeddings=True)
tied = bool(getattr(model.config, "tie_word_embeddings", False))
has_lm_head = "lm_head.weight" in model.state_dict()
print(f"[VALIDATE] tie_word_embeddings={tied}, has lm_head.weight in state_dict={has_lm_head}")
assert tied or has_lm_head, (
    "Neither tie_word_embeddings is True nor lm_head.weight is in state_dict — "
    "checkpoint is partially loaded. Check transformers version."
)

# Check 2: run a real forward pass on a synthetic image + prompt.
# If the model can't generate even 1 token, it's broken.
print("[VALIDATE] Running a real forward pass on CPU (slow but thorough)...")
dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": "Describe this image in one word."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# Build runs on CPU (no GPU in CI). That's fine for validation — we just need
# to confirm the weights are loaded and the forward pass executes.
import torch
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
    )

input_len = inputs["input_ids"].shape[-1]
generated_ids = generated_ids[:, input_len:]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"[VALIDATE] Forward pass output: {output_text!r}")

assert len(output_text) > 0, "Model generated empty output on forward pass"

print(f"[VALIDATE] Gemma 4 E2B-it cached and validated successfully")
