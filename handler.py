"""
Gemma 4 E2B-it RunPod Handler — BigTooth Vision Review

Generic vision wrapper. Receives images + prompt, returns model output.
All business logic (prompts, criteria) is controlled by the caller.
Drop-in replacement for the Qwen2-VL handler — same input/output contract.
"""

import runpod
import torch
import base64
import gc
import json
import re
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E2B-it"

model = None
processor = None


def load_model():
    """Load Gemma 4 E2B-it at startup. Weights are baked into the image."""
    global model, processor

    print(f"Loading {MODEL_ID}...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA capability: sm_{cap[0]}{cap[1]}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded successfully ({param_count:.2f}B params).")


# Audit 13 Nov 2026 (ChatGPT review): photo addresses carry a download token, so they are never logged; only Firebase
# Storage hosts are fetched; downloads and decoded images are capped.
ALLOWED_PHOTO_HOSTS = ("firebasestorage.googleapis.com", "storage.googleapis.com")
MAX_PHOTO_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000  # ~6300x6300; phone photos are far smaller


def _host(url):
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname or "?"
    except Exception:
        return "?"


def download_image(url):
    host = _host(url) if isinstance(url, str) else "?"
    if not isinstance(url, str) or not url.startswith("https://") or host not in ALLOWED_PHOTO_HOSTS:
        print(f"Refused photo address (host {host}): only Firebase Storage is fetched")
        return None
    try:
        with requests.get(url, timeout=30, stream=True) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(64 * 1024):
                data.extend(chunk)
                if len(data) > MAX_PHOTO_BYTES:
                    print(f"Photo too large (> {MAX_PHOTO_BYTES // (1024 * 1024)} MB) from {host}")
                    return None
        return Image.open(BytesIO(bytes(data))).convert("RGB")
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", "?")
        print(f"Error downloading a photo from {host}: HTTP {status}")
        return None
    except Exception as e:
        print(f"Error downloading a photo from {host}: {type(e).__name__}")
        return None


def decode_base64_image(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def handler(job):
    """
    RunPod handler function.

    Input (matches Qwen2-VL handler exactly):
        photoUrls: list[str]        — image URLs (optional)
        photoBase64: list[str]      — base64 encoded images (optional)
        prompt: str                 — the caller's full prompt (required)

    Output:
        success: bool
        output: str                 — raw model output text
        parsed: dict                — parsed JSON if model returned valid JSON (optional)
        error: str                  — present only on failure
    """
    global model, processor

    job_input = job.get("input", {})

    photo_urls = job_input.get("photoUrls", [])
    photo_base64_list = job_input.get("photoBase64", [])
    prompt = job_input.get("prompt")

    if not prompt:
        return {"success": False, "error": "prompt is required"}

    if not photo_urls and not photo_base64_list:
        return {"success": False, "error": "Either photoUrls or photoBase64 is required"}

    images = []
    if photo_base64_list:
        for b64 in photo_base64_list:
            img = decode_base64_image(b64)
            if img:
                images.append(img)
    else:
        for url in photo_urls:
            img = download_image(url)
            if img:
                images.append(img)

    if len(images) == 0:
        return {"success": False, "error": "No images could be loaded"}

    # Build the chat-template message. Gemma 4 expects content items of
    # {"type": "image", "image": <PIL>} and {"type": "text", "text": ...}
    # interleaved. Images first, then the caller's instruction last.
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    try:
        alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"VRAM before inference: {alloc_mb:.0f}MB allocated, {reserved_mb:.0f}MB reserved")

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
            )

        # Strip the prompt tokens — only decode the newly generated portion.
        generated_ids = generated_ids[:, input_len:]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"Model output: {output_text}")

        result = {"success": True, "output": output_text}

        # Extract JSON block if present — same regex as Qwen handler.
        json_match = re.search(r'\{[\s\S]*\}', output_text)
        if json_match:
            try:
                result["parsed"] = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return result

    except RuntimeError as e:
        error_str = str(e)
        print(f"RuntimeError during inference: {error_str}")
        # CUDA errors leave GPU in an unrecoverable state — let worker crash
        # so RunPod restarts it with a clean CUDA context.
        if "CUDA" in error_str or "cuda" in error_str or "device-side assert" in error_str:
            print("Fatal CUDA error — letting worker crash for clean restart")
            raise
        return {"success": False, "error": error_str}

    except Exception as e:
        print(f"Error during inference: {e}")
        return {"success": False, "error": str(e)}

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


load_model()
runpod.serverless.start({"handler": handler})
