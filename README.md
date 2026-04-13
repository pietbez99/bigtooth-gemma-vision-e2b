# BigTooth — Gemma 4 E2B Vision Review

RunPod serverless endpoint for AI-reviewing children's toothbrushing session photos using `google/gemma-4-E2B-it`.

## What this is

- Drop-in replacement for Qwen2-VL-7B-Instruct on the brushing-session review pipeline
- Model baked into the Docker image at build time (no runtime HF downloads)
- Validated at build time with a real forward pass — broken checkpoints fail the build, not production
- Runtime is `HF_HUB_OFFLINE=1` — the container refuses to download anything from HuggingFace, so a missing bake cache fails loudly at startup

## Why E2B, not E4B

E2B is roughly half the size of E4B (~2B active params vs ~4B). For a simple "is there a toothbrush in the mouth" classification task with a lenient prompt, the quality difference is expected to be small, and the operational benefits are large:

- Fits on 24 GB GPUs (RunPod `AMPERE_24` tier — A5000 / 3090, always in high supply)
- Smaller image → faster cold starts and GHCR pulls
- Cheaper per-request

A separate E4B repo (`bigtooth-gemma-vision`) exists if we ever need the bigger model. Start with E2B, escalate only if quality proves inadequate.

## Tech notes

- **Base image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **PyTorch**: 2.5.1 cu124 (supports sm_80 through sm_90 — Ampere, Ada, Hopper)
- **transformers**: `>=5.5.0,<5.6` (first stable release with full Gemma 4 support; 4.x produces partial loads)
- **auto_model**: `AutoModelForImageTextToText` (per the HF model card)

## API

Input (matches the existing Qwen/Gemma handlers exactly):

```json
{
  "input": {
    "photoUrls": ["https://...", "..."],
    "prompt": "You are reviewing 4 photos from a children's toothbrushing app..."
  }
}
```

Output:

```json
{
  "success": true,
  "output": "raw model text...",
  "parsed": { "faceDetected": true, "toothbrushVisible": true, "toothbrushInMouth": true, "confidence": 0.85 }
}
```

## Build & deploy

Push to `main` → GitHub Actions builds and pushes `ghcr.io/pietbez99/bigtooth-gemma-vision-e2b:latest` + `:sha-<commit>`.

Requires `HF_TOKEN` repo secret.
