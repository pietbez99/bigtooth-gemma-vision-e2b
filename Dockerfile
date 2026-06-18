# BigTooth — Gemma 4 E2B Vision Review RunPod Endpoint
# Replaces Qwen2-VL-7B-Instruct for brushing session review
# Model: google/gemma-4-E2B-it (Apache 2.0 license, ~2B active / ~5B effective params)
# Model is baked into the image at build time to avoid cold-start downloads.
#
# CUDA 12.8 / sm_120 (Blackwell) support: RunPod has silently started
# allocating Blackwell hardware (RTX 5090, PRO 6000) under existing tier
# codes that used to mean Ampere/Ada-only. The previous cu124 image crashed
# immediately on any Blackwell allocation with "no kernel image available
# for execution on the device". This build targets cu128 to support all
# current Nvidia generations from Ampere through Blackwell.

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
# Refuse runtime HF downloads. If the baked cache is missing, fail loudly at
# load time instead of silently re-downloading and crashing later.
ENV HF_HUB_OFFLINE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# transformers must be >=5.5.0 — that's the first stable release with full
# Gemma 4 support. Older versions (4.x) produce a partially-initialized model
# with missing lm_head weights and random attention layers.
#
# Note: torch is provided by the base image (RunPod's cu128 base ships
# PyTorch 2.8.0 prebuilt for sm_90 + sm_120). We do NOT pip install torch
# explicitly to avoid risking a wheel-version mismatch with the base.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    "transformers>=5.5.0,<5.6" \
    "accelerate>=0.33.0" \
    "safetensors>=0.4.0" \
    "sentencepiece>=0.1.99" \
    "pillow>=10.0.0" \
    "requests>=2.31.0" \
    "runpod>=1.6.0"

# HF_TOKEN is passed in as a build arg from GitHub Actions (see .github/workflows/build.yml).
# Used only during the pre-download step below. Docker build args are not
# persisted to the runtime environment unless explicitly ENVed.
ARG HF_TOKEN=""

# Pre-download Gemma 4 E2B-it into the image AND validate it loads cleanly.
# HF_HUB_OFFLINE is temporarily overridden to 0 so the download can happen.
# The validation step runs a real CPU forward pass to confirm the weights
# are structurally sound before the image is pushed.
COPY validate_model.py /tmp/validate_model.py
RUN HF_TOKEN="${HF_TOKEN}" HF_HUB_OFFLINE=0 python /tmp/validate_model.py && rm /tmp/validate_model.py

# Verify the cache is actually baked into the image layers.
RUN du -sh /root/.cache/huggingface && ls -la /root/.cache/huggingface/hub/

COPY handler.py /app/handler.py

# Diagnostic CMD: print env state before invoking the handler so we can see
# WHERE in startup the container is dying. Previous build (commit 3726cda)
# silently failed before producing any Python output — likely a python-path
# or torch-import problem with the new cu128 base image.
#
# Once we have a clean working baseline, this can be simplified back to:
#   CMD ["python", "-u", "/app/handler.py"]
CMD ["/bin/bash", "-c", "\
echo '=== [BOOT] container start ==='; \
echo '=== [BOOT] uname ==='; uname -a; \
echo '=== [BOOT] python locations ==='; \
which python python3 python3.11 python3.12 2>&1 || true; \
echo '=== [BOOT] python --version ==='; \
python --version 2>&1 || echo 'python NOT FOUND'; \
python3 --version 2>&1 || echo 'python3 NOT FOUND'; \
echo '=== [BOOT] torch import test ==='; \
python -c 'import torch; print(f\"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}\"); print(f\"cuda_version_compiled={torch.version.cuda}\")' 2>&1 || python3 -c 'import torch; print(f\"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}\"); print(f\"cuda_version_compiled={torch.version.cuda}\")' 2>&1 || echo 'torch IMPORT FAILED'; \
echo '=== [BOOT] transformers import test ==='; \
python -c 'import transformers; print(f\"transformers={transformers.__version__}\")' 2>&1 || python3 -c 'import transformers; print(f\"transformers={transformers.__version__}\")' 2>&1 || echo 'transformers IMPORT FAILED'; \
echo '=== [BOOT] runpod import test ==='; \
python -c 'import runpod; print(f\"runpod={runpod.__version__ if hasattr(runpod, \\\"__version__\\\") else \\\"<no version attr>\\\"}\")' 2>&1 || python3 -c 'import runpod; print(f\"runpod={runpod.__version__ if hasattr(runpod, \\\"__version__\\\") else \\\"<no version attr>\\\"}\")' 2>&1 || echo 'runpod IMPORT FAILED'; \
echo '=== [BOOT] handler.py exists check ==='; \
ls -la /app/handler.py 2>&1; \
echo '=== [BOOT] handing off to handler ==='; \
exec python -u /app/handler.py 2>&1 || exec python3 -u /app/handler.py 2>&1 || { echo 'FAILED TO LAUNCH HANDLER'; sleep 5; exit 1; }\
"]
