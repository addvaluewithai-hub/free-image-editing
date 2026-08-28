from __future__ import annotations

from io import BytesIO
from pathlib import Path

import modal

APP_NAME = "free-image-editing"
MAGE_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
GEN_MODEL_ID = "SceneWorks/Mage-Flow-Turbo"
EDIT_MODEL_ID = "SceneWorks/Mage-Flow-Edit-Turbo"
MODEL_ROOT = Path("/models")
GEN_MODEL_PATH = MODEL_ROOT / "Mage-Flow-Turbo"
EDIT_MODEL_PATH = MODEL_ROOT / "Mage-Flow-Edit-Turbo"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("mage-flow-models", create_if_missing=True)

# Microsoft validates Mage-Flow with Python 3.11, torch 2.13 / CUDA 12.6,
# transformers 5.5, diffusers 0.38 and flash-attn 2.8.3. We pin the upstream
# Mage source commit so a future upstream change cannot silently break deploys.
model_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "build-essential")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            # Modal's CUDA builder image can expose clang++ first. FlashAttention
            # must be compiled with the same compiler family as PyTorch on Linux.
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDAHOSTCXX": "/usr/bin/g++",
            # L4 is Ada (SM 8.9). Restricting the build avoids compiling kernels
            # for GPUs we will never use and makes the image much faster to build.
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "MAX_JOBS": "4",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel ninja packaging")
    .run_commands(
        "git clone https://github.com/microsoft/Mage.git /opt/Mage && "
        f"cd /opt/Mage && git checkout {MAGE_COMMIT}"
    )
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu126 "
        "torch==2.13.0 torchvision==0.28.0"
    )
    .run_commands(
        "pip install "
        "numpy==2.4.3 diffusers==0.38.0 transformers==5.5.0 "
        "accelerate==1.13.0 safetensors==0.8.0 'huggingface_hub[hf_xet]>=0.20' "
        "einops==0.8.2 pydantic==2.12.5 pillow==12.3.0 loguru==0.7.3 "
        "fastapi[standard] python-multipart"
    )
    .run_commands(
        "CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ "
        "TORCH_CUDA_ARCH_LIST=8.9 "
        "pip install --no-build-isolation flash-attn==2.8.3"
    )
    .run_commands("pip install --no-deps -e /opt/Mage/mage_flow")
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

api_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "fastapi[standard]", "python-multipart"
)


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(MODEL_ROOT): model_volume},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def download_models() -> dict[str, str]:
    """Download both Turbo checkpoints once into persistent Modal storage."""
    import os

    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]
    downloaded: dict[str, str] = {}
    for repo_id, path in (
        (GEN_MODEL_ID, GEN_MODEL_PATH),
        (EDIT_MODEL_ID, EDIT_MODEL_PATH),
    ):
        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(path), token=token)
        downloaded[repo_id] = str(path)

    model_volume.commit()
    return downloaded


def _validate_size(width: int, height: int) -> None:
    # Keeping the default ceiling conservative protects the Starter credit and
    # leaves VRAM headroom on the 24 GB L4. Mage itself supports up to 2048.
    if not (512 <= width <= 1536 and 512 <= height <= 1536):
        raise ValueError("width and height must be between 512 and 1536")
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")


@app.cls(
    image=model_image,
    gpu="L4",
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=10 * 60,
)
class Generator:
    @modal.enter()
    def load(self) -> None:
        from mage_flow import MageFlowPipeline

        if not GEN_MODEL_PATH.exists():
            raise RuntimeError("Generation weights are missing. Run download_models first.")
        self.pipe = MageFlowPipeline.from_pretrained(str(GEN_MODEL_PATH), device="cuda")

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
    ) -> bytes:
        _validate_size(width, height)
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")

        image = self.pipe.generate(
            [prompt],
            heights=[height],
            widths=[width],
            seeds=[seed],
            steps=4,
            cfg=1.0,
        )[0]
        out = BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()


@app.cls(
    image=model_image,
    gpu="L4",
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=10 * 60,
)
class Editor:
    @modal.enter()
    def load(self) -> None:
        from mage_flow import MageFlowPipeline

        if not EDIT_MODEL_PATH.exists():
            raise RuntimeError("Editing weights are missing. Run download_models first.")
        self.pipe = MageFlowPipeline.from_pretrained(str(EDIT_MODEL_PATH), device="cuda")

    @modal.method()
    def edit(
        self,
        image_bytes: bytes,
        prompt: str,
        max_size: int = 1024,
        seed: int = 42,
    ) -> bytes:
        from PIL import Image

        if not (512 <= max_size <= 1536):
            raise ValueError("max_size must be between 512 and 1536")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")

        source = Image.open(BytesIO(image_bytes)).convert("RGB")
        result = self.pipe.edit(
            [prompt],
            [source],
            max_size=max_size,
            seeds=[seed],
            steps=4,
            cfg=1.0,
        )[0]
        out = BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()


@app.function(image=api_image, timeout=12 * 60)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    """Private HTTP API. Modal rejects unauthenticated traffic before compute starts."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    web = FastAPI(
        title="Mage-Flow on Modal",
        version="1.0.0",
        description="Private 4-step Mage-Flow Turbo generation + image editing API.",
    )

    class GenerateRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=4000)
        width: int = 1024
        height: int = 1024
        seed: int = 42

    @web.get("/")
    def root():
        return {
            "status": "ok",
            "generation_model": GEN_MODEL_ID,
            "editing_model": EDIT_MODEL_ID,
            "gpu": "L4",
            "docs": "/docs",
            "auth": "Modal proxy authentication required",
        }

    @web.post("/generate")
    def generate_image(req: GenerateRequest):
        try:
            data = Generator().generate.remote(
                req.prompt, req.width, req.height, req.seed
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="image/png")

    @web.post("/edit")
    def edit_image(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        max_size: int = Form(1024),
        seed: int = Form(42),
    ):
        try:
            source = image.file.read()
            if not source:
                raise ValueError("uploaded image is empty")
            data = Editor().edit.remote(source, prompt, max_size, seed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="image/png")

    return web
