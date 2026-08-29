from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import modal

APP_NAME = "sd-turbo-simple-images"
MODEL_ID = "stabilityai/sd-turbo"
MODEL_ROOT = Path("/models")
MODEL_PATH = MODEL_ROOT / "sd-turbo"
T4_GPU_USD_PER_SECOND = 0.000164
WIDTH = 512
HEIGHT = 512

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("sd-turbo-models", create_if_missing=True)

# Download on CPU so Hugging Face transfer time does not burn GPU credits.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1"})
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1"
    )
    .run_commands(
        "pip install diffusers==0.35.1 transformers==4.55.4 accelerate==1.10.1 "
        "safetensors==0.6.2 sentencepiece==0.2.1 pillow==11.3.0"
    )
)

PROMPTS = [
    (
        "01-apple",
        "A single fresh red apple centered on a seamless white studio background, clean product photograph, soft natural shadow, no other objects.",
    ),
    (
        "02-car",
        "A blue compact hatchback car shown in clean three-quarter front view on a neutral light gray studio background, automotive catalog photograph, realistic proportions, no people.",
    ),
    (
        "03-chair",
        "A handcrafted oak dining chair centered in a warm minimalist room, soft window light from the left, realistic wood grain, clean interior photography.",
    ),
    (
        "04-fruit-bowl",
        "A matte ceramic bowl containing three red apples, two yellow bananas and two oranges on a wooden kitchen table, morning window light, realistic food photography, uncluttered background.",
    ),
    (
        "05-bicycle-street",
        "A red city bicycle parked beside a young green tree on a quiet European-style sidewalk, sunny late afternoon, a few simple building facades in the background, natural shadows, realistic street photography.",
    ),
    (
        "06-cat-window",
        "A fluffy orange tabby cat sitting on a wide windowsill beside a small green potted plant, looking outside, warm daylight entering through the window, cozy apartment interior, detailed fur and believable lighting.",
    ),
    (
        "07-coffee-desk",
        "A carefully arranged cozy work desk seen from a slightly elevated angle: an open cream notebook, black pen, white ceramic coffee mug with visible steam, folded eyeglasses and a small brass desk lamp, warm evening light, realistic materials and shadows.",
    ),
    (
        "08-kitchen-scene",
        "A bright modern residential kitchen photographed at eye level, containing a stainless refrigerator, pale wooden dining table, exactly two chairs, a black kettle, a ceramic fruit bowl and a large sunlit window; coherent room geometry, realistic perspective, clean contemporary design.",
    ),
    (
        "09-market-scene",
        "A lively but organized outdoor fruit market stall under a red-and-white striped canopy, baskets clearly filled with apples, bananas and oranges; one smiling shopkeeper behind the counter and exactly two customers in front, sunny morning, believable human poses, coherent object placement, documentary photography.",
    ),
    (
        "10-hard-scene",
        "A challenging photorealistic tabletop composition: a glossy red toy sports car facing left on a dark walnut table, a green apple positioned behind the car, and a half-full clear glass of water to its right. A small round mirror leans against the wall and must show a faint reflection of the red toy car. Behind everything is a large window covered with visible rain droplets and a softly blurred rainy city outside. Warm desk light from the left mixes with cool window light from the right, producing physically believable reflections in the mirror, glass, water and glossy car paint, accurate contact shadows, correct perspective, no text or logos.",
    ),
]


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(MODEL_ROOT): model_volume},
    timeout=60 * 30,
    cpu=2,
    memory=4096,
)
def download_model() -> str:
    import os
    from huggingface_hub import snapshot_download

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_PATH),
        token=os.environ.get("HF_TOKEN"),
    )
    model_volume.commit()
    return str(MODEL_PATH)


@app.cls(
    image=inference_image,
    gpu="T4",
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=10 * 60,
    memory=12288,
)
class SDTurboGenerator:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        if not MODEL_PATH.exists():
            raise RuntimeError("SD-Turbo weights missing. Run download_model first.")

        self.device = "cuda"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @modal.method()
    def generate(self, prompt: str, seed: int) -> bytes:
        import io
        import torch

        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=WIDTH,
                height=HEIGHT,
                generator=generator,
            ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def benchmark(output_dir: str = "sd-turbo-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    generator = SDTurboGenerator()
    durations: list[float] = []
    records: list[dict] = []
    overall_start = time.perf_counter()

    for index, (slug, prompt) in enumerate(PROMPTS, start=1):
        started = time.perf_counter()
        data = generator.generate.remote(prompt, 66000 + index)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)

        path = target / f"{index:02d}-{slug}.png"
        path.write_bytes(data)
        records.append(
            {
                "difficulty": index,
                "slug": slug,
                "seconds": elapsed,
                "bytes": len(data),
                "prompt": prompt,
            }
        )
        print(
            f"SD_TURBO_IMAGE difficulty={index} slug={slug} "
            f"seconds={elapsed:.3f} bytes={len(data)}"
        )

    total_wall = time.perf_counter() - overall_start
    warm = durations[1:]
    estimated_gpu_cost = total_wall * T4_GPU_USD_PER_SECOND
    warm_cost_per_image = statistics.mean(warm) * T4_GPU_USD_PER_SECOND

    result = {
        "experiment": "sd-turbo-t4-graded-difficulty-10",
        "model": MODEL_ID,
        "gpu": "T4",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "steps": 1,
        "image_count": len(PROMPTS),
        "first_call_seconds": durations[0],
        "warm_average_seconds": statistics.mean(warm),
        "warm_min_seconds": min(warm),
        "warm_max_seconds": max(warm),
        "total_wall_seconds": total_wall,
        "t4_gpu_usd_per_second": T4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_upper_bound": estimated_gpu_cost,
        "estimated_usd_per_image_including_cold_start": estimated_gpu_cost / len(PROMPTS),
        "estimated_warm_usd_per_image": warm_cost_per_image,
        "estimated_warm_images_per_30_gpu_only": 30.0 / warm_cost_per_image,
        "scenes": records,
        "note": "Model weights are pre-downloaded on CPU to a Modal Volume so download time does not burn T4 GPU credits. Cost estimate uses end-to-end remote-call wall time times Modal's T4 GPU rate and is conservative; CPU/RAM/storage charges are separate and small.",
    }

    (target / "benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("SD_TURBO_JSON=" + json.dumps(result, separators=(",", ":")))
